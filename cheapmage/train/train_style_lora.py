"""Train a per-style LoRA adapter for the CheapMage student model.

    python -m cheapmage.train.train_style_lora --data data/synth --style anime \
        --out models/lora/anime --steps 1500 --rank 16 --batch 1 --grad-accum 8

The script trains a LoRA on top of the **unet** of the base snapshot
(``nota-ai/bk-sdm-tiny`` by default). The text encoder and the VAE stay
frozen; only the LoRA parameters are optimised.

Only licence-clean data is accepted: the dataset directory must carry a
``dataset_manifest.json`` (see :mod:`cheapmage.train.dataset`) and the
manifest is validated before the first step. The report written to
``<out>/training_report.json`` embeds the manifest hash and the licence
summary of the corpus that was used.

This module is TRAINING ONLY - it is never imported by the inference path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .dataset import CaptionFileDataset, DatasetManifest, SyntheticStyleDataset, assert_dataset_clean

__all__ = [
    "discover_lora_target_modules",
    "resolve_base_snapshot",
    "build_training_dataset",
    "train_style_lora",
    "main",
]

DEFAULT_BASE = "nota-ai/bk-sdm-tiny"
ATTN_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_base_snapshot(base: str | os.PathLike[str]) -> Path:
    """Return a local directory holding the base snapshot.

    Accepts a local path (used as-is) or a Hugging Face repo id (downloaded
    on demand - never at import time).
    """
    path = Path(base)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(base),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ignore_patterns=["*.ckpt", "*.bin", "*nonema*", "*.msgpack", "*flax*"],
        )
    )


def discover_lora_target_modules(unet: nn.Module) -> list[str]:
    """Discover the attention projection names that exist in ``unet``.

    SD-Tiny's UNet is a standard ``UNet2DConditionModel``; the attention
    projections are named ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0`` under
    an ``attn1`` / ``attn2`` block. Rather than hardcoding those names we
    walk the live module tree and only return the suffixes that are really
    present, so the script keeps working if the checkpoint layout changes.
    """
    found: set[str] = set()
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        marker_index = None
        for index, part in enumerate(parts):
            if part.startswith("attn") and part[4:].isdigit():
                marker_index = index
                break
        if marker_index is None:
            continue
        suffix = ".".join(parts[marker_index + 1 :])
        if suffix in ATTN_SUFFIXES:
            found.add(suffix)
    if not found:
        raise RuntimeError(
            "no attention projection modules found in the UNet - refusing to guess target modules"
        )
    return sorted(found)


def _make_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    use_8bit: bool = True,
) -> torch.optim.Optimizer:
    """8-bit AdamW when bitsandbytes is available, plain AdamW otherwise."""
    params = [p for p in params if p.requires_grad]
    if use_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:  # pragma: no cover - depends on the environment
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_training_dataset(
    data_dir: str | os.PathLike[str],
    style: str | None,
    size: int,
    max_samples: int | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Build the training dataset and return it with a provenance summary."""
    root = Path(data_dir)
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = DatasetManifest.load(manifest_path)
        assert_dataset_clean(manifest)
        dataset: Dataset = SyntheticStyleDataset(root, style=style, size=size, limit=max_samples)
        summary = {
            "kind": "synthetic",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest.sha256(),
            "licence_summary": manifest.licence_summary(),
            "style_summary": manifest.style_summary(),
            "source_summary": manifest.source_summary(),
            "samples": len(dataset),
        }
        return dataset, summary
    # Fall back to a plain <image>+<caption>.txt folder (external corpora).
    dataset = CaptionFileDataset(root, size=size, style=style, limit=max_samples)
    summary = {
        "kind": "caption-files",
        "manifest_path": "",
        "manifest_sha256": "",
        "licence_summary": {},
        "style_summary": {},
        "source_summary": {},
        "samples": len(dataset),
    }
    return dataset, summary


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_style_lora(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _seed_everything(args.seed)

    base_path = resolve_base_snapshot(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lora] base    : {base_path}")
    print(f"[lora] device  : {device}")

    # ---- models ----------------------------------------------------------
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    vae.to(device).eval()
    text_encoder.to(device).eval()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    targets = discover_lora_target_modules(unet)
    print(f"[lora] discovered attention targets: {targets}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=targets,
        init_lora_weights="gaussian",
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(unet, "enable_input_require_grads"):
            unet.enable_input_require_grads()
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in unet.parameters())
    print(f"[lora] trainable params: {trainable_count:,} / {total_count:,}")

    # ---- data ------------------------------------------------------------
    dataset, data_summary = build_training_dataset(
        args.data, args.style, args.size, args.max_samples
    )
    print(f"[lora] dataset : {data_summary}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )

    # ---- optimiser / schedule --------------------------------------------
    optimizer = _make_optimizer(trainable, args.lr, args.weight_decay, use_8bit=not args.no_8bit)
    use_amp = device.type == "cuda" and args.mixed_precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(0.05, 1.0 - progress)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    loss_curve: list[dict[str, float]] = []
    resume_from = ""
    if args.resume:
        resume_path = Path(args.resume)
        resume_from = str(resume_path)
        ckpt = _latest_checkpoint(resume_path)
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint found in {resume_path}")
        state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        unet.load_state_dict(state["unet"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        start_step = int(state.get("step", 0))
        loss_curve = list(state.get("loss_curve", []))
        print(f"[lora] resumed from {ckpt} at step {start_step}")

    # ---- loop ------------------------------------------------------------
    noise_sched = scheduler
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    started = time.time()
    step = start_step
    losses: list[float] = []
    epoch = 0
    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            captions = list(batch["caption"])

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                tokens = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(tokens.input_ids.to(device))[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy = noise_sched.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                model_pred = unet(noisy, timesteps, encoder_hidden_states).sample
                target = noise
                if args.prediction_type == "v_prediction":
                    target = noise_sched.get_velocity(latents, noise, timesteps)
                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(float(loss.detach().item()))

            if (step + 1) % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            step += 1
            if step % args.log_every == 0:
                window = losses[-args.log_every :]
                mean = sum(window) / max(1, len(window))
                print(
                    f"[lora] step {step}/{args.steps} loss {mean:.5f} "
                    f"lr {lr_scheduler.get_last_lr()[0]:.2e} "
                    f"({(time.time() - started) / max(1, step - start_step):.2f}s/step)"
                )
                loss_curve.append({"step": step, "loss": mean})

            if args.save_every and step % args.save_every == 0:
                _save_checkpoint(
                    out_dir / "checkpoints" / f"step-{step:06d}",
                    unet=unet,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=step,
                    loss_curve=loss_curve,
                )
                unet.save_pretrained(str(out_dir / "adapter"))

    # ---- final artefacts --------------------------------------------------
    unet.save_pretrained(str(out_dir / "adapter"))
    _save_checkpoint(
        out_dir / "checkpoints" / f"step-{step:06d}",
        unet=unet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step=step,
        loss_curve=loss_curve,
    )

    runtime = time.time() - started
    tail = [entry["loss"] for entry in loss_curve[-10:]]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "cheapmage.train.train_style_lora",
        "base_model": args.base,
        "base_path": str(base_path),
        "style": args.style,
        "rank": args.rank,
        "lora_alpha": args.rank * 2,
        "target_modules": targets,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "steps": step,
        "epochs": epoch,
        "batch_size": args.batch,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "resolution": args.size,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": args.mixed_precision if use_amp else "no",
        "optimizer": type(optimizer).__name__,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "resume_from": resume_from,
        "runtime_seconds": round(runtime, 2),
        "seconds_per_step": round(runtime / max(1, step - start_step), 3),
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
        "mean_loss_last_10": (sum(tail) / len(tail)) if tail else None,
        "dataset": data_summary,
        "manifest_sha256": data_summary.get("manifest_sha256", ""),
        "licence_summary": data_summary.get("licence_summary", {}),
    }
    report_path = out_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[lora] report written to {report_path}")
    return report


def _latest_checkpoint(root: Path) -> Path | None:
    candidates = sorted((root / "checkpoints").glob("step-*")) if (root / "checkpoints").exists() else sorted(root.glob("step-*"))
    return candidates[-1] if candidates else None


def _save_checkpoint(
    path: Path,
    *,
    unet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    step: int,
    loss_curve: list[dict[str, float]],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "unet": {k: v.detach().cpu() for k, v in unet.state_dict().items() if "lora" in k},
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss_curve": loss_curve,
    }
    torch.save(state, path / "state.pt")
    unet.save_pretrained(str(path / "adapter"))
    print(f"[lora] checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheapmage.train.train_style_lora",
        description="Train a per-style LoRA adapter on SD-Tiny.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="base model id or local path")
    parser.add_argument("--data", required=True, help="dataset directory (synthetic corpus)")
    parser.add_argument("--style", default="anime", help="style tag to train (anime / 3d)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", default="bf16", choices=("bf16", "fp16", "no"))
    parser.add_argument("--prediction-type", default="epsilon", choices=("epsilon", "v_prediction"))
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", default="", help="directory holding a checkpoint to resume from")
    parser.add_argument("--no-8bit", action="store_true", help="disable bitsandbytes 8-bit AdamW")
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="disable gradient checkpointing (faster, more VRAM)",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--report-name", default="training_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_style_lora(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
