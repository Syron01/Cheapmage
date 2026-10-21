"""Few-step (LCM-style consistency) distillation of the CheapMage student.

    python -m cheapmage.train.distill_fewstep --data data/synth --style anime \
        --lora models/lora/anime/adapter --out models/distill/anime \
        --steps 1200 --num-steps 8 --teacher-steps 8 --guidance-scale 5.5

Method
------
* **Teacher** - the base UNet with the trained style LoRA merged in
  (frozen), sampled with a DPM++ multistep solver. The teacher produces an
  ODE trajectory ``(x_0 ... x_T, eps_0 ... eps_T)`` under classifier-free
  guidance with ``guidance_scale``.
* **Student** - a fresh copy of the UNet that must reproduce the teacher's
  denoised prediction ``x0`` *and* stay consistent across neighbouring
  steps of the trajectory (the LCM "consistency" term):

  ``L = mse(x0_student(x_k, t_k), x0_teacher(x_k, t_k))
      + lambda * mse(x0_student(x_{k+1}, t_{k+1}), stopgrad(x0_student(x_k, t_k)))``

* **CFG distillation** - the student is trained on the *guided* teacher
  output, so it bakes the guidance into its weights. At inference a single
  forward pass with a low guidance scale (~5-6) reproduces the behaviour of
  a many-step guided sampler.

The distilled UNet is written to ``<out>/unet`` together with a
``distillation_report.json``.

TRAINING ONLY - never imported by the inference path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CaptionFileDataset, DatasetManifest, SyntheticStyleDataset, assert_dataset_clean
from .train_style_lora import (
    DEFAULT_BASE,
    _seed_everything,
    build_training_dataset,
    resolve_base_snapshot,
)

__all__ = ["distill", "main"]


def _load_teacher(base_path: Path, lora_path: str | None, device: torch.device) -> torch.nn.Module:
    """Base UNet + (optional) merged style LoRA, frozen."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet")
    if lora_path:
        from peft import PeftModel

        adapter = Path(lora_path)
        if adapter.is_dir() and not (adapter / "adapter_config.json").exists():
            adapter = adapter / "adapter"
        unet = PeftModel.from_pretrained(unet, str(adapter))
        unet = unet.merge_and_unload()
    unet.to(device).eval()
    unet.requires_grad_(False)
    return unet


def _cfg_eps(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance: float,
) -> torch.Tensor:
    """Classifier-free guided epsilon prediction."""
    if guidance == 1.0:
        return unet(latents, timestep, cond).sample
    eps_cond = unet(latents, timestep, cond).sample
    eps_uncond = unet(latents, timestep, uncond).sample
    return eps_uncond + guidance * (eps_cond - eps_uncond)


def _to_x0(latents: torch.Tensor, eps: torch.Tensor, alphas: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    a = alphas[timestep].view(-1, *([1] * (latents.dim() - 1)))
    return (latents - (1 - a).sqrt() * eps) / a.sqrt()


def distill(args: argparse.Namespace) -> dict[str, Any]:
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _seed_everything(args.seed)
    base_path = resolve_base_snapshot(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ----------------------------------------------------------
    teacher = _load_teacher(base_path, args.lora, device)
    student = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(device)
    student.train()
    student.requires_grad_(True)
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
        student.enable_input_require_grads()

    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(device).eval()
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    solver = DPMSolverMultistepScheduler.from_pretrained(base_path, subfolder="scheduler")
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(base_path, subfolder="scheduler")
    alphas = noise_scheduler.alphas_cumprod.to(device)

    # ---- data ------------------------------------------------------------
    dataset, data_summary = build_training_dataset(
        args.data, args.style, args.size, args.max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and args.mixed_precision == "bf16"

    print(
        f"[distill] teacher=base{'+lora' if args.lora else ''} student=copy "
        f"steps={args.steps} num_steps={args.num_steps} teacher_steps={args.teacher_steps} "
        f"guidance={args.guidance_scale}"
    )

    # ---- teacher trajectory cache ----------------------------------------
    # A cached trajectory keeps its own conditioning so the image/​caption
    # pairing is never broken by the cache.
    traj_cache: dict[
        int, tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]
    ] = {}

    def teacher_trajectory(latents: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor):
        """Return (timesteps, latents list, eps list) of the teacher ODE."""
        solver.set_timesteps(args.teacher_steps, device=device)
        x = latents.clone()
        timesteps: list[torch.Tensor] = []
        xs: list[torch.Tensor] = []
        epss: list[torch.Tensor] = []
        for t in solver.timesteps:
            eps = _cfg_eps(teacher, x, t, cond, uncond, args.teacher_guidance)
            timesteps.append(t)
            xs.append(x.clone())
            epss.append(eps.detach())
            x = solver.step(eps, t, x).prev_sample
        return torch.stack(timesteps), xs, epss

    loss_curve: list[dict[str, float]] = []
    distill_losses: list[float] = []
    consist_losses: list[float] = []
    started = time.time()
    step = 0
    epoch = 0
    cache_keys = list(range(min(args.traj_cache, len(dataset))))

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            captions = list(batch["caption"])
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                tokens = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                cond = text_encoder(tokens.input_ids.to(device))[0]
                uncond_tokens = tokenizer(
                    [""],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                uncond = text_encoder(uncond_tokens.input_ids.to(device))[0].expand_as(cond)

            # teacher trajectory (optionally cached per dataset index)
            batch_sel = latents.shape[0]
            if args.traj_cache > 0 and cache_keys:
                key = random.choice(cache_keys)
                cached = traj_cache.get(key)
                if cached is None:
                    with torch.no_grad():
                        cached = (
                            *teacher_trajectory(latents, cond, uncond),
                            cond.detach(),
                            uncond.detach(),
                        )
                    traj_cache[key] = cached
                timesteps, xs, epss, cond, uncond = cached
                batch_sel = xs[0].shape[0]
            else:
                with torch.no_grad():
                    timesteps, xs, epss = teacher_trajectory(latents, cond, uncond)

            k = random.randrange(max(1, len(xs) - 1))
            t_k = timesteps[k].repeat(batch_sel)
            t_k1 = timesteps[k + 1].repeat(batch_sel)
            x_k = xs[k][:batch_sel]
            x_k1 = xs[k + 1][:batch_sel]
            eps_teacher = epss[k][:batch_sel]

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                eps_s_k = _cfg_eps(student, x_k, t_k, cond, uncond, args.guidance_scale)
                eps_s_k1 = _cfg_eps(student, x_k1, t_k1, cond, uncond, args.guidance_scale)
            x0_s_k = _to_x0(x_k, eps_s_k.float(), alphas, t_k)
            x0_s_k1 = _to_x0(x_k1, eps_s_k1.float(), alphas, t_k1)
            x0_t_k = _to_x0(x_k, eps_teacher.float(), alphas, t_k)

            distill_loss = F.mse_loss(x0_s_k, x0_t_k.detach())
            consist_loss = F.mse_loss(x0_s_k1, x0_s_k.detach())
            loss = distill_loss + args.lambda_consistency * consist_loss

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], args.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            distill_losses.append(float(distill_loss.detach().item()))
            consist_losses.append(float(consist_loss.detach().item()))
            step += 1
            if step % args.log_every == 0:
                window = max(1, args.log_every)
                mean_d = sum(distill_losses[-window:]) / min(window, len(distill_losses))
                mean_c = sum(consist_losses[-window:]) / min(window, len(consist_losses))
                print(
                    f"[distill] step {step}/{args.steps} distill {mean_d:.5f} "
                    f"consistency {mean_c:.5f} "
                    f"({(time.time() - started) / step:.2f}s/step)"
                )
                loss_curve.append(
                    {"step": step, "distill_loss": mean_d, "consistency_loss": mean_c}
                )
            if args.save_every and step % args.save_every == 0:
                _save_student(student, out_dir / "unet")

    _save_student(student, out_dir / "unet")

    runtime = time.time() - started
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "cheapmage.train.distill_fewstep",
        "base_model": args.base,
        "base_path": str(base_path),
        "style": args.style,
        "teacher": "base UNet" + (" + merged style LoRA" if args.lora else ""),
        "teacher_lora": args.lora or "",
        "student": "copy of the base UNet",
        "steps": step,
        "epochs": epoch,
        "num_steps": args.num_steps,
        "teacher_steps": args.teacher_steps,
        "teacher_guidance": args.teacher_guidance,
        "guidance_scale": args.guidance_scale,
        "lambda_consistency": args.lambda_consistency,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "resolution": args.size,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": "bf16" if use_amp else "no",
        "runtime_seconds": round(runtime, 2),
        "seconds_per_step": round(runtime / max(1, step), 3),
        "loss_curve": loss_curve,
        "final_distill_loss": distill_losses[-1] if distill_losses else None,
        "final_consistency_loss": consist_losses[-1] if consist_losses else None,
        "dataset": data_summary,
        "manifest_sha256": data_summary.get("manifest_sha256", ""),
        "licence_summary": data_summary.get("licence_summary", {}),
    }
    (out_dir / "distillation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[distill] report written to {out_dir / 'distillation_report.json'}")
    return report


def _save_student(student: torch.nn.Module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = student.training
    student.eval()
    student.to(torch.float16).save_pretrained(str(out_dir), safe_serialization=True)
    student.to(torch.float32)
    if was_training:
        student.train()
    print(f"[distill] student UNet saved to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheapmage.train.distill_fewstep",
        description="LCM-style few-step consistency distillation.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--data", required=True)
    parser.add_argument("--style", default="anime")
    parser.add_argument("--lora", default="", help="trained style LoRA adapter directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-steps", type=int, default=8, help="intended inference steps")
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--teacher-guidance", type=float, default=7.5)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", default="bf16", choices=("bf16", "no"))
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--traj-cache", type=int, default=64, help="cached teacher trajectories")
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distill(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
