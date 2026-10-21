"""Train a style LoRA for the *student* (SD-Tiny) on teacher-generated data.

This is the step that turns a generic tiny checkpoint into a model with a
reliable anime / 3D look: the student learns the style manifold of the teacher
while keeping its tiny footprint (a rank-16 LoRA is ~6 MB).

Usage:
    python tools/train_student_lora.py --data data/teacher --style anime \
        --out models/adapters/anime --steps 1200 --rank 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from cheapmage.engine import quantize as qz  # noqa: E402


def load_pairs(data_dir: Path, style: str) -> list[tuple[Path, str]]:
    style_dir = data_dir / style
    pairs = []
    for img in sorted(style_dir.glob("*.png")):
        cap = img.with_suffix(".txt")
        if cap.exists():
            pairs.append((img, cap.read_text(encoding="utf-8").strip()))
    return pairs


def build_lora(unet, rank: int, alpha: int) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    model = get_peft_model(unet, config)
    model.print_trainable_parameters()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "teacher"))
    ap.add_argument("--style", choices=["anime", "3d"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--package", default=str(REPO_ROOT / "models" / "cheapmage-v1"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trigger", default="")
    args = ap.parse_args()

    out_dir = Path(args.out or (REPO_ROOT / "models" / "adapters" / args.style))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    pkg = Path(args.package)
    tokenizer = CLIPTokenizer.from_pretrained(str(pkg / "tokenizer"))
    text_encoder = qz.load_component(
        pkg / "text_encoder", CLIPTextModel, dtype=torch.float16, device="cpu", config_cls=CLIPTextConfig
    )
    text_encoder.eval()
    vae = qz.load_component(pkg / "vae", AutoencoderKL, dtype=torch.float16, device=device)
    vae.eval()
    unet = qz.load_component(pkg / "unet", UNet2DConditionModel, dtype=torch.float16, device=device)

    pairs = load_pairs(Path(args.data), args.style)
    if not pairs:
        print(f"no training pairs found under {args.data}/{args.style}")
        return 1
    print(f"training pairs: {len(pairs)}")

    # ---- pre-compute VAE latents (once) --------------------------------
    latents_cache = out_dir / "latents.pt"
    if latents_cache.exists():
        latents = torch.load(latents_cache, map_location="cpu")
    else:
        latents = []
        with torch.no_grad():
            for path, _cap in pairs:
                w = h = args.size
                img = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
                import numpy as np

                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                lat = vae.encode(arr.unsqueeze(0).to(device, torch.float16)).latent_dist.mean
                latents.append((lat * vae.config.scaling_factor).squeeze(0).cpu())
        latents = torch.stack(latents)
        torch.save(latents, latents_cache)
    print(f"latents: {tuple(latents.shape)}")

    # ---- LoRA ------------------------------------------------------------
    unet = build_lora(unet, args.rank, args.alpha)
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    trigger = args.trigger or ("anime" if args.style == "anime" else "render")
    if trigger not in tokenizer.get_vocab():
        raise SystemExit(
            f"trigger '{trigger}' is not in the CLIP vocabulary - pick an existing token "
            "(the text encoder stays frozen, so no new tokens are added)"
        )

    losses: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            idx = rng.randrange(len(pairs))
            cap = pairs[idx][1]
            with torch.no_grad():
                tokens = tokenizer(
                    trigger + ", " + cap,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                emb = text_encoder(tokens.input_ids)[0].to(device, torch.float16)
            lat = latents[idx].unsqueeze(0).to(device, torch.float16)
            if rng.random() < 0.5:
                lat = torch.flip(lat, dims=[3])
            noise = torch.randn_like(lat)
            timesteps = torch.randint(0, 1000, (1,), device=device)
            noisy = torch.sqrt(1 - _betas(timesteps)) * lat + torch.sqrt(_betas(timesteps)) * noise
            pred = unet(noisy, timesteps, encoder_hidden_states=emb).sample
            loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(total)

        if step % 25 == 0:
            print(f"step {step}/{args.steps} loss {sum(losses[-25:]) / 25:.4f} "
                  f"({time.time() - t0:.0f}s)")
        if step % 400 == 0 or step == args.steps:
            adapter_dir = out_dir
            unet.save_pretrained(str(adapter_dir))
            _write_adapter_config(adapter_dir, trigger, args)

    report = {
        "style": args.style,
        "trigger": trigger,
        "steps": args.steps,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "dataset": str(args.data),
        "pairs": len(pairs),
        "loss_first_50": sum(losses[:50]) / max(1, min(50, len(losses))),
        "loss_last_50": sum(losses[-50:]) / max(1, min(50, len(losses))),
        "train_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _betas(timesteps: torch.Tensor) -> torch.Tensor:
    """SD 1.x scaled-linear beta schedule -> alpha_bar."""
    import numpy as np

    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    t = timesteps.clamp(0, 999).long()
    return alphas_cumprod[t].to(torch.float32).to(timesteps.device)


def _write_adapter_config(adapter_dir: Path, trigger: str, args) -> None:
    cfg = {
        "peft_type": "LORA",
        "task_type": "UNET2DConditionModel",
        "r": args.rank,
        "lora_alpha": args.alpha,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "base_model_name_or_path": "nota-ai/bk-sdm-tiny",
        "trigger_word": trigger,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
