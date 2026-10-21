"""Standalone smoke test for a quantised CheapMage package.

This file is copied into the shipped package as ``smoke_test.py``. It has
**no** CheapMage imports on purpose: the inference side can run it to prove
that the quantised artefacts load and generate.

    python smoke_test.py --package models/cheapmage-v1

It performs a short deterministic sampling run (PNDM, 4 steps, fixed seed),
decodes one image, writes ``smoke_test.png`` next to the package and prints
a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantised CheapMage smoke test")
    parser.add_argument("--package", default="models/cheapmage-v1")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompt", default="anime style, portrait of an adult character")
    parser.add_argument("--negative", default="lowres, blurry, watermark, text")
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch
    from diffusers import AutoencoderKL, PNDMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    package = Path(args.package)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(str(package / "tokenizer"))
    text_encoder = torch.load(
        str(package / "text_encoder" / "model.pt"), map_location="cpu", weights_only=False
    ).to(device).eval()
    unet = torch.load(
        str(package / "unet" / "diffusion_pytorch_model.pt"), map_location="cpu", weights_only=False
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(str(package / "vae")).to(device).eval()
    scheduler = PNDMScheduler.from_pretrained(str(package / "scheduler"))
    load_seconds = time.time() - started

    # The quantised Linear layers expect fp32 activations.
    vae.to(torch.float32)

    def encode(text: str) -> torch.Tensor:
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return text_encoder(tokens.input_ids.to(device))[0].to(torch.float32)

    cond = encode(args.prompt)
    uncond = encode(args.negative)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(
        (1, 4, args.height // 8, args.width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    scheduler.set_timesteps(args.steps, device=device)
    started = time.time()
    with torch.no_grad():
        for t in scheduler.timesteps:
            model_input = torch.cat([latents] * 2)
            model_input = scheduler.scale_model_input(model_input, t)
            noise_pred = unet(model_input, t, encoder_hidden_states=torch.cat([uncond, cond])).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + args.guidance * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = vae.decode(latents / vae.config.scaling_factor).sample
    sample_seconds = time.time() - started

    from PIL import Image
    import numpy as np

    array = (image[0].detach().float().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
    array = np.clip(array, 0, 255).astype("uint8")
    out_path = Path(args.out) if args.out else package / "smoke_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(out_path)

    summary = {
        "package": str(package),
        "device": str(device),
        "steps": args.steps,
        "guidance": args.guidance,
        "load_seconds": round(load_seconds, 2),
        "sample_seconds": round(sample_seconds, 2),
        "image": str(out_path),
        "image_bytes": out_path.stat().st_size,
        "ok": True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
