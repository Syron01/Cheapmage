"""Build the shipped CheapMage package (quantized, < 1 GB).

Usage:
    python tools/build_package.py --base models/base_sdtiny --out models/cheapmage-v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from cheapmage.engine import quantize as qz  # noqa: E402

BUDGET_MB = 1000.0


def build(base: Path, out: Path, adapters: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # tokenizer + scheduler + model_index are copied verbatim
    for rel in ("tokenizer", "scheduler"):
        src = base / rel
        if src.exists():
            dst = out / rel
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    if (base / "model_index.json").exists():
        shutil.copy(base / "model_index.json", out / "model_index.json")

    report: dict = {"components": {}}

    # ---- UNet ------------------------------------------------------------
    from diffusers import UNet2DConditionModel

    unet_dir = base / "unet"
    unet = UNet2DConditionModel.from_pretrained(str(unet_dir), dtype=torch.float16)
    stats = qz.save_quantized_component(
        out / "unet",
        unet.state_dict(),
        config=json.loads((unet_dir / "config.json").read_text(encoding="utf-8")),
        linear_key_set=qz.linear_keys(unet),
    )
    report["components"]["unet"] = stats
    del unet

    # ---- text encoder ----------------------------------------------------
    from transformers import CLIPTextModel

    te_dir = base / "text_encoder"
    te = CLIPTextModel.from_pretrained(str(te_dir), dtype=torch.float16)
    stats = qz.save_quantized_component(
        out / "text_encoder",
        te.state_dict(),
        config=json.loads((te_dir / "config.json").read_text(encoding="utf-8")),
        linear_key_set=qz.linear_keys(te),
    )
    report["components"]["text_encoder"] = stats
    del te

    # ---- VAE (kept in FP16: decode quality) ------------------------------
    from diffusers import AutoencoderKL

    vae_dir = base / "vae"
    vae = AutoencoderKL.from_pretrained(str(vae_dir), dtype=torch.float16)
    vae.save_pretrained(str(out / "vae"))
    report["components"]["vae"] = {"fp16_mb": qz.package_size_mb(out / "vae"), "int8_mb": 0.0}
    del vae

    # ---- style adapters (trained LoRAs) ----------------------------------
    if adapters and adapters.exists():
        dst = out / "adapters"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(adapters, dst)
        report["adapters"] = sorted(p.name for p in dst.iterdir() if p.is_dir())

    total_mb = qz.package_size_mb(out)
    report["total_mb"] = round(total_mb, 1)
    report["budget_mb"] = BUDGET_MB
    report["within_budget"] = total_mb <= BUDGET_MB
    report["base_model"] = "nota-ai/bk-sdm-tiny (CreativeML OpenRAIL-M)"
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if total_mb > BUDGET_MB:
        print(f"ERROR: package is {total_mb:.1f} MB, budget is {BUDGET_MB} MB", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: package {total_mb:.1f} MB <= {BUDGET_MB} MB")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO_ROOT / "models" / "base_sdtiny"))
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "cheapmage-v1"))
    ap.add_argument("--adapters", default=str(REPO_ROOT / "models" / "adapters"))
    args = ap.parse_args()
    build(Path(args.base), Path(args.out), Path(args.adapters) if Path(args.adapters).exists() else None)
