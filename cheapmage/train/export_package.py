"""Build the shipped CheapMage package (``models/cheapmage-v1``).

    python -m cheapmage.train.export_package --base models/base_sdtiny \
        --adapters anime=models/lora/anime/adapter 3d=models/lora/3d/adapter \
        --characters models/characters --out models/cheapmage-v1

What the package contains
-------------------------
======================  ====================================================
``unet/``               INT8 dynamic-quantised UNet (qint8, per-channel)
``text_encoder/``       INT8 dynamic-quantised CLIP text encoder
``vae/``                fp16 VAE (fp16 keeps the decode quality)
``tokenizer/``          CLIP tokenizer files (verbatim copy)
``scheduler/``          scheduler configuration
``adapters/<style>/``   the trained style LoRA adapters (safetensors)
``characters/``         optional character (textual inversion) embeddings
``manifest.json``       sizes, SHA-256 and licences of every component
``smoke_test.py``       standalone test that runs the quantised package
======================  ====================================================

The total package size is printed and the script **fails loudly** when the
package exceeds 1000 MB.

TRAINING ONLY - never imported by the inference path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dataset import sha256_file

__all__ = ["export_package", "build_parser", "main"]

MAX_PACKAGE_MB = 1000.0

#: Licences of the shipped components. Overridable on the command line.
BASE_LICENCE = "CreativeML OpenRAIL-M"
BASE_COPYRIGHT = "CreativeML Open RAIL-M license (see base model card)"
LORA_LICENCE = "CC0-1.0 (trained by CheapMage on licence-clean synthetic data)"
CHARACTER_LICENCE = "CC0-1.0 (trained by CheapMage on licence-clean synthetic data)"

_TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


def _file_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        entries.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": item.stat().st_size,
                "mb": round(item.stat().st_size / (1024 * 1024), 4),
                "sha256": sha256_file(item),
            }
        )
    return entries


def _quantize_dynamic_int8(model: Any) -> Any:
    """Dynamic INT8 quantisation, per-channel symmetric weights.

    Linear layers become ``torch.ao.nn.quantized.dynamic.Linear`` with
    qint8 weights; activations stay dynamic (fp32 in / fp32 out).
    """
    import torch
    from torch.ao.quantization import (
        DynamicQuantStub,
        PerChannelMinMaxObserver,
        QConfig,
        quantize_dynamic,
    )

    qconfig = QConfig(
        activation=DynamicQuantStub(),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    return quantize_dynamic(
        model,
        {torch.nn.Linear},
        qconfig=qconfig,
        dtype=torch.qint8,
        inplace=False,
    )


def _parse_adapter_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--adapters entries must be '<style>=<path>', got {spec!r}")
    style, _, path = spec.partition("=")
    return style.strip(), Path(path.strip())


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_package(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from .train_style_lora import resolve_base_snapshot

    base_path = resolve_base_snapshot(args.base)
    out_dir = Path(args.out)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"{out_dir} already exists - pass --force to overwrite it"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] base   : {base_path}")
    print(f"[export] output : {out_dir}")

    components: list[dict[str, Any]] = []

    # ---- UNet -> INT8 ----------------------------------------------------
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet")
    unet.eval()
    unet_dir = out_dir / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    quantized_unet = _quantize_dynamic_int8(unet)
    torch.save(quantized_unet, unet_dir / "diffusion_pytorch_model.pt")
    (unet_dir / "config.json").write_text(
        json.dumps(
            dict(
                json.loads((base_path / "unet" / "config.json").read_text(encoding="utf-8")),
                _class_name="UNet2DConditionModel",
                quantization={
                    "scheme": "torch.ao.quantization.quantize_dynamic",
                    "dtype": "qint8",
                    "weight_qscheme": "per_channel_symmetric",
                    "activation": "dynamic",
                    "container": "torch.save(module)",
                    "loader": "torch.load(path, weights_only=False)",
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    components.append(
        {
            "component": "unet",
            "format": "int8-dynamic (qint8, per-channel weights)",
            "source": str(base_path / "unet"),
            "licence": BASE_LICENCE,
            "notes": BASE_COPYRIGHT,
        }
    )
    del quantized_unet, unet

    # ---- text encoder -> INT8 -------------------------------------------
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder")
    text_encoder.eval()
    te_dir = out_dir / "text_encoder"
    te_dir.mkdir(parents=True, exist_ok=True)
    quantized_te = _quantize_dynamic_int8(text_encoder)
    torch.save(quantized_te, te_dir / "model.pt")
    (te_dir / "config.json").write_text(
        (base_path / "text_encoder" / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    components.append(
        {
            "component": "text_encoder",
            "format": "int8-dynamic (qint8, per-channel weights)",
            "source": str(base_path / "text_encoder"),
            "licence": BASE_LICENCE,
            "notes": BASE_COPYRIGHT,
        }
    )
    del quantized_te, text_encoder

    # ---- VAE -> fp16 -----------------------------------------------------
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae")
    vae.eval().to(torch.float16)
    vae_dir = out_dir / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(vae_dir), safe_serialization=True)
    components.append(
        {
            "component": "vae",
            "format": "fp16 safetensors",
            "source": str(base_path / "vae"),
            "licence": BASE_LICENCE,
            "notes": BASE_COPYRIGHT,
        }
    )
    del vae

    # ---- tokenizer -------------------------------------------------------
    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    tokenizer.save_pretrained(str(tokenizer_dir))
    for name in _TOKENIZER_FILES:
        src = base_path / "tokenizer" / name
        if src.exists() and not (tokenizer_dir / name).exists():
            shutil.copy2(src, tokenizer_dir / name)
    components.append(
        {
            "component": "tokenizer",
            "format": "CLIP tokenizer (json/txt)",
            "source": str(base_path / "tokenizer"),
            "licence": BASE_LICENCE,
            "notes": BASE_COPYRIGHT,
        }
    )

    # ---- scheduler -------------------------------------------------------
    scheduler_dir = out_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        base_path / "scheduler" / "scheduler_config.json",
        scheduler_dir / "scheduler_config.json",
    )
    components.append(
        {
            "component": "scheduler",
            "format": "PNDM scheduler config",
            "source": str(base_path / "scheduler"),
            "licence": BASE_LICENCE,
            "notes": BASE_COPYRIGHT,
        }
    )

    # ---- model index -----------------------------------------------------
    (out_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "_diffusers_version": "0.41.0",
                "scheduler": ["diffusers", "PNDMScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "cheapmage": {
                    "unet": "unet/diffusion_pytorch_model.pt",
                    "text_encoder": "text_encoder/model.pt",
                    "vae": "vae/diffusion_pytorch_model.safetensors",
                    "tokenizer": "tokenizer",
                    "scheduler": "scheduler",
                    "adapters": "adapters",
                    "characters": "characters",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- style LoRA adapters --------------------------------------------
    adapters_dir = out_dir / "adapters"
    for spec in args.adapters:
        style, path = _parse_adapter_spec(spec)
        source = path / "adapter" if (path / "adapter").exists() else path
        target = adapters_dir / style
        target.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            src = source / name
            if not src.exists():
                raise FileNotFoundError(f"missing {name} in {source}")
            shutil.copy2(src, target / name)
        components.append(
            {
                "component": f"adapter:{style}",
                "format": "peft LoRA (safetensors)",
                "source": str(source),
                "licence": LORA_LICENCE,
                "notes": "trained by CheapMage on synthetic data only",
            }
        )
        print(f"[export] adapter {style} <- {source}")

    # ---- character embeddings -------------------------------------------
    if args.characters and Path(args.characters).exists():
        char_dir = out_dir / "characters"
        char_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(Path(args.characters).glob("*.safetensors")):
            shutil.copy2(item, char_dir / item.name)
            meta = item.with_suffix(".json")
            if meta.exists():
                shutil.copy2(meta, char_dir / meta.name)
            components.append(
                {
                    "component": f"character:{item.stem}",
                    "format": "textual inversion embedding (safetensors)",
                    "source": str(item),
                    "licence": CHARACTER_LICENCE,
                    "notes": "trained by CheapMage on synthetic data only",
                }
            )

    # ---- smoke test ------------------------------------------------------
    smoke_src = Path(__file__).with_name("quantized_smoke_test.py")
    shutil.copy2(smoke_src, out_dir / "smoke_test.py")

    # ---- manifest --------------------------------------------------------
    entries = _file_entries(out_dir)
    total_bytes = sum(entry["bytes"] for entry in entries)
    total_mb = total_bytes / (1024 * 1024)
    manifest = {
        "package": "cheapmage-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": {
            "id": args.base,
            "resolved_path": str(base_path),
            "licence": args.base_licence or BASE_LICENCE,
            "copyright": BASE_COPYRIGHT,
        },
        "quantization": {
            "unet": "torch.ao.quantization.quantize_dynamic -> qint8, per_channel_symmetric weights",
            "text_encoder": "torch.ao.quantization.quantize_dynamic -> qint8, per_channel_symmetric weights",
            "vae": "float16",
            "note": (
                "Quantised modules are pickled with torch.save and must be loaded with "
                "torch.load(path, weights_only=False)."
            ),
        },
        "components": components,
        "files": entries,
        "total_bytes": total_bytes,
        "total_mb": round(total_mb, 3),
        "total_size": f"{total_mb:.2f} MB",
        "budget_mb": MAX_PACKAGE_MB,
        "within_budget": total_mb <= MAX_PACKAGE_MB,
        "training_data": {
            "primary": "synthetic, procedurally rendered by cheapmage.train.renderers",
            "licence": "CC0-1.0 / synthetic-generated",
            "see": "docs/DATA_LICENSES.md",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("")
    print("=" * 62)
    print(f"  PACKAGE      : {out_dir}")
    print(f"  FILES        : {len(entries)}")
    print(f"  TOTAL SIZE   : {total_mb:.2f} MB  ({total_bytes:,} bytes)")
    print(f"  BUDGET       : {MAX_PACKAGE_MB:.0f} MB")
    print("=" * 62)
    for entry in sorted(entries, key=lambda item: -item["bytes"])[:12]:
        print(f"   {entry['mb']:9.3f} MB  {entry['path']}")
    print("=" * 62)

    if total_mb > MAX_PACKAGE_MB:
        raise SystemExit(
            f"FATAL: packaged model is {total_mb:.2f} MB which exceeds the "
            f"{MAX_PACKAGE_MB:.0f} MB shipping budget. The package was NOT "
            "accepted - reduce the payload (e.g. drop the fp16 VAE copy) and "
            "re-run the export."
        )

    if not args.skip_smoke_test:
        print("[export] running the quantised smoke test ...")
        result = subprocess.run(
            [sys.executable, str(out_dir / "smoke_test.py"), "--package", str(out_dir)],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"FATAL: the quantised smoke test failed with code {result.returncode}"
            )

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheapmage.train.export_package",
        description="Quantise and package the shipped CheapMage model.",
    )
    parser.add_argument("--base", default="models/base_sdtiny")
    parser.add_argument(
        "--adapters",
        nargs="*",
        default=[],
        help="style adapters as '<style>=<path>' (repeatable)",
    )
    parser.add_argument("--characters", default="", help="directory with character embeddings")
    parser.add_argument("--out", default="models/cheapmage-v1")
    parser.add_argument("--base-licence", default=BASE_LICENCE)
    parser.add_argument("--force", action="store_true", help="overwrite an existing package")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_package(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
