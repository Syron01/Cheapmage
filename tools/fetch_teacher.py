"""Download the TRAINING-ONLY teacher model (not shipped).

Teacher: Lykon/dreamshaper-8 (SD1.5, CreativeML OpenRAIL-M).
It is used exclusively to synthesise a licence-clean style dataset; the shipped
artifact is the tiny SD-Tiny student + our trained adapters.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download

PATTERNS = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "tokenizer/*",
    "scheduler/scheduler_config.json",
]

if __name__ == "__main__":
    out = REPO_ROOT / "models" / "teacher_dreamshaper8"
    path = snapshot_download(
        "Lykon/dreamshaper-8",
        local_dir=str(out),
        allow_patterns=PATTERNS,
    )
    print("teacher at", path)
