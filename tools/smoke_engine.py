"""Smoke test: load the packaged (INT8) model and generate one image per style."""
from __future__ import annotations

import sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cheapmage.config import get_config
from cheapmage.engine import get_engine
from cheapmage.engine.types import GenerationRequest
from cheapmage.config import GenerationConfig, ConsistencyConfig

cfg = get_config()
engine = get_engine(cfg)
t0 = time.time()
report = engine.load()
print("load report:", report.to_dict())

for style in ("anime", "3d"):
    gen = GenerationConfig(
        prompt="a cheerful young woman with short silver hair and blue eyes, wearing a yellow raincoat, full body",
        style_preset=style,
        steps=20,
        cfg_scale=6.0,
        sampler="lcm",
        seed=1234,
        output_format="png",
    )
    t = time.time()
    res = engine.generate(GenerationRequest(generation=gen, consistency=ConsistencyConfig()))
    out = REPO_ROOT / "tools" / f"smoke_{style}.png"
    out.write_bytes(res.image_bytes)
    print(f"{style}: {res.duration_seconds:.2f}s seed={res.seed} sampler={res.sampler} -> {out}")
    print("   memory:", res.memory_mb, "cost:", res.cost["cost_tl"], res.cost["detail"])
