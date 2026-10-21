"""Render the fixed 30-image human-evaluation set and prepare the score sheet.

Usage::

    python -m cheapmage.bench.bench_human_eval --count 30 --out reports/human_eval.json

The script:

1. reads ``cheapmage/bench/human_eval_prompts.json`` (the fixed set),
2. renders every prompt with the DEFAULT generation settings and a fixed seed
   per image, so the set is reproducible,
3. writes ``reports/human_eval.json`` (the manifest that lists every image
   and the file it landed in) and ``reports/human_eval_sheet.csv`` (the empty
   score sheet for the three raters),
4. prints the aggregation helper so the raters can fill in the sheet.

It does **not** score anything: human scoring is the protocol in
``cheapmage/bench/human_eval.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import engine_adapter
from .metrics import environment_info, md_table, todo, write_json

EXIT_OK = 0
EXIT_NOT_READY = 3

PROMPTS_PATH = Path(__file__).resolve().parent / "human_eval_prompts.json"

RATERS = ("R1", "R2", "R3")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cheapmage.bench.bench_human_eval",
        description="Render the fixed 30-image human evaluation set.",
    )
    p.add_argument("--count", type=int, default=30, help="number of images (spec: 30)")
    p.add_argument("--out", type=Path, default=Path("reports/human_eval.json"))
    p.add_argument("--sheet-out", type=Path, default=Path("reports/human_eval_sheet.csv"))
    p.add_argument("--images-dir", type=Path, default=Path("reports/human_eval"))
    p.add_argument("--config", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config_access import get_bench_config

    cfg = get_bench_config(args.config)
    items = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))["items"]
    items = items[: max(1, args.count)]

    command = (
        f"python -m cheapmage.bench.bench_human_eval --count {args.count} "
        f"--out {args.out}"
    )

    try:
        handle = engine_adapter.load_engine()
    except engine_adapter.EngineUnavailable as exc:
        reason = str(exc)
        print("bench_human_eval: NOT RUN - " + reason)
        print("Run once the engine is available:\n  " + command)
        write_json(args.out, {
            **todo(reason, command),
            "benchmark": "human_eval",
            "environment": environment_info(),
        })
        return EXIT_NOT_READY

    args.images_dir.mkdir(parents=True, exist_ok=True)
    gen = cfg.generation
    manifest: list[dict[str, Any]] = []
    for item in items:
        result, elapsed = engine_adapter.generate(
            handle,
            prompt=item["prompt"],
            negative_prompt=gen.negative_prompt,
            style_preset=item["style"],
            width=gen.width,
            height=gen.height,
            steps=gen.steps,
            cfg_scale=gen.cfg_scale,
            sampler=gen.sampler,
            seed=item["seed"],
            output_format="png",
            quality_preset=gen.quality_preset,
            timeout_seconds=float(cfg.api.timeout_seconds),
        )
        path = args.images_dir / f"{item['id']}.png"
        data = result["image_bytes"] if isinstance(result, dict) else result.image_bytes
        path.write_bytes(data)
        manifest.append({
            "id": item["id"],
            "style": item["style"],
            "seed": item["seed"],
            "prompt": item["prompt"],
            "image": str(path),
            "duration_seconds": round(elapsed, 3),
        })
        print(f"  {item['id']} ({item['style']}) -> {path}")

    payload = {
        "benchmark": "human_eval",
        "status": "images_rendered_awaiting_human_scoring",
        "command": command,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "cheapmage/bench/human_eval.md",
        "count": len(manifest),
        "images": manifest,
        "environment": environment_info(),
    }
    write_json(args.out, payload)

    args.sheet_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sheet_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "image_id", "rater", "style_block", "quality_score",
            "slop_mean", "slop_flags", "note",
        ])
        for item in manifest:
            for rater in RATERS:
                writer.writerow([item["id"], rater, "", "", "", "", ""])

    print()
    print(md_table(
        ["image_id", "style", "seed", "file"],
        [[m["id"], m["style"], m["seed"], m["image"]] for m in manifest],
    ))
    print()
    print(f"wrote {args.out} and {args.sheet_out}")
    print("Now have 3 raters score every image with the protocol in "
          "cheapmage/bench/human_eval.md.")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
