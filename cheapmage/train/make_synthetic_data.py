"""CLI: generate the synthetic, licence-clean training corpus.

Example
-------
    python -m cheapmage.train.make_synthetic_data --out data/synth --count 2000 \
        --size 512 --styles anime,3d --seed 0

Everything is rendered procedurally (see :mod:`cheapmage.train.renderers`);
no network access and no third-party artwork is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import build_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheapmage.train.make_synthetic_data",
        description="Render a synthetic anime/3D portrait dataset with captions.",
    )
    parser.add_argument("--out", default="data/synth", help="output directory")
    parser.add_argument(
        "--count", type=int, default=2000, help="number of images PER STYLE"
    )
    parser.add_argument("--size", type=int, default=512, help="square resolution")
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["anime", "3d"],
        help="styles to render (space or comma separated): anime 3d",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic)")
    parser.add_argument(
        "--format", default="jpg", choices=("jpg", "png", "webp"), help="image format"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    styles: list[str] = []
    for chunk in args.styles:
        styles.extend(s.strip() for s in str(chunk).split(",") if s.strip())
    out = Path(args.out)
    print(f"[synth] rendering {args.count} image(s) per style {styles} at {args.size}px -> {out}")
    manifest = build_synthetic_dataset(
        out,
        n_per_style=args.count,
        size=args.size,
        seed=args.seed,
        styles=styles,
        image_format=args.format,
    )
    summary = {
        "images": len(manifest),
        "styles": manifest.style_summary(),
        "licences": manifest.licence_summary(),
        "sources": manifest.source_summary(),
        "bytes": manifest.total_bytes(),
        "manifest_sha256": manifest.sha256(),
        "manifest_path": str(out / "dataset_manifest.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
