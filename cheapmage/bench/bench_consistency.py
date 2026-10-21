"""Character consistency benchmark: same character, 10 different poses.

Usage::

    python -m cheapmage.bench.bench_consistency \\
        --character "a cheerful young woman with short silver hair and blue eyes" \\
        --poses 10 --out reports/consistency.json

What it measures
----------------
For one fixed character description we render N images that differ only in
the pose / camera angle. A consistent model keeps the identity (hair, eye
colour, palette, silhouette) across all of them, so we score every pair of
images with a similarity backend and report mean / min / max.

Similarity backends (the one actually used is recorded in the output)
---------------------------------------------------------------------
``torchvision-resnet18``
    ImageNet-pretrained ResNet-18 pooled features (512-d) + cosine
    similarity. **Only used when the weights are already cached locally** -
    the benchmark never downloads anything.

``descriptor-fallback``
    Dependency-free descriptor built from
    (a) a weighted HSV colour histogram,
    (b) an edge-orientation histogram (magnitude-weighted, 9 bins),
    (c) a 16x16 mean-removed luminance structure map,
    each L2-normalised, weighted 0.35 / 0.35 / 0.30 and concatenated.
    This is a *palette + structure* similarity, not a semantic one, so its
    absolute values are not comparable to the ResNet backend - only the
    relative ordering between runs matters.
"""
from __future__ import annotations

import argparse
import io
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import engine_adapter
from .metrics import (
    environment_info,
    md_table,
    percentile,
    summarize_durations,
    todo,
    write_json,
    write_text,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_READY = 3

# Ten poses / camera angles. Only the pose changes - the character text is
# identical for every image, so any identity drift is the model's fault.
POSES: list[dict[str, str]] = [
    {"name": "front", "text": "standing facing the camera, front view, neutral pose, arms at sides"},
    {"name": "three_quarter", "text": "three-quarter view, standing, slight turn of the torso"},
    {"name": "side", "text": "full side profile view, standing straight"},
    {"name": "back", "text": "seen from behind, back view, standing"},
    {"name": "walking", "text": "walking towards the camera, mid-stride"},
    {"name": "sitting", "text": "sitting on a simple chair, relaxed posture"},
    {"name": "waving", "text": "waving with one hand, friendly gesture"},
    {"name": "arms_up", "text": "arms raised above the head, stretching"},
    {"name": "close_up", "text": "close-up portrait of the face and shoulders, looking at the camera"},
    {"name": "action", "text": "dynamic action pose, running, hair and clothes moving"},
]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _resnet18_weights_path() -> Path | None:
    """Path of the cached ResNet-18 ImageNet weights, if they exist locally."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    try:
        hub_dir = Path(torch.hub.get_dir())
    except Exception:
        return None
    checkpoints = hub_dir / "checkpoints"
    if not checkpoints.is_dir():
        return None
    # torchvision ships the file with a hash suffix: resnet18-f37072fd.pth
    for candidate in sorted(checkpoints.glob("resnet18-*.pth")):
        return candidate
    return None


def _load_resnet18(weights_path: Path):
    """Build a ResNet-18 feature extractor from a **local** weights file."""
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415
    from torchvision import models  # noqa: PLC0415

    model = models.resnet18(weights=None)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()  # keep the 512-d pooled features
    model.eval()
    return model


class DescriptorBackend:
    """Colour + edge + structure descriptor, cosine similarity."""

    name = "descriptor-fallback"
    description = (
        "HSV colour histogram (32x8, value-weighted) + edge orientation "
        "histogram (9 bins, magnitude-weighted) + 16x16 luminance structure"
    )

    def __init__(self) -> None:
        import numpy as np  # noqa: PLC0415

        self.np = np

    # -- feature extraction ------------------------------------------------
    def _features(self, image) -> Any:
        np = self.np
        rgb = image.convert("RGB")

        # (a) colour histogram ------------------------------------------------
        small = rgb.resize((128, 128), _resample())
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
        h = hsv[..., 0] * (255.0 / 360.0)
        s = hsv[..., 1] * (255.0 / 255.0)
        v = hsv[..., 2] * (255.0 / 255.0)
        hist, _, _ = np.histogram2d(
            h.ravel(), s.ravel(), bins=[32, 8], range=[[0, 255], [0, 255]],
            weights=v.ravel(),
        )
        colour = hist.ravel().astype(np.float32)

        # (b) edge orientation histogram -------------------------------------
        gray = np.asarray(rgb.convert("L").resize((256, 256), _resample()),
                          dtype=np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
        bins = np.clip((ang / 20.0).astype(np.int32), 0, 8)
        edge = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=9).astype(np.float32)

        # (c) luminance structure ---------------------------------------------
        struct = np.asarray(rgb.convert("L").resize((16, 16), _resample()),
                            dtype=np.float32)
        struct = struct - struct.mean()
        std = float(np.sqrt((struct ** 2).mean()))
        if std > 1e-6:
            struct = struct / std
        struct = struct.ravel()

        parts = []
        for part, weight in ((colour, 0.35), (edge, 0.35), (struct, 0.30)):
            norm = float(np.sqrt((part ** 2).sum()))
            parts.append(part / norm if norm > 1e-9 else part)
        vec = np.concatenate(parts).astype(np.float32)
        vec_norm = float(np.sqrt((vec ** 2).sum()))
        return vec / vec_norm if vec_norm > 1e-9 else vec

    # -- api ----------------------------------------------------------------
    def embed(self, image) -> Any:
        return self._features(image)

    @staticmethod
    def similarity(a: Any, b: Any) -> float:
        return float((a * b).sum())


class ResNetBackend:
    """ImageNet ResNet-18 pooled features (only from a local weights file)."""

    name = "torchvision-resnet18"
    description = "torchvision resnet18 ImageNet weights, 512-d pooled features"

    def __init__(self, weights_path: Path) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self.model = _load_resnet18(weights_path)
        self.weights_path = str(weights_path)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed(self, image) -> Any:
        torch = self.torch
        img = image.convert("RGB").resize((224, 224), _resample())
        arr = torch.from_numpy(_to_array(img)).permute(2, 0, 1).unsqueeze(0)
        arr = (arr / 255.0 - self.mean) / self.std
        with torch.no_grad():
            feats = self.model(arr)
        vec = feats.squeeze(0)
        return vec / vec.norm().clamp_min(1e-9)

    @staticmethod
    def similarity(a: Any, b: Any) -> float:
        return float((a * b).sum())


def _resample():
    from PIL import Image  # noqa: PLC0415

    return Image.Resampling.LANCZOS


def _to_array(image):
    import numpy as np  # noqa: PLC0415

    return np.asarray(image, dtype="float32")


def pick_backend(prefer: str = "auto") -> tuple[Any, dict[str, Any]]:
    """Select the similarity backend without ever downloading weights."""
    info: dict[str, Any] = {}
    weights = _resnet18_weights_path()
    info["resnet18_weights_cached"] = str(weights) if weights else None
    if prefer in ("auto", "resnet") and weights is not None:
        try:
            backend = ResNetBackend(weights)
            info["backend"] = backend.name
            info["backend_detail"] = backend.description
            info["backend_weights_path"] = backend.weights_path
            return backend, info
        except Exception as exc:
            info["resnet_error"] = f"{exc.__class__.__name__}: {exc}"
    elif prefer == "resnet":
        info["resnet_error"] = (
            "requested resnet backend but the ImageNet weights are not cached "
            "locally; falling back (the bench never downloads weights)"
        )
    backend = DescriptorBackend()
    info["backend"] = backend.name
    info["backend_detail"] = backend.description
    info["backend_weights_path"] = None
    return backend, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cheapmage.bench.bench_consistency",
        description="Measure character consistency across N poses.",
    )
    p.add_argument("--character", type=str,
                   default="a cheerful young woman with short silver hair and blue eyes",
                   help="character description (identical for every pose)")
    p.add_argument("--poses", type=int, default=10,
                   help="number of poses to render (spec: 10)")
    p.add_argument("--out", type=Path, default=Path("reports/consistency.json"))
    p.add_argument("--md-out", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=Path("reports/consistency"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--backend", choices=("auto", "resnet", "descriptor"), default="auto")
    p.add_argument("--style-preset", type=str, default=None,
                   help="override style_preset (default: config value)")
    p.add_argument("--seed", type=int, default=None,
                   help="fixed seed; use the same seed for every pose to isolate "
                        "the pose effect (default: config value)")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="acceptance threshold for the mean pairwise similarity")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config_access import get_bench_config

    cfg = get_bench_config(args.config)
    gen = cfg.generation
    command = (
        f'python -m cheapmage.bench.bench_consistency --character "{args.character}" '
        f"--poses {args.poses} --out {args.out}"
    )
    md_out = args.md_out or Path("reports") / "consistency.md"

    try:
        handle = engine_adapter.load_engine()
    except engine_adapter.EngineUnavailable as exc:
        reason = str(exc)
        print("=" * 72)
        print("bench_consistency: NOT RUN - " + reason)
        print("Run once the engine is available:")
        print("  " + command)
        print("=" * 72)
        write_json(args.out, {
            **todo(reason, command),
            "benchmark": "consistency",
            "character": args.character,
            "poses_requested": args.poses,
            "environment": environment_info(),
        })
        write_text(md_out, _markdown_pending(reason, command))
        return EXIT_NOT_READY

    backend, backend_info = pick_backend(args.backend)
    pose_count = max(1, min(args.poses, len(POSES)))
    poses = POSES[:pose_count]
    if args.poses > len(POSES):
        print(f"note: only {len(POSES)} pose templates exist, using {pose_count} of them")

    args.images_dir.mkdir(parents=True, exist_ok=True)
    images: list[Any] = []
    durations: list[float] = []
    rows: list[dict[str, Any]] = []

    print(f"bench_consistency: {pose_count} poses, backend = {backend.name}")
    for index, pose in enumerate(poses):
        prompt = f"{args.character}, {pose['text']}"
        style = args.style_preset or gen.style_preset
        seed = args.seed if args.seed is not None else gen.seed
        result, elapsed = engine_adapter.generate(
            handle,
            prompt=prompt,
            negative_prompt=gen.negative_prompt,
            style_preset=style,
            width=gen.width,
            height=gen.height,
            steps=gen.steps,
            cfg_scale=gen.cfg_scale,
            sampler=gen.sampler,
            seed=seed,
            output_format="png",
            quality_preset=gen.quality_preset,
            reference_image=cfg.consistency.reference_image,
            reference_strength=cfg.consistency.reference_strength,
            character_embedding=cfg.consistency.character_embedding,
            seed_lock=cfg.consistency.seed_lock,
            timeout_seconds=float(cfg.api.timeout_seconds),
        )
        fields = engine_adapter.result_fields(result)
        durations.append(elapsed)
        path = args.images_dir / f"{index:02d}_{pose['name']}.png"
        _save(result, path)
        image = _load_image(path)
        images.append(image)
        rows.append({
            "index": index,
            "pose": pose["name"],
            "pose_text": pose["text"],
            "prompt": prompt,
            "duration_seconds": round(elapsed, 4),
            "image": str(path),
            "seed": fields.get("seed"),
        })
        print(f"  [{index + 1}/{pose_count}] {pose['name']:14s} {elapsed:.2f}s -> {path}")

    # ---- pairwise similarity ---------------------------------------------
    embeddings = [backend.embed(img) for img in images]
    pairs: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(images)), 2):
        score = backend.similarity(embeddings[i], embeddings[j])
        pairs.append({
            "a": rows[i]["pose"],
            "b": rows[j]["pose"],
            "similarity": round(score, 6),
        })
    scores = [p["similarity"] for p in pairs]

    # centroid similarity: how close is each image to the character centroid
    import numpy as np  # noqa: PLC0415

    stacked = np.stack([np.asarray(e, dtype=np.float32).ravel() for e in embeddings])
    centroid = stacked.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
    centroid_scores = [round(float((e * centroid).sum()), 6) for e in stacked]

    mean_sim = float(np.mean(scores)) if scores else 0.0
    min_sim = float(np.min(scores)) if scores else 0.0
    max_sim = float(np.max(scores)) if scores else 0.0
    p05 = percentile(scores, 5) if scores else 0.0

    payload = {
        "benchmark": "consistency",
        "status": "measured",
        "command": command,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "character": args.character,
        "poses": pose_count,
        "seed": args.seed if args.seed is not None else gen.seed,
        "style_preset": args.style_preset or gen.style_preset,
        "similarity_backend": backend_info,
        "similarity_score_range": "cosine similarity, -1.0 .. 1.0",
        "pairwise": {
            "count": len(pairs),
            "mean": round(mean_sim, 6),
            "min": round(min_sim, 6),
            "max": round(max_sim, 6),
            "p05": round(p05, 6),
            "median": round(percentile(scores, 50), 6) if scores else 0.0,
            "stdev": round(float(np.std(scores)), 6) if scores else 0.0,
            "to_centroid": {
                "mean": round(float(np.mean(centroid_scores)), 6),
                "min": round(float(np.min(centroid_scores)), 6),
                "values": centroid_scores,
            },
        },
        "threshold": args.threshold,
        "pass": bool(mean_sim >= args.threshold),
        "latency_seconds": summarize_durations(durations),
        "images": rows,
        "pairs": pairs,
        "environment": environment_info(),
    }
    write_json(args.out, payload)
    write_text(md_out, _markdown(payload))

    print("-" * 72)
    print(f"backend={backend.name}  mean={mean_sim:.4f} min={min_sim:.4f} "
          f"max={max_sim:.4f} over {len(pairs)} pairs")
    print(f"wrote {args.out} and {md_out}")
    return EXIT_OK if payload["pass"] else EXIT_FAILED


def _save(result: Any, path: Path) -> None:
    data = result["image_bytes"] if isinstance(result, dict) else result.image_bytes
    path.write_bytes(data)


def _load_image(path: Path):
    from PIL import Image  # noqa: PLC0415

    return Image.open(path)


def _markdown(payload: dict[str, Any]) -> str:
    if payload.get("status") != "measured":
        return _markdown_pending(payload.get("reason", ""), payload.get("command", ""))
    pw = payload["pairwise"]
    backend = payload["similarity_backend"]
    out = [
        "# bench_consistency - character consistency over poses",
        "",
        f"_generated {payload['generated_at']}_",
        "",
        md_table(
            ["key", "value"],
            [
                ["character", payload["character"]],
                ["poses", payload["poses"]],
                ["seed", payload["seed"]],
                ["style_preset", payload["style_preset"]],
                ["backend", backend["backend"]],
                ["backend detail", backend.get("backend_detail", "")],
                ["weights path", backend.get("backend_weights_path") or "n/a (fallback)"],
            ],
        ),
        "",
        "## Pairwise similarity (cosine, -1..1)",
        "",
        md_table(
            ["metric", "value", "threshold"],
            [
                ["mean", pw["mean"], payload["threshold"]],
                ["min", pw["min"], "-"],
                ["max", pw["max"], "-"],
                ["p05", pw["p05"], "-"],
                ["median", pw["median"], "-"],
                ["stdev", pw["stdev"], "-"],
                ["to centroid (mean)", pw["to_centroid"]["mean"], "-"],
            ],
        ),
        "",
        f"**Result: {'PASS' if payload['pass'] else 'FAIL'}** "
        f"(mean >= {payload['threshold']})",
        "",
        "## Per-pair scores",
        "",
        md_table(
            ["pose A", "pose B", "similarity"],
            [[p["a"], p["b"], p["similarity"]] for p in payload["pairs"]],
        ),
        "",
    ]
    return "\n".join(out)


def _markdown_pending(reason: str, command: str) -> str:
    return (
        "# bench_consistency - NOT RUN\n\n"
        "**TODO - pending run**\n\n"
        f"Reason: {reason}\n\n"
        "Run this to produce the numbers:\n\n"
        f"```\n{command}\n```\n"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
