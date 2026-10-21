"""Generate the licence-clean style dataset with the teacher model.

The teacher (SD1.5 / dreamshaper-8, OpenRAIL-M) is used ONLY here, to create
images that are then used to train the *student* (SD-Tiny) style adapters.
Nothing from the teacher is shipped.

Output layout:
    data/teacher/<style>/<id>.png
    data/teacher/<style>/<id>.txt        (caption for the student)
    data/teacher/manifest.json            (licence + provenance of every file)

Usage:
    python tools/gen_teacher_data.py --style anime --count 400 --out data/teacher
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

ANIME_STYLE = (
    "masterpiece, best quality, highly detailed, anime style, clean lineart, "
    "cel shaded, flat vibrant colors, sharp focus, character portrait"
)

STYLE_3D = (
    "masterpiece, best quality, stylized 3d character render, unreal engine, "
    "soft subsurface scattering, physically based materials, studio softbox lighting, "
    "rim light, clean neutral backdrop, sharp focus, character portrait"
)

NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra fingers, mutated hands, extra limbs, "
    "deformed face, blurry, watermark, text, jpeg artifacts, worst quality, "
    "plastic skin, doll face, oversmoothed, flat lighting, multiple characters"
)

HAIR = ["silver", "black", "blonde", "twintails", "short brown", "long blue", "pink", "white", "orange", "dark green"]
EYES = ["blue", "green", "amber", "red", "violet", "grey", "teal", "brown", "gold"]
OUTFIT = [
    "wearing a yellow raincoat", "wearing a school uniform with a red ribbon",
    "wearing a black hoodie", "wearing a white blouse and blue skirt",
    "wearing a green bomber jacket", "wearing a red hoodie and shorts",
    "wearing a navy trench coat", "wearing a pink cardigan",
]
POSE = [
    "standing, front view", "standing, three quarter view", "sitting on a chair",
    "walking, side view", "close up portrait, looking at the viewer",
    "waist up, arms crossed", "full body, dynamic pose, wind in the hair",
    "leaning against a wall", "holding a cup of tea", "looking over the shoulder",
]
BG = [
    "clean studio background", "city street at sunset", "cherry blossom park",
    "plain gradient background", "cozy cafe interior", "school rooftop at noon",
    "night sky with stars", "sunlit meadow",
]


def build_prompt(style: str, rng: random.Random) -> str:
    parts = [
        f"1girl, {rng.choice(HAIR)} hair, {rng.choice(EYES)} eyes, {rng.choice(OUTFIT)}",
        rng.choice(POSE),
        rng.choice(BG),
        ANIME_STYLE if style == "anime" else STYLE_3D,
    ]
    return ", ".join(parts)


def caption_for(style: str, rng: random.Random) -> str:
    """Caption used for the student (content only, style via the trigger)."""
    parts = [
        f"a character with {rng.choice(HAIR)} hair and {rng.choice(EYES)} eyes",
        rng.choice(OUTFIT),
        rng.choice(POSE),
    ]
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["anime", "3d"], required=True)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "teacher"))
    ap.add_argument("--teacher", default=str(REPO_ROOT / "models" / "teacher_dreamshaper8"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    args = ap.parse_args()

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        args.teacher, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.out) / args.style
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + (0 if args.style == "anime" else 10_000))
    manifest_path = Path(args.out) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    for i in range(args.count):
        prompt = build_prompt(args.style, rng)
        caption = caption_for(args.style, rng)
        seed = rng.randint(0, 2**31 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        ).images[0]
        path = out_dir / f"{args.style}_{i:05d}.png"
        image.save(path)
        (out_dir / f"{args.style}_{i:05d}.txt").write_text(caption, encoding="utf-8")
        manifest[str(path.relative_to(Path(args.out)))] = {
            "style": args.style,
            "source": "synthetic (teacher-generated)",
            "teacher": "Lykon/dreamshaper-8 (CreativeML OpenRAIL-M)",
            "licence": "CreativeML OpenRAIL-M / generated output",
            "prompt": prompt,
            "caption": caption,
            "seed": seed,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contains_minors": False,
            "contains_nsfw": False,
        }
        if (i + 1) % 25 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            print(f"{args.style}: {i + 1}/{args.count}")

    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done: {args.count} images -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
