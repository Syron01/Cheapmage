"""Generate 10 diverse character test images into 'testing images' directory."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cheapmage.config import GenerationConfig, ConsistencyConfig, get_config
from cheapmage.engine import get_engine
from cheapmage.engine.types import GenerationRequest, SafetyError

OUT_DIR = REPO_ROOT / "testing images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    {
        "filename": "01_anime_sorceress.png",
        "style": "anime",
        "prompt": "anime style, young sorceress with flowing lilac hair, emerald green eyes, holding an ancient crystal staff, hooded velvet cloak, mystical arcane library background, soft lighting",
        "seed": 101,
    },
    {
        "filename": "02_anime_cyberpunk_hacker.png",
        "style": "anime",
        "prompt": "anime style, rogue cyberpunk hacker with messy dark blue hair, glowing visor, black tactical jacket with orange accents, futuristic neon street, night",
        "seed": 102,
    },
    {
        "filename": "03_anime_paladin_knight.png",
        "style": "anime",
        "prompt": "anime style, valiant female paladin with golden braided hair, polished silver plate armor with blue runes, sunlit castle courtyard, clean heroic composition",
        "seed": 103,
    },
    {
        "filename": "04_anime_kitsune_maiden.png",
        "style": "anime",
        "prompt": "anime style, mystical kitsune spirit maiden with elegant white fox ears, traditional red and white robes, moonlit bamboo grove, peaceful night",
        "seed": 204,
    },
    {
        "filename": "05_anime_steampunk_inventor.png",
        "style": "anime",
        "prompt": "anime style, quirky male inventor with brass goggles on forehead, brown leather vest, colorful potion vials, cozy workshop background",
        "seed": 105,
    },
    {
        "filename": "06_3d_scifi_pilot.png",
        "style": "3d",
        "prompt": "3d style, stylized 3d render, energetic female starship pilot with auburn bob cut, sleek white and blue flight suit, cinematic rim lighting, clean render",
        "seed": 106,
    },
    {
        "filename": "07_3d_warrior_monk.png",
        "style": "3d",
        "prompt": "3d style, stylized 3d render, serene warrior monk in crimson and gold robes, calm noble expression, misty bamboo forest background, soft sunlight",
        "seed": 107,
    },
    {
        "filename": "08_3d_elf_ranger.png",
        "style": "3d",
        "prompt": "3d style, stylized 3d render, graceful elf ranger with long silver hair, pointed ears, ornate leather armor, holding a wooden recurve bow, autumn forest",
        "seed": 108,
    },
    {
        "filename": "09_3d_mystic_scholar.png",
        "style": "3d",
        "prompt": "3d style, stylized 3d render, mysterious scholar in obsidian robes, pale skin, purple eyes, glowing ethereal runes around hands, gothic chamber",
        "seed": 109,
    },
    {
        "filename": "10_3d_bounty_hunter.png",
        "style": "3d",
        "prompt": "3d style, stylized 3d render, determined bounty hunter with cybernetic prosthetic arm, rugged trench coat, rain-slicked city alleyway, dramatic lighting",
        "seed": 110,
    },
]

def main() -> None:
    print(f"Loading CheapMage engine to generate {len(PROMPTS)} images into '{OUT_DIR}'...")
    cfg = get_config()
    engine = get_engine(cfg)
    report = engine.load()
    print("Engine loaded:", report.to_dict())

    for idx, item in enumerate(PROMPTS, 1):
        out_path = OUT_DIR / item["filename"]
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"[{idx}/10] {item['filename']} already exists, skipping.")
            continue

        seed = item["seed"]
        success = False
        for attempt in range(3):
            gen = GenerationConfig(
                prompt=item["prompt"],
                style_preset=item["style"],
                width=512,
                height=512,
                steps=20,
                cfg_scale=6.0,
                sampler="dpm++_2m",
                seed=seed + attempt * 77,
                output_format="png",
            )
            req = GenerationRequest(
                generation=gen,
                consistency=ConsistencyConfig(seed_lock=True),
            )
            t0 = time.time()
            try:
                res = engine.generate(req)
                elapsed = time.time() - t0
                out_path.write_bytes(res.image_bytes)
                print(f"[{idx}/10] {item['filename']} ({item['style']}) generated in {elapsed:.2f}s (seed={res.seed}) -> {out_path.name}")
                success = True
                break
            except SafetyError as err:
                print(f"[{idx}/10] Attempt {attempt+1} blocked by moderation: {err}, retrying with alternate seed...")

        if not success:
            print(f"[{idx}/10] WARNING: Could not produce {item['filename']} after retries.")

    print("\nGeneration process complete. Checking directory:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
