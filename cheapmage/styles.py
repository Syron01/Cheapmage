"""Style presets, prompt shaping and the anti-slop vocabulary.

The presets are deliberately short: the student model is tiny, so style is
carried by a small set of high-signal tokens plus a trained style adapter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NEGATIVE_PROMPT_DEFAULT = (
    "lowres, blurry, jpeg artifacts, worst quality, watermark, signature, text, username, "
    "extra fingers, malformed hands, fused fingers, mutated hands, deformed face, "
    "asymmetric eyes, cross-eyed, dead eyes, plastic skin, waxy skin, airbrushed, "
    "oversmoothed skin, doll face, flat lighting, inconsistent lighting, muddy colors, "
    "oversaturated, cluttered background, busy background, duplicate, out of frame, "
    "cropped head, extra limbs, extra head, long neck"
)

# Words that hint at a 3D / CG look.
_3D_HINTS = (
    "3d", "3ds", "3-d", "render", "rendered", "rendering", "cgi", "blender", "octane",
    "unreal", "cinema 4d", "pixar", "disney", "dreamworks", "stylized 3d", "voxel",
    "ray traced", "raytraced", "subsurface scattering", "studio render", "chibi 3d",
    "holographic", "glossy 3d", "stylized render", "game character", "cinematic render",
)

# Words that hint at a 2D anime look.
_ANIME_HINTS = (
    "anime", "manga", "cel", "cel shaded", "cel-shading", "shonen", "shojo", "isekai",
    "waifu", "moe", "studio trigger", "ghibli", "danbooru", "booru", "visual novel",
    "vtuber", "anime style", "lineart", "flat color", "cel animation",
)

_3D_STYLE_TOKENS = (
    "3d render, stylized 3d character, soft subsurface scattering, physically based "
    "materials, soft studio key light, rim light, subtle depth of field, clean neutral "
    "backdrop, full body, sharp focus, high detail"
)

_ANIME_STYLE_TOKENS = (
    "anime style, clean lineart, cel shaded, flat vibrant colors, consistent character "
    "design, soft anime lighting, detailed eyes, full body, sharp focus, high detail"
)

_AUTO_HINTS = {
    "3d": _3D_HINTS,
    "anime": _ANIME_HINTS,
}


@dataclass(frozen=True)
class StylePreset:
    name: str
    tokens: str
    negative_suffix: str = ""
    palette: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return self.tokens
        return f"{prompt}, {self.tokens}"


ANIME = StylePreset(
    name="anime",
    tokens=_ANIME_STYLE_TOKENS,
    negative_suffix="photorealistic, 3d render, cgi, realistic skin texture",
    palette=("saturated but not neon", "cel shaded highlights", "cool shadows with warm bounce"),
)

STYLE_3D = StylePreset(
    name="3d",
    tokens=_3D_STYLE_TOKENS,
    negative_suffix="anime lineart, flat 2d, cel shaded, sketch",
    palette=("neutral studio backdrop", "warm key + cool fill", "soft desaturated background"),
)

PRESETS = {"anime": ANIME, "3d": STYLE_3D}


def detect_style(prompt: str) -> str:
    """Infer ``anime`` or ``3d`` from an English prompt."""
    low = prompt.lower()
    scores = {"anime": 0, "3d": 0}
    for style, hints in _AUTO_HINTS.items():
        for hint in hints:
            if hint in low:
                # longer hints are stronger evidence
                scores[style] += len(hint.split())
    if scores["anime"] == scores["3d"]:
        # Tie-break: presence of a body/camera term, otherwise default to anime
        # (the dominant use case for a roleplay app).
        if any(k in low for k in ("portrait", "character sheet", "waifu", "chibi")):
            return "anime"
        return "anime"
    return max(scores, key=scores.get)


def resolve_style(style_preset: str, prompt: str) -> str:
    if style_preset in ("anime", "3d"):
        return style_preset
    return detect_style(prompt)


def build_prompt(prompt: str, style_preset: str, character_embedding: str | None = None) -> str:
    """Compose the final conditioning prompt."""
    style = resolve_style(style_preset, prompt)
    base = PRESETS[style].apply(prompt)
    if character_embedding:
        base = f"{character_embedding}, {base}"
    return base


def build_negative(negative_prompt: str, style_preset: str, extra: str | None = None) -> str:
    style = resolve_style(style_preset, negative_prompt or "")
    neg = (negative_prompt or "").strip() or NEGATIVE_PROMPT_DEFAULT
    suffix = PRESETS[style].negative_suffix
    if suffix:
        neg = f"{neg}, {suffix}"
    if extra:
        neg = f"{neg}, {extra}"
    return neg


_WORD_RE = re.compile(r"[A-Za-z]+")


def looks_english(text: str) -> bool:
    """True when the prompt contains no non-ASCII letters (English-only policy)."""
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x024F:
            return False
    return True


def prompt_words(prompt: str) -> list[str]:
    return _WORD_RE.findall(prompt)
