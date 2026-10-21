"""Procedural portrait renderer used to build the *synthetic* training set.

Everything here is generated from code with a seeded RNG: no scraped artwork,
no copyrighted characters, no real people, no photographic source data.
The output is deliberately simple - it is a *style* signal (anime-ish flat
cel shading vs. 3D-ish soft volumetric shading), not a photoreal dataset.

The renderer is deterministic: the same ``seed`` always produces the same
images, so datasets are reproducible from the manifest alone.
"""
from __future__ import annotations

import math
import random
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

SKIN_TONES: tuple[tuple[str, str], ...] = (
    ("porcelain", "#f7dcc6"),
    ("light warm", "#eec9a8"),
    ("warm beige", "#d9a97c"),
    ("tan", "#bd8253"),
    ("olive", "#9c6636"),
    ("deep brown", "#714223"),
    ("dark brown", "#4f2f1a"),
)

HAIR_COLORS: tuple[tuple[str, str], ...] = (
    ("jet black", "#22212c"),
    ("charcoal", "#3d3d4a"),
    ("dark brown", "#4f3018"),
    ("chestnut", "#7d4523"),
    ("auburn", "#a9552b"),
    ("blonde", "#e3c66d"),
    ("platinum", "#e6e4db"),
    ("silver", "#c7cad2"),
    ("mint", "#8fd6c2"),
    ("sky blue", "#82b3e4"),
    ("lavender", "#b4a3e7"),
    ("rose", "#e49aa8"),
)

EYE_COLORS: tuple[tuple[str, str], ...] = (
    ("blue", "#4f8fd0"),
    ("green", "#5aa870"),
    ("amber", "#d59a35"),
    ("brown", "#7b4a26"),
    ("grey", "#8a8f9a"),
    ("violet", "#8a63c4"),
    ("teal", "#3fa8a0"),
)

CLOTHING: tuple[tuple[str, str], ...] = (
    ("navy hoodie", "#26375f"),
    ("charcoal jacket", "#383943"),
    ("cream sweater", "#e6dbc2"),
    ("rust coat", "#a64a2f"),
    ("forest green sweater", "#345f45"),
    ("white shirt", "#f0f0ee"),
    ("burgundy top", "#6f2637"),
    ("slate blue top", "#4f6180"),
    ("olive overalls", "#6e6e3d"),
    ("plum dress", "#603556"),
)

BACKGROUNDS: tuple[tuple[str, str], ...] = (
    ("soft grey studio backdrop", "#c9ccd2"),
    ("warm sunset gradient", "#eea566"),
    ("cool blue gradient", "#7fa6d8"),
    ("mint green gradient", "#a6d6c0"),
    ("lavender gradient", "#c1b1e6"),
    ("dusk purple gradient", "#685884"),
    ("cream paper backdrop", "#eee6d6"),
    ("teal gradient", "#5fb3ac"),
)

ANIME_TOKEN = "anime style, clean lineart, cel shaded, flat vibrant colors"
STYLE_3D_TOKEN = (
    "3d render, stylized 3d character, soft subsurface scattering, "
    "physically based materials, soft studio key light"
)
STYLE_TOKENS = {"anime": ANIME_TOKEN, "3d": STYLE_3D_TOKEN}

_HAIR_STYLES = ("short", "medium", "long", "tied", "spiky", "bob")
_EXPRESSIONS = ("neutral", "smiling", "determined", "calm", "cheerful")


# ---------------------------------------------------------------------------
# Low level helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _mix(
    a: str | tuple[int, int, int], b: str | tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    ra, ga, ba = _hex_to_rgb(a) if isinstance(a, str) else a
    rb, gb, bb = _hex_to_rgb(b) if isinstance(b, str) else b
    return (
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )


def _shade(color: str, amount: float) -> tuple[int, int, int]:
    """Darken (``amount`` < 0) or lighten (``amount`` > 0) a hex colour."""
    r, g, b = _hex_to_rgb(color)
    if amount >= 0:
        return (
            int(r + (255 - r) * amount),
            int(g + (255 - g) * amount),
            int(b + (255 - b) * amount),
        )
    return (int(r * (1 + amount)), int(g * (1 + amount)), int(b * (1 + amount)))


def _vertical_gradient(
    size: int, top: str | tuple[int, int, int], bottom: str | tuple[int, int, int]
) -> Image.Image:
    grad = Image.new("RGB", (1, size))
    px = grad.load()
    for y in range(size):
        t = y / max(1, size - 1)
        px[0, y] = _mix(top, bottom, t)
    return grad.resize((size, size), Image.BILINEAR)


def _radial_gradient(
    size: int,
    center: str | tuple[int, int, int],
    edge: str | tuple[int, int, int],
    cx: float = 0.5,
    cy: float = 0.42,
) -> Image.Image:
    """Cheap radial gradient: small radial image, upscaled smoothly."""
    small = 72
    grad = Image.new("RGB", (small, small))
    px = grad.load()
    ccx, ccy = cx * small, cy * small
    maxd = math.hypot(small, small)
    for y in range(small):
        for x in range(small):
            d = min(1.0, math.hypot(x - ccx, y - ccy) / (maxd * 0.72))
            px[x, y] = _mix(center, edge, d)
    return grad.resize((size, size), Image.BICUBIC)


def _ellipse_mask(size: tuple[int, int], box: tuple[float, float, float, float]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([int(v) for v in box], fill=255)
    return mask


def _paste_ellipse(
    base: Image.Image,
    grad: Image.Image,
    box: tuple[float, float, float, float],
) -> None:
    """Paste ``grad`` into ``base`` clipped to an ellipse ``box`` (in-place)."""
    width = max(1, int(box[2] - box[0]))
    height = max(1, int(box[3] - box[1]))
    tile = grad.resize((width, height), Image.BICUBIC)
    mask = _ellipse_mask((width, height), (0, 0, width - 1, height - 1))
    base.paste(tile, (int(box[0]), int(box[1])), mask)


def _rounded_mask(
    size: tuple[int, int],
    box: tuple[float, float, float, float],
    radius: float,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [int(v) for v in box], radius=int(radius), fill=255
    )
    return mask


def _paste_rounded(
    base: Image.Image,
    grad: Image.Image,
    box: tuple[float, float, float, float],
    radius: float,
) -> None:
    width = max(1, int(box[2] - box[0]))
    height = max(1, int(box[3] - box[1]))
    tile = grad.resize((width, height), Image.BICUBIC)
    mask = _rounded_mask((width, height), (0, 0, width - 1, height - 1), radius)
    base.paste(tile, (int(box[0]), int(box[1])), mask)


def _rotate_layer(
    layer: Image.Image, angle: float, center: tuple[float, float]
) -> Image.Image:
    if abs(angle) < 0.05:
        return layer
    return layer.rotate(angle, resample=Image.BICUBIC, center=center)


# ---------------------------------------------------------------------------
# Shared scene description
# ---------------------------------------------------------------------------


def _describe(rng: random.Random, style: str) -> dict[str, Any]:
    skin_name, skin = rng.choice(SKIN_TONES)
    hair_name, hair = rng.choice(HAIR_COLORS)
    eye_name, eye = rng.choice(EYE_COLORS)
    cloth_name, cloth = rng.choice(CLOTHING)
    bg_name, bg = rng.choice(BACKGROUNDS)
    return {
        "style": style,
        "skin_name": skin_name,
        "skin": skin,
        "hair_name": hair_name,
        "hair": hair,
        "hair_style": rng.choice(_HAIR_STYLES),
        "eye_name": eye_name,
        "eye": eye,
        "cloth_name": cloth_name,
        "cloth": cloth,
        "bg_name": bg_name,
        "bg": bg,
        "expression": rng.choice(_EXPRESSIONS),
        "tilt": rng.uniform(-6.0, 6.0),
        "framing": rng.choice(("bust", "half_body")),
        "glasses": rng.random() < 0.15,
    }


def _caption(desc: dict[str, Any]) -> str:
    token = STYLE_TOKENS[desc["style"]]
    parts = [
        token,
        "portrait of an adult character",
        f"{desc['hair_style']} {desc['hair_name']} hair",
        f"{desc['eye_name']} eyes",
        f"{desc['skin_name']} skin tone",
        f"wearing a {desc['cloth_name']}",
        f"{desc['bg_name']}",
        f"{desc['expression']} expression",
        "clean composition, single subject, centered",
    ]
    if desc.get("glasses"):
        parts.append("wearing thin framed glasses")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Anime style renderer
# ---------------------------------------------------------------------------


def render_anime(seed: int, size: int = 512) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(seed)
    desc = _describe(rng, "anime")
    ss = 2  # supersample factor for clean line art
    S = size * ss
    line = (28, 26, 38)
    lw = max(2, int(S * 0.0055))

    bg_top = _shade(desc["bg"], 0.10)
    bg_bottom = _shade(desc["bg"], -0.24)
    img = _vertical_gradient(S, bg_top, bg_bottom).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Background props: an abstract sun disc and a horizon band.
    if rng.random() < 0.6:
        r = S * rng.uniform(0.09, 0.16)
        cx_sun = S * rng.uniform(0.18, 0.82)
        cy_sun = S * rng.uniform(0.10, 0.26)
        _ = draw.ellipse(
            [int(cx_sun - r), int(cy_sun - r), int(cx_sun + r), int(cy_sun + r)],
            fill=_shade(desc["bg"], 0.42),
        )
    draw.rectangle([0, int(S * 0.76), S, S], fill=_shade(desc["bg"], -0.36))

    cx = S * 0.5 + rng.uniform(-0.03, 0.03) * S
    head_cy = S * (0.40 if desc["framing"] == "bust" else 0.34)
    rx = S * 0.132
    ry = S * 0.168
    shoulder_top = head_cy + ry * 1.05

    # ---- body ------------------------------------------------------------
    body = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.ellipse(
        [int(cx - S * 0.26), int(shoulder_top), int(cx + S * 0.26), int(S * 1.2)],
        fill=_hex_to_rgb(desc["cloth"]),
        outline=line,
        width=lw,
    )
    # collar opening
    bd.ellipse(
        [int(cx - S * 0.070), int(shoulder_top - S * 0.030), int(cx + S * 0.070), int(shoulder_top + S * 0.070)],
        fill=_shade(desc["cloth"], -0.22),
        outline=line,
        width=max(1, lw // 2),
    )
    # neck
    bd.polygon(
        [
            (cx - S * 0.040, head_cy + ry * 0.55),
            (cx + S * 0.040, head_cy + ry * 0.55),
            (cx + S * 0.050, shoulder_top + S * 0.010),
            (cx - S * 0.050, shoulder_top + S * 0.010),
        ],
        fill=_shade(desc["skin"], -0.10),
        outline=line,
        width=max(1, lw // 2),
    )
    img.alpha_composite(body)

    # ---- head group ------------------------------------------------------
    head = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(head)
    hair_dark = _shade(desc["hair"], -0.30)

    # back hair (visible as a frame around the head)
    long_hair = desc["hair_style"] in ("long", "tied")
    hair_w = rx * (1.34 if long_hair else 1.20)
    hair_bottom = head_cy + ry * (1.85 if long_hair else 1.10)
    hd.ellipse(
        [
            int(cx - hair_w),
            int(head_cy - ry * 1.30),
            int(cx + hair_w),
            int(hair_bottom),
        ],
        fill=_hex_to_rgb(desc["hair"]),
        outline=line,
        width=lw,
    )

    # face
    hd.ellipse(
        [int(cx - rx * 0.92), int(head_cy - ry * 0.90), int(cx + rx * 0.92), int(head_cy + ry * 1.04)],
        fill=_hex_to_rgb(desc["skin"]),
        outline=line,
        width=lw,
    )

    # bangs: a filled cap whose lower edge is a jagged fringe
    fringe: list[tuple[float, float]] = []
    steps = 7
    for i in range(steps + 1):
        t = i / steps
        px = cx - rx * 1.06 + rx * 2.12 * t
        depth = 0.62 + 0.40 * math.sin(t * math.pi * 1.15 + 0.35)
        py = head_cy - ry * 0.80 + ry * depth
        fringe.append((px, py))
    cap_poly = [
        (cx - rx * 1.16, head_cy - ry * 1.24),
        (cx + rx * 1.16, head_cy - ry * 1.24),
    ]
    cap_poly += list(reversed(fringe))
    hd.polygon(cap_poly, fill=_hex_to_rgb(desc["hair"]))
    hd.line(fringe, fill=line, width=lw, joint="curve")

    # side locks framing the cheeks
    for sign in (-1, 1):
        hd.polygon(
            [
                (cx + sign * rx * 0.98, head_cy - ry * 0.72),
                (cx + sign * rx * 1.12, head_cy + ry * 0.62),
                (cx + sign * rx * 0.80, head_cy + ry * 0.60),
                (cx + sign * rx * 0.82, head_cy - ry * 0.58),
            ],
            fill=_hex_to_rgb(desc["hair"]),
            outline=line,
        )
    if desc["hair_style"] == "spiky":
        for i in range(5):
            t = i / 4
            bx = cx - rx * 0.85 + rx * 1.7 * t
            hd.polygon(
                [
                    (bx - rx * 0.16, head_cy - ry * 0.95),
                    (bx + rng.uniform(-0.1, 0.1) * rx, head_cy - ry * 1.38),
                    (bx + rx * 0.16, head_cy - ry * 0.95),
                ],
                fill=_hex_to_rgb(desc["hair"]),
                outline=line,
            )

    # cel shadow: clipped to the face silhouette
    face_mask = _ellipse_mask(
        (S, S), (cx - rx * 0.92, head_cy - ry * 0.90, cx + rx * 0.92, head_cy + ry * 1.04)
    )
    shade_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade_layer)
    sd.ellipse(
        [int(cx - rx * 0.30), int(head_cy - ry), int(cx + rx * 1.02), int(head_cy + ry)],
        fill=(*_shade(desc["skin"], -0.12), 120),
    )
    clipped = Image.composite(shade_layer, Image.new("RGBA", (S, S)), shade_layer.split()[3])
    clipped.putalpha(ImageChops.multiply(clipped.split()[3], face_mask))
    head.alpha_composite(clipped)

    # eyes
    eye_y = head_cy + ry * 0.06
    eye_dx = rx * 0.44
    eye_rx = rx * 0.29
    eye_ry = ry * 0.20
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        hd.ellipse(
            [int(ex - eye_rx), int(eye_y - eye_ry), int(ex + eye_rx), int(eye_y + eye_ry)],
            fill=(252, 250, 250),
            outline=line,
            width=max(1, lw // 2),
        )
        iris_r = eye_ry * 1.02
        hd.ellipse(
            [int(ex - iris_r), int(eye_y - iris_r), int(ex + iris_r), int(eye_y + iris_r)],
            fill=_hex_to_rgb(desc["eye"]),
            outline=line,
            width=max(1, int(S * 0.002)),
        )
        hd.ellipse(
            [int(ex - iris_r * 0.42), int(eye_y - iris_r * 0.42), int(ex + iris_r * 0.42), int(eye_y + iris_r * 0.42)],
            fill=(22, 20, 28),
        )
        hd.ellipse(
            [int(ex - iris_r * 0.66), int(eye_y - iris_r * 0.78), int(ex - iris_r * 0.24), int(eye_y - iris_r * 0.36)],
            fill=(255, 255, 255),
        )
        # upper lash line
        hd.arc(
            [int(ex - eye_rx * 1.05), int(eye_y - eye_ry * 1.25), int(ex + eye_rx * 1.05), int(eye_y + eye_ry * 0.85)],
            start=185,
            end=355,
            fill=line,
            width=max(2, int(S * 0.006)),
        )
        # brow
        hd.arc(
            [int(ex - eye_rx * 1.1), int(eye_y - eye_ry * 2.4), int(ex + eye_rx * 1.1), int(eye_y - eye_ry * 0.9)],
            start=195,
            end=345,
            fill=hair_dark,
            width=max(2, int(S * 0.005)),
        )

    # nose + mouth
    hd.line(
        [(cx - rx * 0.07, head_cy + ry * 0.34), (cx + rx * 0.03, head_cy + ry * 0.38)],
        fill=_shade(desc["skin"], -0.30),
        width=max(1, int(S * 0.003)),
    )
    mouth_y = head_cy + ry * 0.52
    if desc["expression"] in ("smiling", "cheerful"):
        hd.arc(
            [int(cx - rx * 0.22), int(mouth_y - ry * 0.14), int(cx + rx * 0.22), int(mouth_y + ry * 0.16)],
            start=15,
            end=165,
            fill=line,
            width=max(2, int(S * 0.005)),
        )
    else:
        hd.line(
            [(cx - rx * 0.15, mouth_y), (cx + rx * 0.15, mouth_y)],
            fill=line,
            width=max(2, int(S * 0.004)),
        )
    if desc.get("glasses"):
        for sign in (-1, 1):
            ex = cx + sign * eye_dx
            hd.ellipse(
                [int(ex - eye_rx * 1.4), int(eye_y - eye_ry * 1.5), int(ex + eye_rx * 1.4), int(eye_y + eye_ry * 1.5)],
                outline=line,
                width=max(1, int(S * 0.004)),
            )
        hd.line(
            [(cx - eye_dx + eye_rx * 1.4, eye_y), (cx + eye_dx - eye_rx * 1.4, eye_y)],
            fill=line,
            width=max(1, int(S * 0.004)),
        )

    head = _rotate_layer(head, desc["tilt"], (cx, head_cy))
    img.alpha_composite(head)

    out = img.convert("RGB").resize((size, size), Image.LANCZOS)
    desc["caption"] = _caption(desc)
    return out, desc


# ---------------------------------------------------------------------------
# 3D style renderer
# ---------------------------------------------------------------------------


def render_3d(seed: int, size: int = 512) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(seed ^ 0x5EED)
    desc = _describe(rng, "3d")
    ss = 2
    S = size * ss

    img = _radial_gradient(S, _shade(desc["bg"], 0.24), _shade(desc["bg"], -0.44)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [int(-S * 0.2), int(S * 0.80), int(S * 1.2), int(S * 1.4)],
        fill=(*_shade(desc["bg"], -0.30), 210),
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=S * 0.02))

    cx = S * 0.5 + rng.uniform(-0.025, 0.025) * S
    head_cy = S * (0.40 if desc["framing"] == "bust" else 0.34)
    rx = S * 0.138
    ry = S * 0.172
    shoulder_top = head_cy + ry * 1.10

    # ---- body ------------------------------------------------------------
    body = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    body_grad = _radial_gradient(
        S, _shade(desc["cloth"], 0.18), _shade(desc["cloth"], -0.34), cx=0.38, cy=0.30
    )
    _paste_rounded(
        body,
        body_grad,
        (cx - S * 0.25, shoulder_top, cx + S * 0.25, S * 1.15),
        radius=S * 0.11,
    )
    body = body.filter(ImageFilter.GaussianBlur(radius=S * 0.004))
    bd = ImageDraw.Draw(body)
    bd.rectangle(
        [int(cx - S * 0.038), int(head_cy + ry * 0.55), int(cx + S * 0.038), int(shoulder_top + S * 0.02)],
        fill=_shade(desc["skin"], -0.16),
    )
    body = body.filter(ImageFilter.GaussianBlur(radius=S * 0.003))
    img.alpha_composite(body)

    # ---- head group ------------------------------------------------------
    head = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # hair volume behind the head
    hair_grad = _radial_gradient(
        S, _shade(desc["hair"], 0.16), _shade(desc["hair"], -0.30), cx=0.36, cy=0.26
    )
    _paste_ellipse(
        head,
        hair_grad,
        (cx - rx * 1.18, head_cy - ry * 1.20, cx + rx * 1.18, head_cy + ry * 1.26),
    )

    # ears
    hd = ImageDraw.Draw(head)
    for sign in (-1, 1):
        ex = cx + sign * rx * 0.97
        hd.ellipse(
            [int(ex - rx * 0.15), int(head_cy - ry * 0.04), int(ex + rx * 0.15), int(head_cy + ry * 0.32)],
            fill=_shade(desc["skin"], -0.10),
        )

    # face with volumetric shading
    face_grad = _radial_gradient(
        S, _shade(desc["skin"], 0.20), _shade(desc["skin"], -0.20), cx=0.40, cy=0.34
    )
    _paste_ellipse(head, face_grad, (cx - rx, head_cy - ry, cx + rx, head_cy + ry))

    # hair cap over the forehead (soft, high hairline)
    cap_mask = _ellipse_mask(
        (S, S), (cx - rx * 1.16, head_cy - ry * 1.18, cx + rx * 1.16, head_cy + ry * 0.95)
    )
    band = Image.new("L", (S, S), 0)
    ImageDraw.Draw(band).rectangle([0, 0, S, int(head_cy - ry * 0.28)], fill=255)
    cap_mask = ImageChops.multiply(cap_mask, band)
    cap_tile = _radial_gradient(
        S, _shade(desc["hair"], 0.20), _shade(desc["hair"], -0.26), cx=0.36, cy=0.22
    ).convert("RGBA")
    head.paste(cap_tile, (0, 0), cap_mask)

    # fringe: jagged hair edge over the forehead, painted with the hair gradient
    fringe_mask = Image.new("L", (S, S), 0)
    fmd = ImageDraw.Draw(fringe_mask)
    edge: list[tuple[float, float]] = []
    steps = 8
    for i in range(steps + 1):
        t = i / steps
        fx = cx - rx * 1.16 + rx * 2.32 * t
        depth = 0.16 + 0.26 * math.sin(t * math.pi * 1.2 + 0.5)
        edge.append((fx, head_cy - ry * (0.34 - depth)))
    fmd.polygon(
        [(cx - rx * 1.30, head_cy - ry * 1.40), (cx + rx * 1.30, head_cy - ry * 1.40)]
        + list(reversed(edge)),
        fill=255,
    )
    fringe_mask = ImageChops.multiply(fringe_mask, _ellipse_mask(
        (S, S), (cx - rx * 1.16, head_cy - ry * 1.18, cx + rx * 1.16, head_cy + ry * 0.95)
    ))
    fringe_tile = _radial_gradient(
        S, _shade(desc["hair"], 0.12), _shade(desc["hair"], -0.22), cx=0.36, cy=0.24
    ).convert("RGBA")
    head.paste(fringe_tile, (0, 0), fringe_mask)
    # specular highlight on the hair
    spec = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(spec).ellipse(
        [int(cx - rx * 0.85), int(head_cy - ry * 0.98), int(cx - rx * 0.10), int(head_cy - ry * 0.52)],
        fill=(255, 255, 255, 110),
    )
    head.alpha_composite(spec.filter(ImageFilter.GaussianBlur(radius=S * 0.012)))

    # eyes
    eye_y = head_cy + ry * 0.10
    eye_dx = rx * 0.40
    eye_rx = rx * 0.22
    eye_ry = ry * 0.115
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        hd.ellipse(
            [int(ex - eye_rx), int(eye_y - eye_ry), int(ex + eye_rx), int(eye_y + eye_ry)],
            fill=(246, 243, 240),
        )
        iris_r = eye_ry * 1.18
        iris = _radial_gradient(
            int(iris_r * 2),
            _shade(desc["eye"], 0.26),
            _shade(desc["eye"], -0.32),
            cx=0.40,
            cy=0.34,
        ).convert("RGBA")
        _paste_ellipse(iris, iris, (0, 0, iris_r * 2, iris_r * 2))
        iris_mask = _ellipse_mask(iris.size, (0, 0, iris_r * 2 - 1, iris_r * 2 - 1))
        head.paste(iris, (int(ex - iris_r), int(eye_y - iris_r)), iris_mask)
        hd.ellipse(
            [int(ex - iris_r * 0.40), int(eye_y - iris_r * 0.40), int(ex + iris_r * 0.40), int(eye_y + iris_r * 0.40)],
            fill=(20, 18, 22),
        )
        hd.ellipse(
            [int(ex - iris_r * 0.62), int(eye_y - iris_r * 0.76), int(ex - iris_r * 0.20), int(eye_y - iris_r * 0.32)],
            fill=(255, 255, 255),
        )
        # eyelid shadow + brow
        hd.arc(
            [int(ex - eye_rx * 1.12), int(eye_y - eye_ry * 1.45), int(ex + eye_rx * 1.12), int(eye_y + eye_ry * 1.25)],
            start=190,
            end=350,
            fill=_shade(desc["skin"], -0.26),
            width=max(2, int(S * 0.004)),
        )
        hd.arc(
            [int(ex - eye_rx * 1.2), int(eye_y - eye_ry * 2.3), int(ex + eye_rx * 1.2), int(eye_y - eye_ry * 1.1)],
            start=198,
            end=342,
            fill=_shade(desc["hair"], -0.22),
            width=max(2, int(S * 0.005)),
        )

    # nose shadow + mouth
    hd.ellipse(
        [int(cx - rx * 0.11), int(head_cy + ry * 0.30), int(cx + rx * 0.11), int(head_cy + ry * 0.44)],
        fill=_shade(desc["skin"], -0.18),
    )
    mouth_y = head_cy + ry * 0.58
    if desc["expression"] in ("smiling", "cheerful"):
        hd.arc(
            [int(cx - rx * 0.20), int(mouth_y - ry * 0.12), int(cx + rx * 0.20), int(mouth_y + ry * 0.16)],
            start=20,
            end=160,
            fill=_shade(desc["skin"], -0.34),
            width=max(2, int(S * 0.005)),
        )
    else:
        hd.line(
            [(cx - rx * 0.14, mouth_y), (cx + rx * 0.14, mouth_y)],
            fill=_shade(desc["skin"], -0.34),
            width=max(2, int(S * 0.004)),
        )

    # rim light on the right edge
    rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(rim).arc(
        [int(cx - rx * 1.05), int(head_cy - ry * 1.02), int(cx + rx * 1.05), int(head_cy + ry * 1.02)],
        start=285,
        end=55,
        fill=(232, 240, 255, 150),
        width=max(2, int(S * 0.007)),
    )
    head.alpha_composite(rim.filter(ImageFilter.GaussianBlur(radius=S * 0.005)))

    head = _rotate_layer(head, desc["tilt"] * 0.4, (cx, head_cy))
    head = head.filter(ImageFilter.GaussianBlur(radius=S * 0.0015))
    img.alpha_composite(head)

    # vignette
    vig = Image.new("L", (S, S), 0)
    ImageDraw.Draw(vig).ellipse(
        [int(-S * 0.15), int(-S * 0.15), int(S * 1.15), int(S * 1.15)],
        fill=255,
    )
    vig = vig.filter(ImageFilter.GaussianBlur(radius=S * 0.09))
    dark = Image.new("RGBA", (S, S), (0, 0, 0, 70))
    img = Image.composite(img, Image.alpha_composite(img, dark), vig)

    out = img.convert("RGB").resize((size, size), Image.LANCZOS)
    desc["caption"] = _caption(desc)
    return out, desc


def render(seed: int, size: int = 512, style: str = "anime") -> tuple[Image.Image, dict[str, Any]]:
    """Render one synthetic portrait. ``style`` is ``"anime"`` or ``"3d"``."""
    if style == "anime":
        return render_anime(seed, size)
    if style == "3d":
        return render_3d(seed, size)
    raise ValueError(f"unknown style: {style!r}")
