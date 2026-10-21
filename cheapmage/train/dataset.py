"""Training datasets and the dataset manifest (TRAINING ONLY).

Nothing in this module is imported by the inference engine. It contains:

* :func:`build_synthetic_dataset` - procedurally renders a licence-clean
  synthetic dataset (the *primary* training data for CheapMage).
* :class:`SyntheticStyleDataset` - a ``torch`` dataset over the synthetic
  corpus, filtered by style.
* :class:`CaptionFileDataset` - a ``torch`` dataset over an arbitrary folder
  of ``<image>.jpg`` + ``<image>.txt`` pairs.
* :class:`DatasetManifest` - the provenance / licence ledger that every
  training run records in its report.

Licensing rules enforced here (see ``docs/DATA_LICENSES.md``):

1. Only ``ALLOWED_LICENSES`` may appear in a manifest.
2. Datasets tagged as containing copyrighted characters or real people are
   refused outright.
3. Any caption that combines a minor-related term with a sexual term is
   refused. There is **no** acceptable amount of sexual content involving
   minors in any CheapMage dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .renderers import STYLE_TOKENS, render

__all__ = [
    "DatasetLicenceError",
    "DatasetManifest",
    "CaptionFileDataset",
    "SyntheticStyleDataset",
    "build_synthetic_dataset",
    "sha256_file",
    "sha256_text",
    "assert_dataset_clean",
    "ALLOWED_LICENSES",
    "BANNED_TAGS",
    "MINOR_TERMS",
    "NSFW_TERMS",
    "COPYRIGHT_TERMS",
]

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

#: Licences we are willing to train on.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "ODbL-1.0",
        "MIT",
        "Apache-2.0",
        "public-domain",
        "synthetic-generated",
    }
)

#: Dataset tags that make a corpus unusable, no matter the licence.
BANNED_TAGS: frozenset[str] = frozenset(
    {
        "copyrighted_characters",
        "copyrighted-characters",
        "real_people",
        "real-people",
        "celebrity",
        "famous_person",
        "nsfw_minors",
        "minor_nsfw",
        "loli",
        "shota",
    }
)

#: Terms that indicate the depicted subject may be a minor.
MINOR_TERMS: tuple[str, ...] = (
    "child",
    "children",
    "kid",
    "kids",
    "minor",
    "minors",
    "underage",
    "toddler",
    "infant",
    "baby",
    "preteen",
    "schoolgirl",
    "schoolboy",
    "loli",
    "lolicon",
    "shota",
    "shotacon",
    "little girl",
    "little boy",
)

#: Terms that indicate sexual / explicit content.
NSFW_TERMS: tuple[str, ...] = (
    "nude",
    "naked",
    "nsfw",
    "sexual",
    "sex",
    "porn",
    "explicit",
    "hentai",
    "erotic",
    "orgasm",
    "topless",
    "bottomless",
    "undressed",
    "strip",
    "lingerie",
    "fetish",
    "lewd",
    "ecchi",
)

#: Well known copyrighted brands / franchises that must never appear in a
#: caption. The shipped data is synthetic, so this list should never match.
COPYRIGHT_TERMS: tuple[str, ...] = (
    "pokemon",
    "pikachu",
    "naruto",
    "one piece",
    "dragon ball",
    "ghibli",
    "studio trigger",
    "danbooru",
    "waifu",
    "disney",
    "pixar",
    "marvel",
    "dc comics",
    "batman",
    "superman",
    "coca-cola",
    "coca cola",
    "nike",
    "adidas",
    "apple inc",
    "mickey mouse",
    "elsa",
    "super mario",
)

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class DatasetLicenceError(RuntimeError):
    """Raised when a dataset violates the CheapMage data policy."""


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contains_any(text: str, terms: Sequence[str]) -> list[str]:
    low = text.lower()
    return [term for term in terms if term in low]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """Per-image licence / provenance ledger.

    A manifest is the single artefact that proves *what was trained on*.
    Every training report embeds its SHA-256 so a run can be traced back to
    an exact corpus.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generator: str = "cheapmage.train.dataset"
    seed: int = 0
    base_model: str | None = None
    licence: str = "synthetic-generated"
    notes: str = ""

    # -- construction ------------------------------------------------------
    def add(
        self,
        *,
        image: str | os.PathLike[str],
        caption: str,
        style: str,
        licence: str,
        source: str,
        source_url: str = "",
        caption_file: str | os.PathLike[str] | None = None,
        width: int = 0,
        height: int = 0,
        tags: Iterable[str] = (),
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(image)
        record: dict[str, Any] = {
            "id": path.stem,
            "image": path.name,
            "caption_file": Path(caption_file).name if caption_file else "",
            "caption": caption,
            "style": style,
            "licence": licence,
            "source": source,
            "source_url": source_url,
            "sha256": sha256_file(path) if path.exists() else "",
            "bytes": path.stat().st_size if path.exists() else 0,
            "width": width,
            "height": height,
            "tags": sorted(set(tags)),
        }
        if extra:
            record["extra"] = extra
        self.records.append(record)
        return record

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_utc": self.created_utc,
            "generator": self.generator,
            "seed": self.seed,
            "base_model": self.base_model,
            "licence": self.licence,
            "notes": self.notes,
            "records": self.records,
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DatasetManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            records=list(raw.get("records", [])),
            version=int(raw.get("version", 1)),
            created_utc=str(raw.get("created_utc", "")),
            generator=str(raw.get("generator", "")),
            seed=int(raw.get("seed", 0)),
            base_model=raw.get("base_model"),
            licence=str(raw.get("licence", "")),
            notes=str(raw.get("notes", "")),
        )

    # -- derived information ------------------------------------------------
    def sha256(self) -> str:
        """Stable content hash of the corpus (order independent)."""
        payload = json.dumps(self.records, sort_keys=True, ensure_ascii=False)
        return sha256_text(payload)

    def licence_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in self.records:
            key = str(record.get("licence", "unknown"))
            summary[key] = summary.get(key, 0) + 1
        return dict(sorted(summary.items()))

    def source_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in self.records:
            key = str(record.get("source", "unknown"))
            summary[key] = summary.get(key, 0) + 1
        return dict(sorted(summary.items()))

    def style_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for record in self.records:
            key = str(record.get("style", "unknown"))
            summary[key] = summary.get(key, 0) + 1
        return dict(sorted(summary.items()))

    def total_bytes(self) -> int:
        return int(sum(int(r.get("bytes", 0)) for r in self.records))

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Policy enforcement
# ---------------------------------------------------------------------------


def assert_dataset_clean(
    manifest: DatasetManifest | Iterable[dict[str, Any]],
    *,
    strict_licence: bool = True,
) -> None:
    """Refuse any corpus that breaks the CheapMage data policy.

    Raises :class:`DatasetLicenceError` with an explicit reason.
    """
    records = manifest.records if isinstance(manifest, DatasetManifest) else list(manifest)
    for record in records:
        licence = str(record.get("licence", "")).strip()
        if strict_licence and licence not in ALLOWED_LICENSES:
            raise DatasetLicenceError(
                f"record {record.get('id', '?')!r} has licence {licence!r} which is not in "
                f"{sorted(ALLOWED_LICENSES)}"
            )
        tags = {str(t).lower() for t in record.get("tags", [])}
        banned = tags & BANNED_TAGS
        if banned:
            raise DatasetLicenceError(
                f"record {record.get('id', '?')!r} is tagged {sorted(banned)}; such corpora are refused"
            )
        caption = str(record.get("caption", ""))
        hits = _contains_any(caption, COPYRIGHT_TERMS)
        if hits:
            raise DatasetLicenceError(
                f"record {record.get('id', '?')!r} caption references copyrighted material {hits}"
            )
        minors = _contains_any(caption, MINOR_TERMS)
        nsfw = _contains_any(caption, NSFW_TERMS)
        if minors and nsfw:
            raise DatasetLicenceError(
                f"record {record.get('id', '?')!r} caption contains both minor-related {minors} "
                f"and sexual terms {nsfw}; sexual content involving minors is never allowed"
            )
        if "nsfw" in tags and minors:
            raise DatasetLicenceError(
                f"record {record.get('id', '?')!r} is NSFW and references minors {minors}"
            )


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------


def build_synthetic_dataset(
    out_dir: str | os.PathLike[str],
    n_per_style: int = 1000,
    size: int = 512,
    seed: int = 0,
    styles: Sequence[str] = ("anime", "3d"),
    *,
    licence: str = "synthetic-generated",
    source: str = "cheapmage.train.renderers",
    image_format: str = "jpg",
    manifest_name: str = "dataset_manifest.json",
) -> DatasetManifest:
    """Render a synthetic portrait corpus with matching caption files.

    The result is deterministic: ``seed`` fully determines the corpus. Every
    image gets a ``<name>.txt`` caption file and a manifest entry with the
    licence, provenance and SHA-256 of the image bytes.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        generator="cheapmage.train.renderers.build_synthetic_dataset",
        seed=seed,
        licence=licence,
        notes=(
            "Procedurally rendered portraits. No scraped artwork, no copyrighted "
            "characters, no real people, no sexual content involving minors."
        ),
    )
    suffix = ".jpg" if image_format.lower() in ("jpg", "jpeg") else f".{image_format.lower()}"
    count = 0
    for style_index, style in enumerate(styles):
        style = style.strip().lower()
        # PowerShell turns the bare token ``3d`` into the number ``3``; accept it.
        if style == "3":
            style = "3d"
        if style not in STYLE_TOKENS:
            raise ValueError(f"unknown style {style!r}; expected one of {sorted(STYLE_TOKENS)}")
        style_dir = out / style
        style_dir.mkdir(parents=True, exist_ok=True)
        for index in range(n_per_style):
            image_seed = seed + style_index * 1_000_000 + index
            image, desc = render(image_seed, size=size, style=style)
            name = f"{style}_{index:06d}"
            image_path = style_dir / f"{name}{suffix}"
            image.save(image_path, quality=95)
            caption = desc["caption"]
            caption_path = style_dir / f"{name}.txt"
            caption_path.write_text(caption + "\n", encoding="utf-8")
            manifest.add(
                image=image_path,
                caption_file=caption_path,
                caption=caption,
                style=style,
                licence=licence,
                source=source,
                source_url="",
                width=size,
                height=size,
                tags=["synthetic", "procedural", style],
                extra={"render_seed": image_seed, "attributes": desc},
            )
            count += 1
    assert_dataset_clean(manifest)
    manifest.save(out / manifest_name)
    return manifest


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _to_tensor(image: Image.Image) -> "Any":
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_image(path: Path, size: int | None) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if size and (image.width != size or image.height != size):
        image = image.resize((size, size), Image.LANCZOS)
    return image


class SyntheticStyleDataset(Dataset):
    """Torch dataset over the synthetic corpus described by a manifest.

    Parameters
    ----------
    root:
        Directory containing ``dataset_manifest.json`` (and the images).
    style:
        Only return samples of this style (``"anime"`` / ``"3d"``). ``None``
        returns every style.
    size:
        Optional square resolution the images are resized to.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        style: str | None = None,
        size: int | None = 512,
        *,
        manifest_name: str = "dataset_manifest.json",
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.style = style.lower() if style else None
        self.size = size
        manifest_path = self.root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"dataset manifest not found: {manifest_path}. "
                "Generate one with `python -m cheapmage.train.make_synthetic_data`."
            )
        self.manifest = DatasetManifest.load(manifest_path)
        assert_dataset_clean(self.manifest)
        self.records = [
            r
            for r in self.manifest.records
            if self.style is None or str(r.get("style", "")).lower() == self.style
        ]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"no records for style={self.style!r} in {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, index: int) -> Path:
        record = self.records[index]
        candidate = self.root / str(record.get("image", ""))
        if candidate.exists():
            return candidate
        return self.root / str(record.get("style", "")) / str(record.get("image", ""))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = _load_image(self.image_path(index), self.size)
        return {
            "pixel_values": _to_tensor(image),
            "caption": str(record.get("caption", "")),
            "style": str(record.get("style", "")),
            "id": str(record.get("id", index)),
        }


class CaptionFileDataset(Dataset):
    """Torch dataset over ``<image>.jpg`` + ``<image>.txt`` pairs.

    Used for external (opt-in) corpora materialised on disk. The caption
    file holds one line: the prompt, optionally including the style token.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        size: int | None = 512,
        *,
        style: str | None = None,
        limit: int | None = None,
        extensions: Sequence[str] = _IMAGE_SUFFIXES,
    ) -> None:
        self.root = Path(root)
        self.size = size
        self.style = style
        self.samples: list[tuple[Path, Path]] = []
        for ext in extensions:
            for image_path in sorted(self.root.rglob(f"*{ext}")):
                caption_path = image_path.with_suffix(".txt")
                if caption_path.exists():
                    self.samples.append((image_path, caption_path))
        if limit is not None:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"no <image>+<caption>.txt pairs found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, caption_path = self.samples[index]
        image = _load_image(image_path, self.size)
        caption = caption_path.read_text(encoding="utf-8").strip().splitlines()
        caption = caption[0] if caption else ""
        return {
            "pixel_values": _to_tensor(image),
            "caption": caption,
            "style": self.style or "",
            "id": image_path.stem,
        }
