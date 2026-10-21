"""CheapMage TRAINING-ONLY package.

Everything in ``cheapmage.train`` exists to *produce* weights. It is never
imported by the inference engine, the API layer or the safety layer, and it
must never be imported at package import time by those modules.

Design rules for this package:

* No import-time side effects: importing any module here must not touch the
  network, the filesystem or the GPU.
* No hardcoded absolute paths: paths are resolved from ``REPO_ROOT`` or from
  CLI arguments.
* Every dataset that enters training must carry a licence + provenance record
  (see :class:`cheapmage.train.dataset.DatasetManifest`).
* Synthetic-first: the default (and only shipped) training data is generated
  procedurally by this repository.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "DatasetManifest",
    "DatasetLicenceError",
    "CaptionFileDataset",
    "SyntheticStyleDataset",
    "build_synthetic_dataset",
    "discover_lora_target_modules",
    "load_character_embedding",
]

_LAZY: dict[str, tuple[str, str]] = {
    "DatasetManifest": ("cheapmage.train.dataset", "DatasetManifest"),
    "DatasetLicenceError": ("cheapmage.train.dataset", "DatasetLicenceError"),
    "CaptionFileDataset": ("cheapmage.train.dataset", "CaptionFileDataset"),
    "SyntheticStyleDataset": ("cheapmage.train.dataset", "SyntheticStyleDataset"),
    "build_synthetic_dataset": ("cheapmage.train.dataset", "build_synthetic_dataset"),
    "discover_lora_target_modules": (
        "cheapmage.train.train_style_lora",
        "discover_lora_target_modules",
    ),
    "load_character_embedding": (
        "cheapmage.train.train_character_embedding",
        "load_character_embedding",
    ),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy loader
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'cheapmage.train' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(__all__) | set(globals()))
