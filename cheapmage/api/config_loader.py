"""Resilient configuration loading for the API layer.

``config/config.yaml`` stores ``safety.nsfw_mode: off``.  PyYAML implements
YAML 1.1, where the bare words ``off``/``on``/``yes``/``no`` are **booleans**,
so the file as-is fails ``AppConfig.model_validate`` with

    safety.nsfw_mode
      Input should be 'off' or 'on' [type=literal_error, input_value=False]

The configuration module is a contract owned by another team, so the API layer
does not patch it.  Instead it sanitises the offending value into a private
copy of the file (in the temp directory) and loads that.  When the upstream
loader learns about YAML 1.2 (or the file is quoted) this fallback simply
never triggers.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..config import DEFAULT_CONFIG_PATH, ConfigStore
from ..config import get_config_store as _global_get_config_store

_FALLBACK_STORE: ConfigStore | None = None


def _sanitize(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``(raw, changed)`` with YAML 1.1 booleans coerced to strings."""
    changed = False
    safety = raw.get("safety")
    if isinstance(safety, dict):
        mode = safety.get("nsfw_mode")
        if isinstance(mode, bool):
            safety["nsfw_mode"] = "on" if mode else "off"
            changed = True
    return raw, changed


def _fallback_store(source: Path) -> ConfigStore:
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file is not a mapping: {source}")
    raw, changed = _sanitize(raw)
    if not changed:
        raise ValueError(f"config file could not be sanitised: {source}")
    target = Path(tempfile.gettempdir()) / "cheapmage-config-sanitized.yaml"
    target.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return ConfigStore(target)


def load_config_store(path: str | os.PathLike[str] | None = None) -> ConfigStore:
    """Return the active :class:`ConfigStore`, tolerating YAML 1.1 booleans."""
    global _FALLBACK_STORE
    try:
        return _global_get_config_store(path)
    except Exception as exc:
        if path is None and _FALLBACK_STORE is not None:
            return _FALLBACK_STORE
        source = Path(path or os.environ.get("CHEAPMAGE_CONFIG", DEFAULT_CONFIG_PATH))
        try:
            store = _fallback_store(source)
        except Exception:
            raise exc
        if path is None:
            _FALLBACK_STORE = store
        return store


__all__ = ["load_config_store"]
