"""Config access for the bench harness.

``config/config.yaml`` is a contract file and must not be modified by the
bench. It currently contains a value that YAML 1.1 resolves to a *boolean*
rather than a string::

    safety:
      nsfw_mode: off            # YAML 1.1 -> False

With the default ``yaml.safe_load`` this becomes ``False`` and
``cheapmage.config.AppConfig`` fails validation
(``Input should be 'off' or 'on'``). The bench therefore loads the YAML with
a loader whose boolean resolver only accepts ``true``/``false`` (and their
capitalised variants), so ``off``/``on``/``yes``/``no`` stay strings.

This module is **bench-only**: it never changes the behaviour of the shipped
runtime. The upstream fix belongs in the config loader used by
``cheapmage.config`` - until then, the engine itself may still be unable to
start (see "Known blockers" in ``docs/REPORT.md``).
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig, DEFAULT_CONFIG_PATH

_BOOL_TAG = "tag:yaml.org,2002:bool"
_STRICT_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


class BenchConfigLoader(yaml.SafeLoader):
    """SafeLoader that only resolves ``true``/``false`` as booleans."""


BenchConfigLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
BenchConfigLoader.add_implicit_resolver(
    _BOOL_TAG, _STRICT_BOOL_RE, list("tTfF")
)

_STATE: dict[str, Any] = {"config": None, "path": None}


def config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Path of the config file the bench should use."""
    if path:
        return Path(path)
    env = os.environ.get("CHEAPMAGE_CONFIG")
    if env:
        return Path(env)
    return Path(DEFAULT_CONFIG_PATH)


def load_raw(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Parse the YAML config with the strict boolean resolver."""
    p = config_path(path)
    raw = yaml.load(p.read_text(encoding="utf-8"), Loader=BenchConfigLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file does not contain a mapping: {p}")
    return raw


def get_bench_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Return the validated config, cached per path."""
    p = config_path(path)
    key = str(p.resolve())
    if _STATE["config"] is not None and _STATE["path"] == key:
        return _STATE["config"]
    cfg = AppConfig.model_validate(load_raw(p))
    _STATE["config"] = cfg
    _STATE["path"] = key
    return cfg


def set_nsfw_mode(mode: str, path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Flip ``safety.nsfw_mode`` on the cached config (bench-local copy)."""
    if mode not in ("on", "off"):
        raise ValueError("nsfw_mode must be 'on' or 'off'")
    cfg = get_bench_config(path)
    data = cfg.model_dump()
    data["safety"]["nsfw_mode"] = mode
    updated = AppConfig.model_validate(data)
    _STATE["config"] = updated
    return updated
