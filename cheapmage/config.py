"""Typed configuration for CheapMage.

The YAML file in ``config/`` is the single source of truth for runtime knobs.
Only an authorized admin may flip ``safety.nsfw_mode``; prompt text can never
influence it.
"""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

StylePreset = Literal["anime", "3d", "auto"]
Sampler = Literal["euler_a", "dpm++_2m", "lcm"]
OutputFormat = Literal["png", "webp"]
QualityPreset = Literal["fast", "balanced", "quality"]
Device = Literal["auto", "cpu", "cuda", "mps"]


class HardwareConfig(BaseModel):
    device: Device = "auto"
    max_vram_mb: int = 4096
    max_ram_mb: int = 2048
    model_offload: bool = True
    dtype: Literal["int8", "fp16", "fp32"] = "int8"

    @field_validator("max_vram_mb", "max_ram_mb")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("memory budgets must be positive")
        return v


class GenerationConfig(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    style_preset: StylePreset = "auto"
    width: int = 512
    height: int = 512
    aspect_ratio: Literal["1:1", "2:3", "3:2", "9:16", "16:9"] = "1:1"
    steps: int = Field(20, ge=8, le=30)
    cfg_scale: float = Field(6.0, ge=1.0, le=12.0)
    sampler: Sampler = "lcm"
    seed: int = -1
    batch_size: int = Field(1, ge=1, le=1)
    output_format: OutputFormat = "webp"
    quality_preset: QualityPreset = "balanced"

    @field_validator("width", "height")
    @classmethod
    def _valid_size(cls, v: int) -> int:
        if v not in (512, 640, 768):
            raise ValueError("width/height must be one of 512, 640, 768")
        return v


class ConsistencyConfig(BaseModel):
    seed_lock: bool = True
    reference_image: str | None = None
    reference_strength: float = Field(0.5, ge=0.0, le=1.0)
    character_embedding: str | None = None


class QualityPresetSpec(BaseModel):
    steps: int = Field(20, ge=8, le=30)
    cfg_scale: float = Field(6.0, ge=1.0, le=12.0)
    sampler: Sampler = "lcm"
    vae_tiling: bool = False


class CostConfig(BaseModel):
    target_per_image_try: float = 0.01
    log_cost_per_request: bool = True
    gpu_hourly_tl: float = 4.50
    local_gpu_hourly_tl: float = 0.0
    electricity_kwh_price: float = 2.5
    gpu_power_watts: int = 120
    cloud_utilization: float = 0.35


class SafetyConfig(BaseModel):
    nsfw_mode: Literal["off", "on"] = "off"
    nsfw_requires_age_verification: bool = True
    nsfw_allowed_for: str = "age-verified adult users only"
    moderation_layer: bool = True
    real_person_likeness: Literal["block", "allow"] = "block"
    copyrighted_characters: Literal["block", "allow"] = "block"
    blocked_response: str = (
        "Request blocked by the content safety policy. The generated image was not returned."
    )


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    timeout_seconds: int = 20
    max_concurrency: int = 2
    admin_token: str = "change-me"
    expose_config: bool = True


class AppConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    quality_presets: dict[str, QualityPresetSpec] = Field(default_factory=dict)
    cost: CostConfig = Field(default_factory=CostConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    model_dir: str = str(REPO_ROOT / "models" / "cheapmage-v1")
    source_path: str = str(DEFAULT_CONFIG_PATH)


class AppConfigError(RuntimeError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class ConfigStore:
    """Thread-safe holder for the active configuration."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path or os.environ.get("CHEAPMAGE_CONFIG", DEFAULT_CONFIG_PATH))
        self._lock = threading.RLock()
        self._config = self._load()

    def _load(self) -> AppConfig:
        if not self.path.exists():
            raise AppConfigError(f"config file not found: {self.path}")
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(raw)

    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    def reload(self) -> AppConfig:
        with self._lock:
            self._config = self._load()
            return self._config

    def update_safety(self, **changes: Any) -> SafetyConfig:
        """Only safety settings may be mutated at runtime (admin-only path)."""
        allowed = {"nsfw_mode", "nsfw_requires_age_verification", "moderation_layer",
                   "real_person_likeness", "copyrighted_characters", "blocked_response"}
        unknown = set(changes) - allowed
        if unknown:
            raise AppConfigError(f"not updatable via API: {sorted(unknown)}")
        with self._lock:
            data = self._config.model_dump()
            data["safety"].update(changes)
            self._config = AppConfig.model_validate(data)
            return self._config.safety

    def set_nsfw_mode(self, mode: str) -> SafetyConfig:
        if mode not in ("on", "off"):
            raise AppConfigError("nsfw_mode must be 'on' or 'off'")
        return self.update_safety(nsfw_mode=mode)

    def public_dict(self) -> dict[str, Any]:
        with self._lock:
            data = self._config.model_dump()
        data.pop("api", None)
        return data

    def dump_yaml(self) -> str:
        with self._lock:
            return yaml.safe_dump(self._config.model_dump(), sort_keys=False, allow_unicode=True)


_store: ConfigStore | None = None


def get_config_store(path: str | os.PathLike[str] | None = None) -> ConfigStore:
    global _store
    if _store is None:
        _store = ConfigStore(path)
    return _store


def get_config() -> AppConfig:
    return get_config_store().config
