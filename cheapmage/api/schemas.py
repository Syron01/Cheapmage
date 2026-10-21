"""Pydantic v2 request/response models for the CheapMage HTTP API.

The models here are the *wire* contract.  They mirror the generation knobs in
:mod:`cheapmage.config` but are stricter about what a client may send.

Validation rules
----------------
* Structural violations (bad width, steps out of range, ``batch_size > 1``)
  raise a plain pydantic error -> HTTP 422.
* A non-English prompt raises a custom error type (``non_english_prompt``)
  which the app maps to HTTP 400 (it is a policy rejection, not a schema
  violation).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from ..config import ConsistencyConfig, GenerationConfig
from ..styles import NEGATIVE_PROMPT_DEFAULT, looks_english

StylePreset = Literal["anime", "3d", "auto"]
Sampler = Literal["euler_a", "dpm++_2m", "lcm"]
AspectRatio = Literal["1:1", "2:3", "3:2", "9:16", "16:9"]
OutputFormat = Literal["png", "webp"]
QualityPreset = Literal["fast", "balanced", "quality"]
NsfwMode = Literal["off", "on"]

ALLOWED_SIZES = (512, 640, 768)
MAX_PROMPT_CHARS = 2000


class ConsistencyRequest(BaseModel):
    """Character consistency knobs (seed lock, reference image, embedding)."""

    model_config = ConfigDict(extra="forbid")

    seed_lock: bool = True
    reference_image: str | None = None
    reference_strength: float = Field(0.5, ge=0.0, le=1.0)
    character_embedding: str | None = None

    @field_validator("reference_image")
    @classmethod
    def _check_reference(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if v.lower().startswith("data:"):
            if not v.lower().startswith("data:image/"):
                raise PydanticCustomError(
                    "invalid_reference_image",
                    "reference_image data URL must start with 'data:image/'",
                )
            if "," not in v:
                raise PydanticCustomError(
                    "invalid_reference_image",
                    "reference_image data URL is malformed (missing payload)",
                )
            return v
        return v

    @field_validator("character_embedding")
    @classmethod
    def _check_embedding(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    def to_config(self) -> ConsistencyConfig:
        return ConsistencyConfig(
            seed_lock=self.seed_lock,
            reference_image=self.reference_image,
            reference_strength=self.reference_strength,
            character_embedding=self.character_embedding,
        )


class GenerateRequest(BaseModel):
    """A single image generation request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    negative_prompt: str = NEGATIVE_PROMPT_DEFAULT
    style_preset: StylePreset = "auto"
    width: int = 512
    height: int = 512
    aspect_ratio: AspectRatio = "1:1"
    steps: int = Field(20, ge=8, le=30)
    cfg_scale: float = Field(6.0, ge=1.0, le=12.0)
    sampler: Sampler = "lcm"
    seed: int = -1
    batch_size: int = Field(1, ge=1, le=1)
    output_format: OutputFormat = "webp"
    quality_preset: QualityPreset = "balanced"
    consistency: ConsistencyRequest = Field(default_factory=ConsistencyRequest)

    # ------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------
    @field_validator("prompt")
    @classmethod
    def _check_prompt(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("prompt must not be empty")
        if len(v) > MAX_PROMPT_CHARS:
            raise PydanticCustomError(
                "prompt_too_long",
                "prompt must be at most {limit} characters",
                {"limit": MAX_PROMPT_CHARS},
            )
        if not looks_english(v):
            raise PydanticCustomError(
                "non_english_prompt",
                "prompt must be written in English (ASCII/Latin letters only)",
            )
        return v

    @field_validator("negative_prompt")
    @classmethod
    def _check_negative(cls, v: str | None) -> str:
        v = (v or "").strip()
        return v or NEGATIVE_PROMPT_DEFAULT

    @field_validator("width", "height")
    @classmethod
    def _check_size(cls, v: int) -> int:
        if v not in ALLOWED_SIZES:
            raise ValueError(
                f"width/height must be one of {ALLOWED_SIZES}"
            )
        return v

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, v: int) -> int:
        if v < -1:
            raise ValueError("seed must be >= -1 (-1 means random)")
        return v

    # ------------------------------------------------------------------
    # conversion
    # ------------------------------------------------------------------
    def to_generation_config(
        self,
        quality_presets: dict[str, Any] | None = None,
    ) -> GenerationConfig:
        """Build the engine-facing :class:`GenerationConfig`.

        ``steps`` / ``cfg_scale`` / ``sampler`` that the caller did *not* send
        explicitly are taken from the selected ``quality_preset``.
        """
        preset = quality_presets.get(self.quality_preset) if quality_presets else None
        explicit = self.model_fields_set

        def pick(field: str, fallback: Any) -> Any:
            if field in explicit:
                return getattr(self, field)
            if preset is not None and hasattr(preset, field):
                return getattr(preset, field)
            return fallback

        return GenerationConfig(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            style_preset=self.style_preset,
            width=self.width,
            height=self.height,
            aspect_ratio=self.aspect_ratio,
            steps=int(pick("steps", self.steps)),
            cfg_scale=float(pick("cfg_scale", self.cfg_scale)),
            sampler=str(pick("sampler", self.sampler)),
            seed=self.seed,
            batch_size=self.batch_size,
            output_format=self.output_format,
            quality_preset=self.quality_preset,
        )


class GenerateResponse(BaseModel):
    """Successful generation result (image is base64 encoded)."""

    model_config = ConfigDict(extra="ignore")

    image: str
    image_format: str
    seed: int
    duration_seconds: float
    device: str
    memory_mb: dict[str, float] = Field(default_factory=dict)
    estimated_cost_tl: float = 0.0
    cost: dict[str, Any] = Field(default_factory=dict)
    moderation: dict[str, Any] = Field(default_factory=dict)
    style_used: str = "anime"
    steps: int = 20
    sampler: str = "lcm"
    cfg_scale: float = 6.0
    width: int = 512
    height: int = 512
    request_id: str = ""


class HealthResponse(BaseModel):
    """Service health - always answerable, even when the model is not loaded."""

    model_config = ConfigDict(extra="ignore")

    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    model_size_mb: float = 0.0
    load_seconds: float = 0.0
    vram_mb: float = 0.0
    ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    dtype: str = "int8"
    uptime_seconds: float = 0.0
    concurrency_limit: int = 2
    queue_depth: int = 0


class ConfigResponse(BaseModel):
    """The public (redacted) configuration."""

    model_config = ConfigDict(extra="ignore")

    config: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class SafetyUpdateRequest(BaseModel):
    """Body of ``POST /config`` - safety section only."""

    model_config = ConfigDict(extra="forbid")

    nsfw_mode: NsfwMode | None = None
    nsfw_requires_age_verification: bool | None = None
    moderation_layer: bool | None = None
    real_person_likeness: Literal["block", "allow"] | None = None
    copyrighted_characters: Literal["block", "allow"] | None = None
    blocked_response: str | None = None

    def changes(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump(exclude_none=True).items()}


class ErrorResponse(BaseModel):
    """Uniform error body."""

    model_config = ConfigDict(extra="allow")

    error: str
    code: str = "error"
    request_id: str | None = None
    moderation: dict[str, Any] | None = None
