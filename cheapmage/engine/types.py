"""Shared request/result contracts between the engine, the API and the bench."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import GenerationConfig, ConsistencyConfig


@dataclass
class GenerationRequest:
    generation: GenerationConfig
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    user_id: str | None = None
    age_verified: bool = False

    def merged(self) -> "GenerationRequest":
        return self


@dataclass
class GenerationResult:
    image_bytes: bytes
    image_format: str
    seed: int
    duration_seconds: float
    device: str
    memory_mb: dict[str, float]
    cost: dict[str, Any]
    moderation: dict[str, Any]
    style_used: str
    prompt_used: str
    negative_prompt_used: str
    steps: int
    sampler: str
    cfg_scale: float
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelNotLoadedError(RuntimeError):
    pass


class GenerationTimeoutError(TimeoutError):
    pass


class SafetyError(RuntimeError):
    def __init__(self, message: str, code: str = "blocked") -> None:
        super().__init__(message)
        self.code = code
