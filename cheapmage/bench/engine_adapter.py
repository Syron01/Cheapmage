"""Thin adapter between the bench harness and ``cheapmage.engine``.

The bench must not import the engine at module import time (a missing engine
must degrade gracefully, not crash the harness), and it must not hard-code a
single engine signature while the engine is still being written.

This module therefore:

* imports the engine lazily,
* verifies that the model weights are actually present on disk,
* tries a small number of documented call signatures and records which one
  worked (``call_signature``), so the report tells us exactly what contract
  the engine exposes today.

Expected (preferred) engine contract::

    from cheapmage.engine import get_engine
    engine = get_engine()
    result = engine.generate(GenerationRequest(generation=GenerationConfig(...)))
    # result: GenerationResult with .image_bytes, .duration_seconds, .memory_mb,
    #         .cost, .moderation, .seed, ...
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import directory_size_bytes


@dataclass
class EngineUnavailable(Exception):
    """Raised when the engine (or its weights) cannot be used."""

    reason: str
    hint: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.reason}{(' | ' + self.hint) if self.hint else ''}"


@dataclass
class EngineHandle:
    """A loaded engine plus the metadata we need for the report."""

    engine: Any
    device: str | None = None
    model_dir: str | None = None
    model_bytes: int = 0
    import_path: str | None = None
    call_signature: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def model_weights_present(model_dir: str | Path) -> bool:
    """True when ``model_dir`` exists and holds at least one weight file."""
    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        return False
    for dirpath, _dirnames, filenames in os_walk(root):
        for name in filenames:
            if name.endswith((".safetensors", ".bin", ".ckpt", ".pt", ".pth")):
                return True
    return False


def os_walk(root: Path):
    """Tiny wrapper so the import list stays obvious."""
    import os

    return os.walk(root)


def load_engine(config_path: str | Path | None = None) -> EngineHandle:
    """Load the engine, or raise :class:`EngineUnavailable` with a clear reason.

    Never downloads anything: if the weights are missing we say so and stop.
    """
    from .config_access import get_bench_config

    cfg = get_bench_config()
    model_dir = cfg.model_dir
    size = directory_size_bytes(model_dir)

    if not Path(model_dir).exists():
        raise EngineUnavailable(
            f"model weights directory not found: {model_dir}",
            "hint: train/convert the student model first, or point "
            "`model_dir` at an existing checkpoint.",
        )
    if size == 0:
        raise EngineUnavailable(
            f"model weights directory is empty: {model_dir}",
            "hint: the pipeline weights (.safetensors) have not been written yet.",
        )

    try:
        module = importlib.import_module("cheapmage.engine")
    except Exception as exc:  # ImportError and anything the engine raises
        raise EngineUnavailable(
            f"cheapmage.engine is not importable yet ({exc.__class__.__name__}: {exc})",
            "hint: the engine package is still being implemented; run "
            "`python -m cheapmage.bench.bench_speed --requests 100 --warmup 5` "
            "once it lands.",
        ) from exc

    get_engine = getattr(module, "get_engine", None)
    if get_engine is None:
        raise EngineUnavailable(
            "cheapmage.engine does not expose get_engine()",
            "hint: expected `cheapmage.engine.get_engine()` to return the engine.",
        )

    try:
        engine = get_engine()
    except Exception as exc:
        raise EngineUnavailable(
            f"get_engine() failed ({exc.__class__.__name__}: {exc})",
            "hint: the weights may be incomplete or the device may be unavailable.",
        ) from exc

    device = getattr(engine, "device", None)
    if device is not None and not isinstance(device, str):
        device = str(device)

    return EngineHandle(
        engine=engine,
        device=device,
        model_dir=str(model_dir),
        model_bytes=size,
        import_path=getattr(module, "__file__", None),
    )


def generate(
    handle: EngineHandle,
    *,
    prompt: str,
    negative_prompt: str = "",
    style_preset: str = "auto",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 6.0,
    sampler: str = "lcm",
    seed: int = -1,
    output_format: str = "webp",
    quality_preset: str = "balanced",
    reference_image: str | None = None,
    reference_strength: float = 0.5,
    character_embedding: str | None = None,
    seed_lock: bool = True,
    age_verified: bool = False,
    timeout_seconds: float | None = 20.0,
) -> tuple[Any, float]:
    """Run one generation and return ``(result, measured_wall_seconds)``.

    ``measured_wall_seconds`` is the wall clock time *we* observed; the engine
    also reports its own ``duration_seconds``. Both are kept so the benchmark
    can cross-check them instead of trusting a single source.
    """
    from ..config import ConsistencyConfig, GenerationConfig
    from ..engine.types import GenerationRequest

    generation = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        style_preset=style_preset,  # type: ignore[arg-type]
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,  # type: ignore[arg-type]
        seed=seed,
        output_format=output_format,  # type: ignore[arg-type]
        quality_preset=quality_preset,  # type: ignore[arg-type]
    )
    consistency = ConsistencyConfig(
        seed_lock=seed_lock,
        reference_image=reference_image,
        reference_strength=reference_strength,
        character_embedding=character_embedding,
    )
    request = GenerationRequest(
        generation=generation, consistency=consistency, age_verified=age_verified
    )

    engine = handle.engine
    started = time.perf_counter()
    result = _call_engine(engine, request, generation)
    elapsed = time.perf_counter() - started

    if timeout_seconds and elapsed > timeout_seconds:
        from ..engine.types import GenerationTimeoutError

        raise GenerationTimeoutError(
            f"generation took {elapsed:.2f}s, budget is {timeout_seconds:.2f}s"
        )
    return result, elapsed


def _call_engine(engine: Any, request: Any, generation: Any) -> Any:
    """Try the documented call signatures, record the one that worked."""
    attempts: list[tuple[str, Any]] = [
        ("generate(GenerationRequest)", lambda: engine.generate(request)),
        (
            "generate(**generation_fields)",
            lambda: engine.generate(**request.generation.model_dump()),
        ),
        (
            "generate(prompt=..., **fields)",
            lambda: engine.generate(
                prompt=generation.prompt,
                negative_prompt=generation.negative_prompt,
                style_preset=generation.style_preset,
                width=generation.width,
                height=generation.height,
                steps=generation.steps,
                cfg_scale=generation.cfg_scale,
                sampler=generation.sampler,
                seed=generation.seed,
                output_format=generation.output_format,
                quality_preset=generation.quality_preset,
            ),
        ),
    ]
    last_error: Exception | None = None
    for signature, call in attempts:
        try:
            return call()
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "engine.generate() did not accept any known call signature "
        f"(last error: {last_error})"
    )


def result_fields(result: Any) -> dict[str, Any]:
    """Extract the fields the bench reports from a ``GenerationResult``."""
    if isinstance(result, dict):
        get = result.get
    else:
        def get(key, default=None):  # type: ignore[misc]
            return getattr(result, key, default)

    duration = get("duration_seconds")
    if duration is None:
        duration = get("duration")
    memory = get("memory_mb") or {}
    if not isinstance(memory, dict):
        memory = {"raw": str(memory)}
    return {
        "seed": get("seed"),
        "duration_seconds": duration,
        "device": get("device"),
        "memory_mb": memory,
        "cost": get("cost") or {},
        "moderation": get("moderation") or {},
        "style_used": get("style_used"),
        "steps": get("steps"),
        "sampler": get("sampler"),
        "cfg_scale": get("cfg_scale"),
        "width": get("width"),
        "height": get("height"),
        "image_bytes_len": len(get("image_bytes") or b""),
    }
