"""Thin wrapper around the CheapMage engine singleton.

Responsibilities
----------------
* lazy import of ``cheapmage.engine.get_engine()`` (the engine may live in a
  different process / may not be installed at all),
* bound the number of concurrent generations (``api.max_concurrency``),
* enforce the wall-clock budget (``api.timeout_seconds``),
* expose ``health()`` and ``memory_snapshot()``,
* fall back to a deterministic in-process placeholder when the real model is
  unavailable so the service stays usable (and testable) offline.

Nothing in here does image generation logic beyond the placeholder fallback.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from ..config import ConfigStore
from ..cost import estimate_cost
from ..engine.types import (
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
    ModelNotLoadedError,
)
from ..styles import detect_style
from .config_loader import load_config_store

log = logging.getLogger("cheapmage.api.engine")


class QueueFullError(RuntimeError):
    """Raised when the generation queue is saturated past the time budget."""


# ----------------------------------------------------------------------
# placeholder image helpers (deterministic, dependency free)
# ----------------------------------------------------------------------
def _seed_color(seed: int) -> tuple[int, int, int]:
    h = (int(seed) & 0xFFFFFFFF) or 1
    r = (h * 2654435761) & 0xFF
    g = (h * 40503) & 0xFF
    b = (h * 2246822519) & 0xFF
    return (r, g, b)


def _png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Minimal, dependency free PNG encoder (8 bit RGB, solid colour)."""
    row = bytes(rgb) * width
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # filter type 0
        raw.extend(row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _encode_placeholder(width: int, height: int, seed: int, fmt: str) -> tuple[bytes, str]:
    """Return ``(image_bytes, image_format)`` for the placeholder image."""
    rgb = _seed_color(seed)
    if fmt == "webp":
        try:  # Pillow is an optional nicety here, never a hard requirement.
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (width, height), rgb).save(buf, format="WEBP", quality=90)
            return buf.getvalue(), "webp"
        except Exception:  # pragma: no cover - depends on optional Pillow build
            pass
    return _png_bytes(width, height, rgb), "png"


class PlaceholderEngine:
    """In-process fallback engine.

    ``PLACEHOLDER`` - this is **not** a real diffusion model.  It renders a
    deterministic solid colour image derived from the seed so that the API can
    be exercised (and health-checked) without the real weights being present.
    """

    is_placeholder = True

    def __init__(self, store: ConfigStore) -> None:
        self._store = store
        self._started = time.time()

    # -- engine protocol ------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        gen = request.generation
        started = time.monotonic()
        seed = int(gen.seed)
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        image_bytes, image_format = _encode_placeholder(
            gen.width, gen.height, seed, gen.output_format
        )
        duration = max(time.monotonic() - started, 1e-4)
        device = str(self._store.config.hardware.device)
        if device == "auto":
            device = "cpu"
        estimate = estimate_cost(self._store.config.cost, device, duration)
        return GenerationResult(
            image_bytes=image_bytes,
            image_format=image_format,
            seed=seed,
            duration_seconds=duration,
            device=device,
            memory_mb={"vram_mb": 0.0, "ram_mb": 1.0},
            cost=estimate.to_dict(),
            moderation={
                "blocked": False,
                "placeholder": True,
                "model_loaded": False,
                "note": "placeholder image - the real CheapMage model is not loaded",
            },
            style_used=detect_style(gen.prompt),
            prompt_used=gen.prompt,
            negative_prompt_used=gen.negative_prompt,
            steps=gen.steps,
            sampler=gen.sampler,
            cfg_scale=gen.cfg_scale,
            width=gen.width,
            height=gen.height,
            metadata={"placeholder": True},
        )

    def health(self) -> dict[str, Any]:
        return {
            "model_loaded": False,
            "device": "cpu",
            "model_size_mb": 0.0,
            "load_seconds": 0.0,
            "vram_mb": 0.0,
            "ram_mb": 1.0,
            "peak_vram_mb": 0.0,
            "peak_ram_mb": 1.0,
            "uptime_seconds": time.time() - self._started,
            "placeholder": True,
        }

    def memory_snapshot(self) -> dict[str, Any]:
        return {"vram_mb": 0.0, "ram_mb": 1.0, "peak_vram_mb": 0.0, "peak_ram_mb": 1.0}


class EngineClient:
    """Owns the engine singleton and all of the runtime guard rails."""

    def __init__(
        self,
        store: ConfigStore | None = None,
        engine: Any | None = None,
        allow_placeholder: bool = True,
    ) -> None:
        self._store = store or load_config_store()
        api = self._store.config.api
        self._timeout = float(api.timeout_seconds or 20)
        self._max_concurrency = max(1, int(api.max_concurrency or 1))
        self._semaphore = threading.Semaphore(self._max_concurrency)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._waiting = 0
        self._engine = engine
        self._engine_probed = engine is not None
        self._allow_placeholder = allow_placeholder
        self._started = time.time()

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def store(self) -> ConfigStore:
        return self._store

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._waiting

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def engine(self) -> Any | None:
        """The engine object (may be ``None`` when the real model is absent)."""
        self._probe_engine()
        return self._engine

    # ------------------------------------------------------------------
    # engine discovery
    # ------------------------------------------------------------------
    def _probe_engine(self) -> Any | None:
        if self._engine is not None:
            return self._engine
        if self._engine_probed:
            return None
        self._engine_probed = True
        try:
            from ..engine import get_engine  # noqa: PLC0415 - lazy on purpose

            self._engine = get_engine()
            log.info("cheapmage engine loaded: %s", type(self._engine).__name__)
        except Exception as exc:  # pragma: no cover - depends on engine package
            log.warning("cheapmage engine unavailable (%s); using placeholder", exc)
            self._engine = None
        return self._engine

    def set_engine(self, engine: Any | None) -> None:
        """Inject an engine (used by the tests and by the bench)."""
        self._engine = engine
        self._engine_probed = True

    @staticmethod
    def _model_available(engine: Any) -> bool:
        """Is the real model package present on disk?

        The engine loads lazily, so the only reliable pre-flight check is the
        model package itself.  When it is missing the API serves the
        placeholder image instead of exploding with a load error.
        """
        package_dir = getattr(engine, "package_dir", None)
        if not callable(package_dir):
            return True
        try:
            directory = Path(package_dir())
        except Exception:  # pragma: no cover - engine dependent
            return True
        return (directory / "unet").exists()

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one generation, honouring the concurrency and time budgets.

        The concurrency slot is held by the *worker*, not by the request: when
        a generation times out the engine keeps running in the background and
        the slot stays occupied until the engine is really done.  That keeps
        the "at most ``max_concurrency`` generations in flight" guarantee.
        """
        deadline = time.monotonic() + self._timeout

        with self._lock:
            self._waiting += 1
        acquired = self._semaphore.acquire(timeout=self._timeout)
        with self._lock:
            self._waiting -= 1
        if not acquired:
            raise QueueFullError(
                f"generation queue is full ({self._max_concurrency} slots, "
                f"{self._timeout:.0f}s budget exhausted)"
            )

        slot_lock = threading.Lock()
        slot_released = False

        def _release_slot() -> None:
            nonlocal slot_released
            with slot_lock:
                if slot_released:
                    return
                slot_released = True
                self._semaphore.release()

        with self._lock:
            self._in_flight += 1
        state: dict[str, bool] = {"worker_started": False}
        try:
            return self._generate_locked(request, deadline, _release_slot, state)
        except BaseException:
            # the worker never started -> give the slot back
            if not state["worker_started"]:
                _release_slot()
                with self._lock:
                    self._in_flight -= 1
            raise

    def _generate_locked(
        self,
        request: GenerationRequest,
        deadline: float,
        release_slot: Any,
        state: dict[str, bool],
    ) -> GenerationResult:
        engine = self._probe_engine()
        if engine is None or not self._model_available(engine):
            if not self._allow_placeholder:
                raise ModelNotLoadedError("cheapmage model is not loaded")
            engine = PlaceholderEngine(self._store)

        runner = getattr(engine, "generate", None) or getattr(engine, "run", None)
        if runner is None:
            raise ModelNotLoadedError(
                f"engine {type(engine).__name__} exposes no generate() method"
            )

        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["result"] = runner(request)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                box["error"] = exc
            finally:
                with self._lock:
                    self._in_flight -= 1
                release_slot()

        worker = threading.Thread(target=_target, name="cheapmage-generate", daemon=True)
        state["worker_started"] = True
        worker.start()
        remaining = max(deadline - time.monotonic(), 0.0)
        worker.join(timeout=remaining)
        if worker.is_alive():
            raise GenerationTimeoutError(
                f"generation exceeded the {self._timeout:.0f}s budget"
            )
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, ModelNotLoadedError) and self._allow_placeholder:
                return PlaceholderEngine(self._store).generate(request)
            raise exc
        return box["result"]

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Never raises - a missing model is a valid health state."""
        cfg = self._store.config
        info: dict[str, Any] = {}
        engine = self._probe_engine()
        if engine is not None:
            probe = getattr(engine, "health", None)
            if callable(probe):
                try:
                    info = probe() or {}
                except Exception as exc:  # pragma: no cover - engine dependent
                    log.warning("engine health probe failed: %s", exc)
                    info = {}
        elif not self._engine_probed or self._engine is None:
            # the real engine is missing -> placeholder state
            info = PlaceholderEngine(self._store).health()

        def _num(key: str, default: float = 0.0) -> float:
            value = info.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        device = str(info.get("device") or cfg.hardware.device)
        if device == "auto":
            device = "cpu"
        uptime = _num("uptime_seconds", 0.0) or (time.time() - self._started)
        return {
            "status": "ok",
            "model_loaded": bool(info.get("model_loaded", False)),
            "device": device,
            "model_size_mb": _num("model_size_mb"),
            "load_seconds": _num("load_seconds"),
            "vram_mb": _num("vram_mb"),
            "ram_mb": _num("ram_mb"),
            "peak_vram_mb": _num("peak_vram_mb"),
            "peak_ram_mb": _num("peak_ram_mb"),
            "dtype": str(info.get("dtype") or cfg.hardware.dtype),
            "uptime_seconds": max(uptime, 0.0),
            "concurrency_limit": self._max_concurrency,
            "queue_depth": self.queue_depth,
        }

    def memory_snapshot(self) -> dict[str, Any]:
        engine = self._probe_engine()
        if engine is not None:
            probe = getattr(engine, "memory_snapshot", None)
            if callable(probe):
                try:
                    return dict(probe() or {})
                except Exception as exc:  # pragma: no cover - engine dependent
                    log.warning("engine memory snapshot failed: %s", exc)
        return {"vram_mb": 0.0, "ram_mb": 0.0, "peak_vram_mb": 0.0, "peak_ram_mb": 0.0}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def encode_base64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")
