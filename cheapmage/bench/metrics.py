"""Reusable measurement helpers for the CheapMage bench harness.

Design rules that every benchmark in this package obeys:

1. **No fabrication.** A helper either returns a real measurement or raises /
   returns ``None``. Benchmarks translate "not measurable yet" into an
   explicit ``TODO - pending run`` marker plus the exact command to run.
2. **No downloads at import time.** ``torch``, ``psutil`` and ``torchvision``
   are imported lazily, inside the function that needs them.
3. **No global state.** Everything that can be sampled can be reset, so the
   same process can run several benchmarks back to back.

Memory units
------------
All memory numbers are reported in **MB = MiB = 1024 * 1024 bytes** (that is
what GPU vendors and the ``torch.cuda`` API themselves use). The
``*_bytes`` helpers return raw bytes so nothing is lost.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

MB = 1024 * 1024
GB = 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile of ``values``.

    ``p`` is expressed in percent (``95`` -> the 95th percentile). The
    interpolation rule matches ``numpy.percentile(..., method="linear")``:
    with ``n`` sorted samples the result is a blend of the two neighbouring
    order statistics, so it is defined for ``n == 1`` as well.
    """
    data = sorted(float(v) for v in values)
    if not data:
        raise ValueError("percentile() of an empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p}")
    if len(data) == 1:
        return data[0]
    k = (len(data) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    frac = k - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def summarize_durations(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics for a list of durations (seconds).

    Returns a dict with the keys ``avg``, ``median``, ``p95``, ``min``,
    ``max`` and ``n``. Extra keys (``p50``, ``p99``, ``stdev``) are included
    because they are free, but the six keys above are the contract.
    """
    data = [float(v) for v in values]
    if not data:
        return {
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
            "p50": 0.0,
            "p99": 0.0,
            "stdev": 0.0,
            "total": 0.0,
        }
    return {
        "avg": round(statistics.fmean(data), 6),
        "median": round(percentile(data, 50), 6),
        "p95": round(percentile(data, 95), 6),
        "min": round(min(data), 6),
        "max": round(max(data), 6),
        "n": len(data),
        "p50": round(percentile(data, 50), 6),
        "p99": round(percentile(data, 99), 6),
        "stdev": round(statistics.pstdev(data), 6) if len(data) > 1 else 0.0,
        "total": round(sum(data), 6),
    }


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------


def _windows_peak_working_set_bytes() -> int | None:
    """Peak working set of this process via ``GetProcessMemoryInfo``.

    Returns ``None`` when the call is not available (non-Windows, or the
    ``psapi`` DLL could not be loaded).
    """
    if not sys.platform.startswith("win"):
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (OSError, AttributeError):  # pragma: no cover - platform specific
        return None

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb
    )
    if not ok:
        return None
    return int(counters.PeakWorkingSetSize)


def _resource_peak_rss_bytes() -> int | None:
    """``ru_maxrss`` fallback (KB on Linux, bytes on macOS)."""
    try:
        import resource  # noqa: PLC0415 - optional, platform dependent
    except ImportError:  # pragma: no cover - Windows
        return None
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):  # pragma: no cover
        return None
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


def current_ram_bytes() -> int:
    """Resident set size of this process right now, in bytes."""
    try:
        import psutil  # noqa: PLC0415 - optional dependency

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - psutil missing or denied
        pass
    if sys.platform.startswith("win"):
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        try:
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(kernel32.GetCurrentProcess()),
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.WorkingSetSize)
        except Exception:  # pragma: no cover
            pass
    try:
        import resource  # noqa: PLC0415

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # pragma: no cover
        return 0


def peak_ram_bytes() -> int | None:
    """Peak resident memory of this process in bytes (``None`` if unknown).

    Order of preference:
      1. Windows ``GetProcessMemoryInfo`` -> ``PeakWorkingSetSize`` (true peak).
      2. ``resource.getrusage`` -> ``ru_maxrss`` (true peak, Linux/macOS).
      3. ``psutil`` RSS (current, not a true peak) - only as a last resort and
         it is flagged as such by :func:`peak_ram_source`.
    """
    if sys.platform.startswith("win"):
        val = _windows_peak_working_set_bytes()
        if val:
            return val
    val = _resource_peak_rss_bytes()
    if val:
        return val
    try:
        import psutil  # noqa: PLC0415

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover
        return None


def peak_ram_source() -> str:
    """Which mechanism :func:`peak_ram_bytes` will use on this platform."""
    if sys.platform.startswith("win") and _windows_peak_working_set_bytes():
        return "win32:GetProcessMemoryInfo.PeakWorkingSetSize"
    if _resource_peak_rss_bytes():
        return "resource:ru_maxrss"
    return "psutil:rss(current only)"


def peak_ram_mb() -> float | None:
    """Peak resident memory in MB (``None`` when it cannot be determined)."""
    raw = peak_ram_bytes()
    return None if raw is None else round(raw / MB, 2)


def current_ram_mb() -> float:
    """Current resident memory in MB."""
    return round(current_ram_bytes() / MB, 2)


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def peak_vram_bytes() -> int | None:
    """Peak CUDA memory allocated by this process in bytes (``None`` if n/a)."""
    if not _cuda_available():
        return None
    import torch  # noqa: PLC0415

    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # pragma: no cover
        return None


def peak_vram_mb() -> float | None:
    """Peak CUDA memory in MB (``None`` when CUDA is not available)."""
    raw = peak_vram_bytes()
    return None if raw is None else round(raw / MB, 2)


def current_vram_bytes() -> int | None:
    """Currently allocated CUDA memory in bytes (``None`` if n/a)."""
    if not _cuda_available():
        return None
    import torch  # noqa: PLC0415

    try:
        return int(torch.cuda.memory_allocated())
    except Exception:  # pragma: no cover
        return None


def reset_vram_peak() -> None:
    """Reset the CUDA peak-memory counters so the next peak is scoped."""
    if not _cuda_available():
        return
    import torch  # noqa: PLC0415

    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass


def gpu_name() -> str | None:
    """Human readable name of the active CUDA device."""
    if not _cuda_available():
        return None
    import torch  # noqa: PLC0415

    try:
        return torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Background RSS sampler
# ---------------------------------------------------------------------------


class MemSampler:
    """Background thread sampling this process' RSS every ``interval`` seconds.

    Usage::

        with MemSampler(interval=0.05) as sampler:
            do_work()
        print(sampler.peak_mb)

    The sampler is safe to use as a context manager and can be stopped
    explicitly with :meth:`stop`. It never raises: if the platform refuses to
    report RSS, ``samples`` stays empty and ``peak_mb`` is ``0.0``.
    """

    def __init__(self, interval: float = 0.05, name: str = "mem-sampler") -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = float(interval)
        self.name = name
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[int] = []
        self.started_at: float | None = None
        self.stopped_at: float | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "MemSampler":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "MemSampler":
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self.stopped_at = time.time()
        return self

    def __enter__(self) -> "MemSampler":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- internals ---------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = current_ram_bytes()
            except Exception:  # pragma: no cover - defensive
                rss = 0
            with self._lock:
                self.samples.append(rss)
            self._stop_event.wait(self.interval)

    # -- results -----------------------------------------------------------
    @property
    def peak_bytes(self) -> int:
        with self._lock:
            return max(self.samples) if self.samples else 0

    @property
    def peak_mb(self) -> float:
        return round(self.peak_bytes / MB, 2)

    @property
    def mean_mb(self) -> float:
        with self._lock:
            if not self.samples:
                return 0.0
            return round(statistics.fmean(self.samples) / MB, 2)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_s": self.interval,
            "samples": self.sample_count,
            "peak_mb": self.peak_mb,
            "mean_mb": self.mean_mb,
            "source": peak_ram_source(),
        }


# ---------------------------------------------------------------------------
# Model size / environment helpers
# ---------------------------------------------------------------------------


def directory_size_bytes(path: str | os.PathLike[str]) -> int:
    """Total size of every file below ``path`` (0 when it does not exist)."""
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.is_file() and not fp.is_symlink():
                    total += fp.stat().st_size
            except OSError:  # pragma: no cover
                continue
    return total


def model_size_bytes(model_dir: str | os.PathLike[str]) -> int:
    """Bytes of the model weights on disk."""
    return directory_size_bytes(model_dir)


def model_size_gb(model_dir: str | os.PathLike[str]) -> float:
    """Model size in GB (1 GB = 1024^3 bytes)."""
    return round(directory_size_bytes(model_dir) / GB, 4)


def subdir_sizes_bytes(model_dir: str | os.PathLike[str]) -> dict[str, int]:
    """Per-component size (unet / text_encoder / vae / other) in bytes."""
    root = Path(model_dir)
    out: dict[str, int] = {}
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if child.is_dir():
            out[child.name] = directory_size_bytes(child)
    return out


def environment_info() -> dict[str, Any]:
    """Best-effort description of the machine the benchmark ran on."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": _cuda_available(),
        "gpu": gpu_name(),
        "peak_ram_source": peak_ram_source(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:  # pragma: no cover
        info["torch"] = None
    try:
        import PIL  # noqa: PLC0415

        info["pillow"] = PIL.__version__
    except Exception:  # pragma: no cover
        info["pillow"] = None
    try:
        import numpy  # noqa: PLC0415

        info["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover
        info["numpy"] = None
    try:
        import diffusers  # noqa: PLC0415

        info["diffusers"] = diffusers.__version__
    except Exception:  # pragma: no cover
        info["diffusers"] = None
    try:
        import transformers  # noqa: PLC0415

        info["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        info["transformers"] = None
    return info


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join("" if c is None else str(c) for c in row) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write ``text`` (UTF-8), creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def todo(reason: str, command: str) -> dict[str, Any]:
    """A machine-readable "not measured yet" marker.

    Benchmarks use this instead of inventing numbers. ``docs/REPORT.md`` is
    generated from the same structure, so a TODO stays visibly a TODO.
    """
    return {"status": "todo_pending_run", "reason": reason, "command": command}


def resolve_repo_root() -> Path:
    """Repository root (``cheapmage/bench/metrics.py`` -> repo root)."""
    return Path(__file__).resolve().parent.parent.parent
