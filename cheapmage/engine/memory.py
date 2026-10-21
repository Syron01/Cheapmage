"""Memory accounting helpers (no third-party deps, Windows + POSIX)."""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import sys
import threading
import time

_MB = 1024.0 * 1024.0


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _win_process_memory() -> tuple[float, float]:
    """(working_set_mb, peak_working_set_mb) on Windows."""
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _PMC()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PMC),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.WorkingSetSize / _MB, counters.PeakWorkingSetSize / _MB


def process_rss_mb() -> float:
    """Resident set size of this process in MB."""
    if sys.platform == "win32":
        try:
            return _win_process_memory()[0]
        except Exception:
            return 0.0
    try:
        with open(f"/proc/{os.getpid()}/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / _MB
    except Exception:
        return 0.0


def vram_mb() -> dict[str, float]:
    """GPU memory usage in MB (allocated / reserved / peak)."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / _MB,
                "reserved": torch.cuda.memory_reserved() / _MB,
                "peak_allocated": torch.cuda.max_memory_allocated() / _MB,
                "peak_reserved": torch.cuda.max_memory_reserved() / _MB,
            }
    except Exception:
        pass
    return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}


def ram_mb() -> dict[str, float]:
    return {
        "process_rss": process_rss_mb(),
        "process_peak_rss": peak_rss_mb(),
    }


def peak_rss_mb() -> float:
    if sys.platform == "win32":
        try:
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return counters.PeakWorkingSetSize / _MB
        except Exception:
            return 0.0
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


class MemorySampler:
    """Background sampler of RSS (and VRAM when available)."""

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak_rss = 0.0
        self.peak_vram = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        import torch

        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, process_rss_mb())
            if torch.cuda.is_available():
                self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated() / _MB)
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._stop.clear()
        self.peak_rss = 0.0
        self.peak_vram = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def directory_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / _MB


def now() -> float:
    return time.time()
