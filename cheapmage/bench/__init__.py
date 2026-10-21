"""CheapMage measurement harness.

This package contains the *bench* tooling: it never generates images for
content that the safety layer is supposed to block, it never downloads
anything at import time, and it never fabricates a measurement.

Sub-modules (each one is a runnable CLI):

* :mod:`cheapmage.bench.metrics` - reusable helpers (percentiles, duration
  summaries, VRAM/RAM peaks, a background RSS sampler).
* :mod:`cheapmage.bench.bench_speed` - latency / memory / cost benchmark.
* :mod:`cheapmage.bench.bench_consistency` - character consistency over poses.
* :mod:`cheapmage.bench.bench_moderation` - safety corpus + zero tolerance.
* :mod:`cheapmage.bench.bench_cost` - cost table and break-even analysis.

Importing this package must stay side-effect free: no model is loaded, no
network access happens, no report is written.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
