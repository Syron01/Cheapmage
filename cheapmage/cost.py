"""Per-image cost estimation in TRY (TL).

Two cost models:

* **local** - electricity + amortised hardware depreciation, used when the model
  runs on a local GPU/CPU.
* **cloud** - GPU hourly rental price scaled by a utilisation factor, used when
  the model runs on a rented GPU (the deployment mode that has to compete with
  ~0.50 TL/image API pricing).

The estimate is a function of the *measured* wall time of the request, so the
log always reflects the real cost of the run that just happened.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class CostEstimate:
    device: str
    seconds: float
    cost_tl: float
    model: str
    gpu_hourly_tl: float
    utilization: float
    power_watts: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_cost_per_second(cfg, seconds: float) -> float:
    # 15W average draw for a CPU-only inference node, electricity only.
    kwh = (cfg.gpu_power_watts * 0.15 / 1000.0) * (seconds / 3600.0)
    return kwh * cfg.electricity_kwh_price


def estimate_cost(cfg, device: str, seconds: float, model_name: str = "cheapmage-v1") -> CostEstimate:
    """Estimate the TRY cost of one generation.

    ``cfg`` is a :class:`cheapmage.config.CostConfig`.
    """
    seconds = max(float(seconds), 0.0)
    is_cloud = os.environ.get("CHEAPMAGE_DEPLOYMENT", "local").lower() == "cloud"

    if device == "cpu":
        cost = _cpu_cost_per_second(cfg, seconds)
        hourly = 0.0
        util = 0.0
        detail = "local cpu, electricity only"
    elif is_cloud:
        hourly = cfg.gpu_hourly_tl
        util = max(cfg.cloud_utilization, 0.01)
        cost = hourly * (seconds / 3600.0) * util
        detail = f"cloud gpu @ {hourly:.2f} TL/h x utilisation {util:.2f}"
    else:
        hourly = cfg.local_gpu_hourly_tl
        util = 1.0
        kwh = (cfg.gpu_power_watts / 1000.0) * (seconds / 3600.0)
        electricity = kwh * cfg.electricity_kwh_price
        amortised = hourly * (seconds / 3600.0)
        cost = electricity + amortised
        detail = (
            f"local gpu, {cfg.gpu_power_watts}W -> {kwh * 1000:.3f} kWh "
            f"x {cfg.electricity_kwh_price:.2f} TL/kWh"
        )

    return CostEstimate(
        device=device,
        seconds=round(seconds, 3),
        cost_tl=round(cost, 6),
        model=model_name,
        gpu_hourly_tl=hourly,
        utilization=util,
        power_watts=cfg.gpu_power_watts,
        detail=detail,
    )


def cloud_unit_price(cfg, device: str = "cuda") -> float:
    """Cost of one image if served from a cloud GPU at the given wall time."""
    return estimate_cost(cfg, device, 3.5).cost_tl
