"""Cost table and break-even analysis.

Usage::

    python -m cheapmage.bench.bench_cost                # uses reports/speed.json
    python -m cheapmage.bench.bench_cost --seconds 2.4  # explicit wall time

The table answers three questions:

1. What does one image cost in TRY on the current device?
2. How does that compare with the market price of **0.50 TL/image** that
   CheapMage has to beat?
3. At which wall time per image do we cross the internal target of
   **0.01 TL/image** (the break-even point)?

The wall time column is only filled from a *measured* run
(``reports/speed.json`` produced by ``bench_speed``) or from an explicit
``--seconds`` argument. When neither is available the cell says
``TODO - pending run`` instead of a guessed number - the break-even column
and the market comparison are pure arithmetic on the config and therefore
always available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .metrics import md_table, write_json, write_text

EXIT_OK = 0
EXIT_FAILED = 1

MARKET_PRICE_TL = 0.50  # market price per image we must beat (TL)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cheapmage.bench.bench_cost",
        description="Print the cost table (TL per image) and break-even analysis.",
    )
    p.add_argument("--seconds", type=float, default=None,
                   help="wall time per image in seconds (overrides the speed report)")
    p.add_argument("--speed-report", type=Path, default=Path("reports/speed.json"),
                   help="speed benchmark output used for the measured wall time")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("reports/cost.json"))
    p.add_argument("--md-out", type=Path, default=None)
    return p


def _load_speed_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("status") != "measured":
        return None
    lat = data.get("latency_seconds") or {}
    if not lat.get("avg"):
        return None
    return data


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config_access import get_bench_config
    from ..cost import estimate_cost

    cfg = get_bench_config(args.config)
    cost_cfg = cfg.cost

    speed = _load_speed_report(args.speed_report)
    measured_seconds: float | None = None
    source = "TODO - pending run"
    if args.seconds is not None:
        measured_seconds = float(args.seconds)
        source = "cli --seconds"
    elif speed is not None:
        measured_seconds = float(speed["latency_seconds"]["avg"])
        source = f"measured ({args.speed_report}, avg of {speed['latency_seconds']['n']} requests)"

    devices = ["cuda", "cpu"]
    rows: list[dict[str, Any]] = []
    for device in devices:
        if device == "cuda" and cfg.hardware.device == "cpu":
            continue
        if measured_seconds is None:
            rows.append({
                "device": device,
                "seconds": None,
                "hourly_tl": cost_cfg.gpu_hourly_tl if device == "cuda" else 0.0,
                "utilisation": cost_cfg.cloud_utilization if device == "cuda" else 0.0,
                "tl_per_image": None,
                "vs_market": None,
            })
            continue
        est = estimate_cost(cost_cfg, device, measured_seconds)
        rows.append({
            "device": device,
            "seconds": round(measured_seconds, 4),
            "hourly_tl": est.gpu_hourly_tl,
            "utilisation": est.utilization,
            "tl_per_image": est.cost_tl,
            "detail": est.detail,
            "vs_market": round(MARKET_PRICE_TL / est.cost_tl, 1) if est.cost_tl > 0 else None,
        })

    # ---- break-even ------------------------------------------------------
    # cloud: cost = hourly * (s / 3600) * utilisation  ==  target
    hourly = cost_cfg.gpu_hourly_tl
    util = max(cost_cfg.cloud_utilization, 1e-9)
    target = cost_cfg.target_per_image_try
    break_even_cloud = target * 3600.0 / (hourly * util) if hourly > 0 else None

    # local electricity only: cost = W/1000 * (s/3600) * price_per_kWh == target
    break_even_local = None
    if cost_cfg.gpu_power_watts > 0 and cost_cfg.electricity_kwh_price > 0:
        break_even_local = target * 3600.0 / (
            (cost_cfg.gpu_power_watts / 1000.0) * cost_cfg.electricity_kwh_price
        )

    # how many images per hour would keep us at the target price
    images_per_hour_at_target = (
        (3600.0 / break_even_cloud) if break_even_cloud else None
    )

    payload = {
        "benchmark": "cost",
        "status": "measured" if measured_seconds is not None else "todo_pending_run",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_time_source": source,
        "wall_time_seconds": measured_seconds,
        "target_per_image_try": target,
        "market_price_tl_per_image": MARKET_PRICE_TL,
        "rows": rows,
        "break_even": {
            "cloud_seconds_per_image_at_target": (
                round(break_even_cloud, 3) if break_even_cloud else None
            ),
            "local_electricity_seconds_per_image_at_target": (
                round(break_even_local, 3) if break_even_local else None
            ),
            "images_per_hour_at_target": (
                round(images_per_hour_at_target, 2) if images_per_hour_at_target else None
            ),
            "formula_cloud": "cost = gpu_hourly_tl * (seconds / 3600) * utilisation",
            "formula_local": "cost = (watts / 1000) * (seconds / 3600) * tl_per_kwh",
        },
        "config": {
            "gpu_hourly_tl": cost_cfg.gpu_hourly_tl,
            "local_gpu_hourly_tl": cost_cfg.local_gpu_hourly_tl,
            "electricity_kwh_price": cost_cfg.electricity_kwh_price,
            "gpu_power_watts": cost_cfg.gpu_power_watts,
            "cloud_utilization": cost_cfg.cloud_utilization,
        },
        "speed_report": str(args.speed_report) if speed else None,
    }
    write_json(args.out, payload)
    write_text(args.md_out or Path("reports") / "cost.md", _markdown(payload))
    print(_markdown(payload))
    print(f"wrote {args.out}")
    return EXIT_OK if measured_seconds is not None else EXIT_FAILED


def _markdown(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload["rows"]:
        if row["seconds"] is None:
            rows.append([
                row["device"], "TODO - pending run", row["hourly_tl"], row["utilisation"],
                "TODO - pending run", "TODO - pending run",
            ])
        else:
            rows.append([
                row["device"], f"{row['seconds']:.3f} s", f"{row['hourly_tl']:.2f} TL/h",
                f"{row['utilisation']:.2f}", f"{row['tl_per_image']:.6f} TL",
                f"{row['vs_market']}x cheaper than {MARKET_PRICE_TL:.2f} TL"
                if row.get("vs_market") else "n/a",
            ])
    be = payload["break_even"]
    lines = [
        "# bench_cost - cost per image (TRY)",
        "",
        f"_generated {payload['generated_at']}_",
        "",
        f"Wall time source: **{payload['wall_time_source']}**",
        "",
        "## Cost table",
        "",
        md_table(
            ["device", "wall time / image", "GPU hourly", "utilisation",
             "TL per image", "vs market (0.50 TL/image)"],
            rows,
        ),
        "",
        "## Break-even",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["target price per image", f"{payload['target_per_image_try']:.3f} TL"],
                ["break-even wall time (cloud GPU)",
                 f"{be['cloud_seconds_per_image_at_target']} s"
                 if be["cloud_seconds_per_image_at_target"] else "n/a"],
                ["break-even wall time (local electricity)",
                 f"{be['local_electricity_seconds_per_image_at_target']} s"
                 if be["local_electricity_seconds_per_image_at_target"] else "n/a"],
                ["images/hour that keeps us at the target",
                 be["images_per_hour_at_target"]],
                ["cloud formula", be["formula_cloud"]],
                ["local formula", be["formula_local"]],
            ],
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
