"""Latency / memory / cost benchmark.

Usage::

    python -m cheapmage.bench.bench_speed --requests 100 --warmup 5 --out reports/speed.json

What it does
------------
* Loads the **default** configuration (``config/config.yaml``) - the same
  settings a normal user gets, no CLI overrides of the generation knobs.
* Calls the engine directly (``cheapmage.engine.get_engine().generate``).
* For every request it records: wall time, engine-reported duration, peak
  VRAM, peak RAM (background sampler), and the per-image cost estimate from
  ``cheapmage.cost.estimate_cost``.
* Writes ``reports/speed.json`` (machine readable) and ``reports/speed.md``
  (the markdown tables for ``docs/REPORT.md``).

If the engine or its weights are missing the benchmark does **not** invent
numbers: it writes a report whose payload is a ``TODO - pending run`` marker
plus the exact command to run, prints the reason, and exits with code 3.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from . import engine_adapter
from .metrics import (
    MemSampler,
    environment_info,
    md_table,
    peak_vram_mb,
    reset_vram_peak,
    summarize_durations,
    todo,
    write_json,
    write_text,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_READY = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cheapmage.bench.bench_speed",
        description="Measure generation latency, memory and cost with DEFAULT settings.",
    )
    p.add_argument("--requests", type=int, default=100,
                   help="number of measured requests (spec: >= 100)")
    p.add_argument("--warmup", type=int, default=5,
                   help="warm-up requests executed before measuring (not counted)")
    p.add_argument("--out", type=Path, default=Path("reports/speed.json"),
                   help="JSON output path")
    p.add_argument("--md-out", type=Path, default=None,
                   help="markdown output path (default: <out>.md sibling 'speed.md')")
    p.add_argument("--config", type=Path, default=None,
                   help="config file (default: config/config.yaml)")
    p.add_argument("--seed", type=int, default=None,
                   help="fixed seed for every request (default: config value, -1 = random)")
    p.add_argument("--sample-interval", type=float, default=0.05,
                   help="RSS sampling interval in seconds (default: 50 ms)")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-request timeout in seconds (default: api.timeout_seconds)")
    p.add_argument("--keep-images", type=Path, default=None,
                   help="optional directory to store the generated images")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .config_access import get_bench_config

    cfg = get_bench_config(args.config)
    gen = cfg.generation
    command = (
        f"python -m cheapmage.bench.bench_speed --requests {args.requests} "
        f"--warmup {args.warmup} --out {args.out}"
    )

    md_out = args.md_out or (args.out.with_suffix(".md") if args.out.suffix == ".json"
                             else Path(str(args.out) + ".md"))
    if md_out.name == args.out.name:
        md_out = Path("reports") / "speed.md"

    try:
        handle = engine_adapter.load_engine()
    except engine_adapter.EngineUnavailable as exc:
        reason = str(exc)
        print("=" * 72)
        print("bench_speed: NOT RUN - " + reason)
        print("No measurements were produced. Fill in the numbers once the")
        print("engine is available by running:")
        print("  " + command)
        print("=" * 72)
        payload = {
            **todo(reason, command),
            "benchmark": "speed",
            "environment": environment_info(),
            "config_source": str(cfg.source_path),
        }
        write_json(args.out, payload)
        write_text(md_out, _markdown_pending("bench_speed", reason, command))
        return EXIT_NOT_READY

    from ..cost import estimate_cost

    timeout = args.timeout if args.timeout is not None else float(cfg.api.timeout_seconds)
    durations: list[float] = []
    engine_durations: list[float] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    vram_peak_mb = 0.0
    sampler = MemSampler(interval=args.sample_interval)

    print(f"bench_speed: {args.requests} measured + {args.warmup} warm-up requests")
    print(f"  device setting : {cfg.hardware.device}  (model_dir: {handle.model_dir})")
    print(f"  generation     : {gen.width}x{gen.height} steps={gen.steps} "
          f"cfg={gen.cfg_scale} sampler={gen.sampler} style={gen.style_preset}")

    reset_vram_peak()
    sampler.start()
    started_all = time.perf_counter()
    try:
        for index in range(args.warmup + args.requests):
            warmup = index < args.warmup
            reset_vram_peak()
            seed = args.seed if args.seed is not None else gen.seed
            try:
                result, elapsed = engine_adapter.generate(
                    handle,
                    prompt=gen.prompt,
                    negative_prompt=gen.negative_prompt,
                    style_preset=gen.style_preset,
                    width=gen.width,
                    height=gen.height,
                    steps=gen.steps,
                    cfg_scale=gen.cfg_scale,
                    sampler=gen.sampler,
                    seed=seed,
                    output_format=gen.output_format,
                    quality_preset=gen.quality_preset,
                    reference_image=cfg.consistency.reference_image,
                    reference_strength=cfg.consistency.reference_strength,
                    character_embedding=cfg.consistency.character_embedding,
                    seed_lock=cfg.consistency.seed_lock,
                    timeout_seconds=timeout,
                )
            except Exception as exc:
                failures.append({
                    "index": index,
                    "warmup": warmup,
                    "error": f"{exc.__class__.__name__}: {exc}",
                })
                print(f"  [{index + 1}/{args.warmup + args.requests}] FAILED: {exc}")
                continue

            fields = engine_adapter.result_fields(result)
            engine_duration = fields.get("duration_seconds")
            duration = float(engine_duration) if engine_duration is not None else elapsed
            vram_now = peak_vram_mb() or 0.0
            vram_peak_mb = max(vram_peak_mb, vram_now)

            if not warmup:
                durations.append(duration)
                if engine_duration is not None:
                    engine_durations.append(float(engine_duration))
                cost = estimate_cost(cfg.cost, fields.get("device") or handle.device or "cpu",
                                     duration, model_name=Path(handle.model_dir or "").name)
                rows.append({
                    "index": index - args.warmup,
                    "duration_seconds": round(duration, 4),
                    "engine_duration_seconds": (
                        round(float(engine_duration), 4) if engine_duration is not None else None
                    ),
                    "vram_mb": vram_now,
                    "ram_mb": sampler.peak_mb,
                    "cost_tl": cost.cost_tl,
                    "seed": fields.get("seed"),
                    "style_used": fields.get("style_used"),
                    "steps": fields.get("steps"),
                })
            if args.keep_images is not None and not warmup:
                _save_image(args.keep_images, index - args.warmup, result)
            if not warmup and (index - args.warmup) % 10 == 0:
                print(f"  [{index + 1}/{args.warmup + args.requests}] "
                      f"{duration:.3f}s vram={vram_now:.0f}MB")
    finally:
        sampler.stop()
    total_wall = time.perf_counter() - started_all

    summary = summarize_durations(durations)
    costs = [r["cost_tl"] for r in rows]
    cost_summary = summarize_durations(costs) if costs else summarize_durations([])
    ram_peak = max([sampler.peak_mb] + [r["ram_mb"] for r in rows]) if rows else sampler.peak_mb
    vram_overall = max(vram_peak_mb, peak_vram_mb() or 0.0)

    payload: dict[str, Any] = {
        "benchmark": "speed",
        "status": "measured",
        "command": command,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requests_measured": summary["n"],
        "warmup": args.warmup,
        "settings": {
            "config_source": str(cfg.source_path),
            "prompt": gen.prompt,
            "negative_prompt": gen.negative_prompt,
            "style_preset": gen.style_preset,
            "width": gen.width,
            "height": gen.height,
            "steps": gen.steps,
            "cfg_scale": gen.cfg_scale,
            "sampler": gen.sampler,
            "seed": gen.seed,
            "output_format": gen.output_format,
            "quality_preset": gen.quality_preset,
            "device_setting": cfg.hardware.device,
            "dtype": cfg.hardware.dtype,
            "model_offload": cfg.hardware.model_offload,
        },
        "model": {
            "model_dir": handle.model_dir,
            "size_bytes": handle.model_bytes,
            "size_gb": round(handle.model_bytes / (1024 ** 3), 4),
        },
        "latency_seconds": summary,
        "engine_reported_latency_seconds": summarize_durations(engine_durations),
        "cost_tl_per_image": cost_summary,
        "memory": {
            "peak_vram_mb": round(vram_overall, 2),
            "peak_ram_mb": round(ram_peak, 2),
            "vram_budget_mb": cfg.hardware.max_vram_mb,
            "ram_budget_mb": cfg.hardware.max_ram_mb,
            "sampler": sampler.to_dict(),
        },
        "throughput_images_per_minute": (
            round(60.0 / summary["avg"], 3) if summary["avg"] > 0 else None
        ),
        "total_wall_seconds": round(total_wall, 3),
        "device": handle.device,
        "environment": environment_info(),
        "failures": failures,
        "rows": rows,
    }
    write_json(args.out, payload)
    write_text(md_out, _markdown(payload))

    print("-" * 72)
    print(f"avg {summary['avg']:.3f}s  median {summary['median']:.3f}s  "
          f"p95 {summary['p95']:.3f}s  min {summary['min']:.3f}s  max {summary['max']:.3f}s "
          f"(n={summary['n']})")
    print(f"peak VRAM {vram_overall:.0f} MB   peak RAM {ram_peak:.0f} MB")
    print(f"wrote {args.out} and {md_out}")
    return EXIT_OK if not failures else EXIT_FAILED


def _save_image(directory: Path, index: int, result: Any) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        data = result["image_bytes"] if isinstance(result, dict) else result.image_bytes
        fmt = (result.get("image_format") if isinstance(result, dict)
               else getattr(result, "image_format", "png"))
        (directory / f"{index:04d}.{fmt or 'png'}").write_bytes(data)
    except Exception:  # pragma: no cover - image dumping is best effort
        pass


def _markdown(payload: dict[str, Any]) -> str:
    if payload.get("status") != "measured":
        return _markdown_pending("bench_speed", payload.get("reason", ""),
                                 payload.get("command", ""))

    lat = payload["latency_seconds"]
    mem = payload["memory"]
    cost = payload["cost_tl_per_image"]
    settings = payload["settings"]
    rows = [
        ["avg", f"{lat['avg']:.3f}"],
        ["median", f"{lat['median']:.3f}"],
        ["p95", f"{lat['p95']:.3f}"],
        ["min", f"{lat['min']:.3f}"],
        ["max", f"{lat['max']:.3f}"],
        ["n (measured requests)", lat["n"]],
    ]
    out = [
        "# bench_speed - latency, memory, cost",
        "",
        f"_generated {payload['generated_at']} - device `{payload.get('device')}`_",
        "",
        "## Settings (DEFAULT from `config/config.yaml`)",
        "",
        md_table(
            ["key", "value"],
            [
                ["width x height", f"{settings['width']}x{settings['height']}"],
                ["steps", settings["steps"]],
                ["cfg_scale", settings["cfg_scale"]],
                ["sampler", settings["sampler"]],
                ["style_preset", settings["style_preset"]],
                ["seed", settings["seed"]],
                ["output_format", settings["output_format"]],
                ["quality_preset", settings["quality_preset"]],
                ["dtype", settings["dtype"]],
                ["model_offload", settings["model_offload"]],
            ],
        ),
        "",
        "## Latency (seconds)",
        "",
        md_table(["metric", "value"], rows),
        "",
        "## Memory",
        "",
        md_table(
            ["metric", "value (MB)"],
            [
                ["peak VRAM", mem["peak_vram_mb"]],
                ["peak RAM", mem["peak_ram_mb"]],
                ["VRAM budget", mem["vram_budget_mb"]],
                ["RAM budget", mem["ram_budget_mb"]],
            ],
        ),
        "",
        "## Cost per image (TRY)",
        "",
        md_table(
            ["metric", "value"],
            [
                ["avg TL/image", f"{cost['avg']:.6f}"],
                ["max TL/image", f"{cost['max']:.6f}"],
            ],
        ),
        "",
        f"Throughput: {payload.get('throughput_images_per_minute')} images/minute.",
        "",
    ]
    if payload.get("failures"):
        out += ["## Failures", ""]
        out.append(md_table(
            ["index", "error"],
            [[f["index"], f["error"]] for f in payload["failures"]],
        ))
        out.append("")
    return "\n".join(out)


def _markdown_pending(name: str, reason: str, command: str) -> str:
    return (
        f"# {name} - NOT RUN\n\n"
        f"**TODO - pending run**\n\n"
        f"Reason: {reason}\n\n"
        f"Run this to produce the numbers:\n\n"
        f"```\n{command}\n```\n"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
