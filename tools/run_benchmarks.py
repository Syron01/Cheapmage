"""Run the whole CheapMage bench suite and merge the results.

Usage::

    python tools/run_benchmarks.py --requests 100
    python tools/run_benchmarks.py --requests 3 --warmup 1   # quick smoke run

Every benchmark is executed as a subprocess so one failing (or crashing)
benchmark cannot take the others down. The exit code is non-zero when any
benchmark failed or could not run, and a merged summary is written to
``reports/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, argv: list[str]) -> dict:
    print("=" * 78)
    print(f"[{name}] {' '.join(argv)}")
    print("=" * 78)
    started = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        code = proc.returncode
    except Exception as exc:  # pragma: no cover - defensive
        code = -1
        print(f"failed to start: {exc}")
    elapsed = time.perf_counter() - started
    print(f"[{name}] exit={code} in {elapsed:.1f}s")
    return {"step": name, "command": " ".join(argv), "exit_code": code,
            "seconds": round(elapsed, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python tools/run_benchmarks.py")
    ap.add_argument("--requests", type=int, default=100,
                    help="measured requests for bench_speed (spec: >= 100)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--character", type=str,
                    default="a cheerful young woman with short silver hair and blue eyes")
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--skip-human-eval", action="store_true")
    ap.add_argument("--skip", type=str, default="", help="comma separated step names")
    args = ap.parse_args()

    py = sys.executable
    reports = args.reports
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    steps: list[tuple[str, list[str]]] = [
        ("speed", [py, "-m", "cheapmage.bench.bench_speed",
                   "--requests", str(args.requests), "--warmup", str(args.warmup),
                   "--out", str(reports / "speed.json")]),
        ("consistency", [py, "-m", "cheapmage.bench.bench_consistency",
                         "--character", args.character,
                         "--poses", str(args.poses),
                         "--out", str(reports / "consistency.json")]),
        ("moderation", [py, "-m", "cheapmage.bench.bench_moderation",
                        "--out", str(reports / "moderation.json")]),
        ("cost", [py, "-m", "cheapmage.bench.bench_cost",
                  "--out", str(reports / "cost.json")]),
    ]
    if not args.skip_human_eval:
        steps.append(("human_eval", [py, "-m", "cheapmage.bench.bench_human_eval",
                                     "--count", "30",
                                     "--out", str(reports / "human_eval.json")]))

    results = []
    for name, argv in steps:
        if name in skip:
            print(f"[{name}] skipped")
            continue
        results.append(run_step(name, argv))

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": results,
        "reports": {name: _status(reports / f"{name}.json")
                    for name in ("speed", "consistency", "moderation", "cost", "human_eval")},
    }
    out = reports / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print("=" * 78)
    for name, status in summary["reports"].items():
        print(f"{name:12s} {status}")
    print(f"wrote {out}")

    failed = [r for r in results if r["exit_code"] != 0]
    return 1 if failed else 0


def _status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    return str(data.get("status", "unknown"))


if __name__ == "__main__":
    sys.exit(main())
