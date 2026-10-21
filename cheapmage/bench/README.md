# CheapMage bench harness

Measurement tooling for CheapMage. Every number that appears in
`docs/REPORT.md` is produced by one of these commands - nothing is written
by hand.

## Principles

1. **No invented measurements.** If a benchmark cannot run (missing engine,
   missing weights, missing dependency) it writes a machine-readable
   `TODO - pending run` marker containing the reason *and* the exact command
   that produces the number. It never guesses.
2. **No downloads at import time** and, for the similarity backend, no
   downloads at all: ResNet-18 weights are used only if they are already in
   the local `torch.hub` cache, otherwise a dependency-free descriptor is used
   and the output records which backend ran.
3. **Safety first.** The moderation benchmark never generates an image for a
   prompt in the zero-tolerance category, and it fails the run (non-zero exit
   code) when any such prompt is allowed.
4. **Reproducible.** Fixed corpus, fixed prompt set, fixed seeds, and the
   exact command line is embedded in every report.

## Layout

| File | Purpose |
|---|---|
| `metrics.py` | Reusable helpers: `percentile`, `summarize_durations`, `peak_vram_mb`, `peak_ram_mb`, `MemSampler` (RSS every 50 ms), `model_size_gb`, markdown/JSON writers, `todo()` marker helper. |
| `engine_adapter.py` | Lazy, fault-tolerant bridge to `cheapmage.engine.get_engine()`; tries a few documented call signatures and records which one worked. |
| `bench_speed.py` | Latency / memory / cost benchmark with DEFAULT settings. |
| `bench_consistency.py` | Same character, 10 poses, pairwise similarity. |
| `bench_moderation.py` | Fixed safety corpus, NSFW off/on, zero tolerance. |
| `bench_cost.py` | Cost table, market comparison, break-even wall time. |
| `bench_human_eval.py` | Renders the fixed 30-image human-evaluation set + score sheet. |
| `human_eval.md` | Human evaluation protocol and rubric (1-5 scales). |
| `human_eval_prompts.json` | The fixed 30 evaluation prompts. |

## Commands

```bash
# 1. Latency, memory, cost (>= 100 requests, default settings)
python -m cheapmage.bench.bench_speed --requests 100 --warmup 5 --out reports/speed.json

# 2. Character consistency over 10 poses
python -m cheapmage.bench.bench_consistency \
    --character "a cheerful young woman with short silver hair and blue eyes" \
    --poses 10 --out reports/consistency.json

# 3. Safety corpus, NSFW off and on, zero tolerance
python -m cheapmage.bench.bench_moderation --out reports/moderation.json

# 4. Cost table (reads reports/speed.json when it exists)
python -m cheapmage.bench.bench_cost --out reports/cost.json

# 5. Human evaluation set (30 images) + score sheet
python -m cheapmage.bench.bench_human_eval --count 30 --out reports/human_eval.json

# All of the above, in order
python tools/run_benchmarks.py --requests 100
```

## Outputs

| Report | JSON | Markdown |
|---|---|---|
| speed | `reports/speed.json` | `reports/speed.md` |
| consistency | `reports/consistency.json` | `reports/consistency.md` |
| moderation | `reports/moderation.json` | `reports/moderation.md` |
| cost | `reports/cost.json` | `reports/cost.md` |
| human eval | `reports/human_eval.json` | `reports/human_eval_sheet.csv` |

Every JSON report carries the same envelope keys: `status`, `command`,
`generated_at`, `environment`. `status` is one of `measured`,
`measured_with_reference_stub`, `images_rendered_awaiting_human_scoring` or
`todo_pending_run`.

## Acceptance thresholds

| Metric | Threshold | Source |
|---|---|---|
| latency avg | <= 3.5 s | `docs/EVALUATION.md` |
| latency p95 | <= 8.0 s (API timeout is 20 s) | `docs/EVALUATION.md` |
| model size | <= 1.5 GB | `docs/EVALUATION.md` |
| peak VRAM | <= 4096 MB (config budget) | `config/config.yaml` |
| peak RAM | <= 2048 MB (config budget) | `config/config.yaml` |
| cost per image | <= 0.01 TL | `cost.target_per_image_try` |
| consistency (mean pairwise, 10 poses) | >= 0.75 | `bench_consistency --threshold` |
| anime quality (human, 3 raters) | >= 3.8 / 5 | `human_eval.md` |
| 3D quality (human, 3 raters) | >= 3.6 / 5 | `human_eval.md` |
| AI slop (human, 3 raters) | <= 1.8 / 5 | `human_eval.md` |
| moderation, minor-related sexual | **0 allowed, ever** | zero tolerance |
| moderation, explicit adult, NSFW off | 100 % blocked | policy |
| moderation, real person / copyrighted | 100 % blocked | policy |
| moderation, non-English | 100 % blocked | English-only policy |

## Notes

- `bench_speed` calls the engine **directly** (not through the API) so that
  the measurement is not polluted by HTTP overhead; the API timeout of 20 s
  is still enforced per request.
- The RSS sampler runs in a daemon thread and samples every 50 ms; the peak
  it reports is the peak *working set*, which on Windows comes from
  `GetProcessMemoryInfo.PeakWorkingSetSize` (a true peak, not a sample).
- The consistency backend is recorded in `similarity_backend.backend`. The
  fallback descriptor measures palette + structure, so its absolute values
  are **not** comparable to the ResNet backend - only trends within the same
  backend are.
- The moderation benchmark falls back to a clearly-labelled reference policy
  stub when no production moderator exists; such runs are reported as
  `measured_with_reference_stub` and must not be cited as a production
  safety measurement.
