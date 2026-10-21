# bench_cost - cost per image (TRY)

_generated 2026-10-22T00:24:29+0300_

Wall time source: **measured (reports\speed.json, avg of 100 requests)**

## Cost table

| device | wall time / image | GPU hourly | utilisation | TL per image | vs market (0.50 TL/image) |
| --- | --- | --- | --- | --- | --- |
| cuda | 1.972 s | 0.00 TL/h | 1.00 | 0.000164 TL | 3048.8x cheaper than 0.50 TL |
| cpu | 1.972 s | 0.00 TL/h | 0.00 | 0.000025 TL | 20000.0x cheaper than 0.50 TL |

## Break-even

| quantity | value |
| --- | --- |
| target price per image | 0.010 TL |
| break-even wall time (cloud GPU) | 22.857 s |
| break-even wall time (local electricity) | 120.0 s |
| images/hour that keeps us at the target | 157.5 |
| cloud formula | cost = gpu_hourly_tl * (seconds / 3600) * utilisation |
| local formula | cost = (watts / 1000) * (seconds / 3600) * tl_per_kwh |
