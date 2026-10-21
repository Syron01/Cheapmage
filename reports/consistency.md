# bench_consistency - character consistency over poses

_generated 2026-10-22T00:24:20+0300_

| key | value |
| --- | --- |
| character | a cheerful young woman with short silver hair and blue eyes |
| poses | 10 |
| seed | -1 |
| style_preset | auto |
| backend | descriptor-fallback |
| backend detail | HSV colour histogram (32x8, value-weighted) + edge orientation histogram (9 bins, magnitude-weighted) + 16x16 luminance structure |
| weights path | n/a (fallback) |

## Pairwise similarity (cosine, -1..1)

| metric | value | threshold |
| --- | --- | --- |
| mean | 0.357632 | 0.75 |
| min | 0.201961 | - |
| max | 0.539765 | - |
| p05 | 0.255599 | - |
| median | 0.363944 | - |
| stdev | 0.067383 | - |
| to centroid (mean) | 0.649514 | - |

**Result: FAIL** (mean >= 0.75)

## Per-pair scores

| pose A | pose B | similarity |
| --- | --- | --- |
| front | three_quarter | 0.433921 |
| front | side | 0.30485 |
| front | back | 0.396308 |
| front | walking | 0.395954 |
| front | sitting | 0.286273 |
| front | waving | 0.374216 |
| front | arms_up | 0.374959 |
| front | close_up | 0.380837 |
| front | action | 0.429031 |
| three_quarter | side | 0.367353 |
| three_quarter | back | 0.428344 |
| three_quarter | walking | 0.36989 |
| three_quarter | sitting | 0.282034 |
| three_quarter | waving | 0.445041 |
| three_quarter | arms_up | 0.354497 |
| three_quarter | close_up | 0.412435 |
| three_quarter | action | 0.539765 |
| side | back | 0.352595 |
| side | walking | 0.328359 |
| side | sitting | 0.342413 |
| side | waving | 0.363944 |
| side | arms_up | 0.32806 |
| side | close_up | 0.368916 |
| side | action | 0.42185 |
| back | walking | 0.33181 |
| back | sitting | 0.426648 |
| back | waving | 0.381209 |
| back | arms_up | 0.37542 |
| back | close_up | 0.407197 |
| back | action | 0.448831 |
| walking | sitting | 0.204037 |
| walking | waving | 0.25442 |
| walking | arms_up | 0.347207 |
| walking | close_up | 0.341611 |
| walking | action | 0.439455 |
| sitting | waving | 0.268049 |
| sitting | arms_up | 0.201961 |
| sitting | close_up | 0.273636 |
| sitting | action | 0.260317 |
| waving | arms_up | 0.347555 |
| waving | close_up | 0.354345 |
| waving | action | 0.324177 |
| arms_up | close_up | 0.296501 |
| arms_up | action | 0.299661 |
| close_up | action | 0.427558 |
