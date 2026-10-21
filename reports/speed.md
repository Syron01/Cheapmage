# bench_speed - latency, memory, cost

_generated 2026-10-22T00:23:41+0300 - device `cpu`_

## Settings (DEFAULT from `config/config.yaml`)

| key | value |
| --- | --- |
| width x height | 512x512 |
| steps | 20 |
| cfg_scale | 6.0 |
| sampler | lcm |
| style_preset | auto |
| seed | -1 |
| output_format | webp |
| quality_preset | balanced |
| dtype | int8 |
| model_offload | True |

## Latency (seconds)

| metric | value |
| --- | --- |
| avg | 1.972 |
| median | 1.827 |
| p95 | 2.272 |
| min | 1.669 |
| max | 3.107 |
| n (measured requests) | 100 |

## Memory

| metric | value (MB) |
| --- | --- |
| peak VRAM | 1411.46 |
| peak RAM | 3343.28 |
| VRAM budget | 4096 |
| RAM budget | 2048 |

## Cost per image (TRY)

| metric | value |
| --- | --- |
| avg TL/image | 0.000164 |
| max TL/image | 0.000259 |

Throughput: 30.428 images/minute.
