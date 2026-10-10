# Decode screening, not repeated acceptance

All runs are sequential fresh CLI processes on the P9 varied 8199-token
documentation fixture, producing 128 tokens. Operator settings retain 262144
context, INT8 KV, dedicated 8192 workspace, original BF16 projections and MTP2.
Startup weight handles close in every run. GPU prefill timing events are not
enabled here; the handle-release pairs in the parent report do enable them.
Do not attribute their timing difference to these decode changes.

Each directory contains exact command/env, executable/input SHA-256, raw
stdout/stderr, GPU samples and process CPU samples. Every completed screen has
the same 128 output tokens as the GPU reference, and zero post-residency expert
and MTP source reads. CPU/GPU work assignment can change floating-point
rounding; equal tokens alone are not a committed-state or full-quality proof.

The `screen-*` runs use efficiency3; `combo*` use efficiency4 (eight-entry GPU
reuse tile); subsequent `tile*-workers*` use efficiency5. These binaries differ;
use the manifests, not their names, as the exact provenance.

| Single-run setting | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| GPU, kernel copying | 2846.08 | 45.47 |
| GPU, batched DMA | 2728.61 | 44.00 |
| CPU50, original row loop | 2778.30 | 61.86 |
| CPU50, unrolled row loop | 2720.89 | 64.10 |
| CPU25, unrolled | 2680.60 | 53.87 |
| CPU75, unrolled | 2813.16 | 71.46 |
| GPU, exact workspace price | 2803.66 | 45.66 |
| GPU, elastic KV + exact price | 2844.30 | 51.55 |
| GPU, MMVF rows experiment | 2180.43 | 42.81 |
| CPU75 + elastic KV + exact price | 2776.94 | 77.74 |
| Above + GPU reuse tile8 | 2766.14 | 79.01 |
| Above + 40 CPU tasks per phase | 2772.51 | 79.64 |
| CPU50, same combined settings | 2424.18 | 73.20 |
| CPU100, same combined settings | 2823.53 | 79.78 |
| CPU75 + kernel copying | 2825.42 | 81.55 |
| Above, GPU reuse tile3, 16 participating cores | 2796.82 | 81.60 |
| Above, GPU reuse tile4, 16 participating cores | 2820.84 | 82.77 |
| Tile3, 8 participating cores | 2765.54 | 75.25 |
| Tile3, 12 participating cores | 2890.55 | 82.79 |

CPU percentages refer to cold experts; hot experts always stay on the GPU.
The maximum measured process CPU sample with CPU experts is about 49% of all
32 logical processors, with concrete Gate/Up and Down work in the raw stats.
This confirms computation, not a target utilization or an inference average.

Exact pricing budgets the actual owned prefill buffers plus a margin. Elastic
KV initially maps 16K cells and leaves more slots for experts (4765 in these
combined screens, versus 3174 with the earlier full-KV conservative budget).
It keeps the 262K virtual context capacity and original INT8 format. Growth,
cache replacement and session restore need the separate acceptance gates.

These are one run per setting. The roughly 1.88x peak versus the earlier
44 tok/s is a screening result, not a guarantee across prompts or a final
repeated benchmark. Draft acceptance depends on the prompt. The user relaxed
the exact 2x target in favor of stable performance; final selection uses paired
runs, long outputs and correctness/quality checks recorded in the parent report.
