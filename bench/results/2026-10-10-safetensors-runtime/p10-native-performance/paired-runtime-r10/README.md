# Three alternating core-runtime pairs

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14. Same original
NVIDIA safetensors, 262144 context, INT8 KV, 8192 dedicated prefill, BF16 MTP2.
Previous desktop is efficiency2 with handles already released. Candidate uses
efficiency10, CPU75/kernel/tasks40, canonical FP32 arithmetic, row unroll,
GPU weight reuse, exact workspace pricing, elastic KV and ordinary T1 commit.
Both profiles retain two drafts; no draft-depth increase explains the gain.

| Round | Previous prefill | Candidate prefill | Previous decode | Candidate decode |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2821.01 | 2830.84 | 45.07 | 85.81 |
| 2 | 2877.80 | 2843.34 | 45.30 | 86.21 |
| 3 | 2855.46 | 2828.60 | 45.23 | 83.06 |

Units are tokens/second. Medians: prefill 2855.46 -> 2830.84; decode
45.23 -> 85.81 (1.897x). Fresh sequential CLI processes; round 2 reverses
order. The varied P9 document has 8199 input tokens, 8198 processed in the
prefill phase and 128 generated tokens. All six output sequences are exactly
identical. Startup/first graph capture are excluded from printed decode timing.
No state hashing, logit dumps or prefill CUDA timing events are enabled.
Expert and MTP post-residency source counters remain zero in all six runs.

CPU routed entries are zero for the previous profile; candidate records
7.19..7.20 distinct / 9.45..9.47 routed CPU experts per layer of each verifier
round. Process CPU peaks are 39.2..49.6%, normalized across 32 logical CPUs,
including startup and prefill; these are not decode-average utilization values.
GPU clocks/memory/power and process CPU samples are retained with each command,
input/executable hash, output IDs and raw stdout/stderr.

The subsequent efficiency11 source change only adds disk-session identity
fields for the new arithmetic/commit choices. Its positive cache acceptance is
recorded separately; this table measures efficiency10 explicitly. The final
selected-binary measurements are recorded separately after the remaining gates.
Reproduce with tools/native_cli_bench.py and summarize with
native_performance_pair_summary.py, using the embedded command/env and frozen
P9 details-requests.json fixture. The original 822 -> 2465 prefill handle-release
comparison uses separate same-binary runs and CUDA events; do not attribute
that improvement to this CPU/cache comparison or mix its timing settings.
