# Final selected-binary comparison

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14. Same original NVIDIA
safetensors; previous desktop efficiency2 versus selected efficiency11. Both
use 262144 capacity, INT8 KV, dedicated 8192 prefill and two MTP drafts with
original BF16 projections. Startup weight handles are already released in both.
Candidate adds CPU75/kernel/tasks40, canonical FP32 expert arithmetic, row
unroll, GPU weight reuse, exact workspace pricing, elastic KV and ordinary T1
commit. The draft depth is unchanged.

| Round | Previous prefill | Selected prefill | Previous decode | Selected decode |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2850.64 | 2869.39 | 45.51 | 64.57 |
| 2 | 2814.53 | 2819.40 | 45.28 | 81.59 |
| 3 | 2797.23 | 2916.00 | 45.05 | 85.16 |
| Median | 2814.53 | 2869.39 | 45.28 | 81.59 |

Units: tokens/second. Decode median is 1.802x; prefill ratio 1.019x. Three fresh
alternating pairs, round 2 reversed; same varied P9 8199-token input, 8198
prefill tokens and exactly equal 128-token outputs across all six runs. All
expert/MTP post-residency source bytes are zero. No state hashes, logit dumps
or prefill CUDA timing events run here; builds and correctness engines have
finished. CLI startup/first graph capture are excluded from printed decode.
CLI first-token timing also includes initialization outside the prefill phase
and must not be described as warm HTTP time to first token.

The slower first selected run (64.57 tok/s) is retained, not discarded or
replaced. Its verifier host work is 9.921 ms/round versus 0.122 in the preceding
efficiency10 first pair: about half a second of extra host work over 51 rounds.
Pool-minus-host timing is similar. Ngram I/O counters are cumulative across
prefill and decode and do not independently identify the operating-system or
storage cause; no such cause is claimed. Final selected decode range is
64.57..85.16 tok/s. The preceding efficiency10 pairs (85.81 median) remain
archived separately, not substituted for the selected-binary median.

The complete stdout/stderr, commands/env, binary/input hashes, GPU clocks,
power/memory and process CPU samples are beside this file. CPU entries are
nonzero only in the candidate. Process CPU peaks include all phases and are
normalized across 32 logical CPUs; they are not decode-average utilization.
Original local manifest byte hashes and archived UTF8/LF hashes are mapped in
archive-provenance.json; input token-file bytes are preserved exactly.

Reproduce with tools/native_cli_bench.py and the embedded configs/commands,
then tools/native_performance_pair_summary.py --prefix delivered-pair. Keep
all three pairs, including slow runs. Earlier handle-release prefill evidence
822 -> 2465 uses separate same-binary/event-enabled measurements; do not mix
that timing setup with this comparison.
