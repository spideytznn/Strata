# Native draft depth, 512-token continuations

RTX 5090 / 9950X3D / 96 GB, Windows, driver 617.14. Sequential private stdin
engines, no HTTP. Original safetensors, 262144 configured context, INT8 KV,
dedicated 8192 prefill, original BF16 projections, fully CUDA-pinned experts,
CPU75 cold-expert split, kernel copying, 40 pool tasks, exact workspace pricing,
elastic KV, canonical FP32 expert arithmetic and four-entry GPU weight reuse.
Binary/environment/config hashes and full output IDs are in `results.json`.

| Draft tokens | English cold / warm | Chinese cold / warm | Code cold / warm | Six-request median |
| --- | ---: | ---: | ---: | ---: |
| 0 | 63.58 / 63.07 | 62.21 / 63.34 | 60.02 / 60.39 | 62.64 |
| 1 | 88.15 / 88.27 | 67.82 / 86.36 | 78.77 / 83.86 | 85.11 |
| 2 | 91.64 / 98.50 | 97.40 / 102.97 | 88.54 / 91.73 | 94.56 |

Units: generated tokens / engine decode seconds, excluding prompt processing.
Every request generates 512 tokens and ends at the length cap. For each prompt,
all six output sequences (three depths, cold/warm) match exactly. Warm prefix
reuse is nonzero; expert/MTP post-residency source bytes are zero in all engines.

One-draft acceptance is 87.18%, 90.67%, 92.11% on English, Chinese and code;
two-draft acceptance is 83.07%, 87.37%, 86.63%. Each cold/warm pair has the same
acceptance count. Acceptance alone does not predict latency. Chinese draft1
cold records about 1.25 seconds cumulative blocked ngram time, versus about
0.04 seconds for the entire draft2 engine. These are distinct fresh processes;
the timing difference cannot be assigned entirely to draft depth.

The zero-draft run retains the loaded MTP weights to isolate draft computation;
it is not a measurement of a smaller engine with all draft weights unloaded.
Adaptive caches and ngram row caches remain enabled as in the operator profile.
First-request wall/TTFT includes graph capture; the printed decode speed does
not. A finite 512-token suite does not prove all speculative continuations or
EOS paths. Longer continuations and session restoration are separate checks.

Reproduce with `tools/native_operator_bench.py --new 512 --configs ...` using
the three configs preserved in the result. Cross-config output equality and
each config's own cold/warm equality are strict checks, not report-only checks.
No comparison to the original desktop speed is made by this table.
