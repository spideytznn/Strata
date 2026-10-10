# Native runtime performance follow-up, 2026-10-10

RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, Windows, NVIDIA driver 617.14.
Original NVIDIA NVFP4 safetensors, 262144 context, INT8 KV, dedicated 8192-token
prefill workspace, BF16 projections, native MTP2, adaptive expert cache, 2048 MiB
reserve. Only the startup handle release changes in the first comparison.

## Startup weight handles: three alternating pairs

Same `strata-efficiency2.exe` in all six runs. Command, environment, executable
and input hashes, raw stdout/stderr, GPU and process CPU samples are beside this
file. Input is the varied 8199-token documentation fixture from P9; the batched
prefill processes 8198 tokens as 8192 + 6, followed by the token path.

| Round | Handles kept: prefill | Handles closed: prefill | Kept: decode | Closed: decode |
| --- | ---: | ---: | ---: | ---: |
| 1 | 834.31 | 2465.30 | 44.48 | 41.42 |
| 2 | 815.26 | 2464.11 | 44.08 | 44.08 |
| 3 | 822.01 | 2473.20 | 44.07 | 43.96 |
| Median | 822.01 | 2465.30 | 44.08 | 43.96 |

Units: tokens/second. Round 2 reverses the off/on order. Each process loads its
own model sequentially; no API service or other engine runs concurrently. All
six generated 128-token sequences are exactly equal. All six report zero
post-residency expert and MTP source bytes. Source counters are logical requested
I/O, not physical SSD counters. Ngram reads remain permitted.

The prefill improvement is 3.00x on this fixture. Decode is unchanged within
run variation; this is **not** a doubled decode result. First graph capture is
outside the printed decode timing. CLI process duration includes startup and
must not be read as steady server first-token latency.

The handle experiment supports releasing buffered startup handles before runtime
ngram reads. It does not identify a specific Windows cache/driver mechanism.
The original file, offsets, ngram precision and expert bits are unchanged.

## Acceptance and further experiments

`io-fix-pair-acceptance/results.json` records the complete old desktop config and
handle-fix config. All nine requests per config pass: 8199- and 24583-token
documents, their warm repeats, A/B/A, and disk SAVE/RESTORE. The generated tokens
and all committed main-model state fingerprints match across configs. Both
warm document outputs also equal their own initial outputs. Prefix reuse is
8192 and 24576 tokens; A/B/A resumes at 28 tokens, disk restore at 43. Expert and
MTP source bytes after residency remain zero.

On the 24K initial request, the fixed config reports no layer-1 PLE wait in all
three 8192-token chunks. This is a correctness sequence after an earlier 8K
document, not an independent three-round long-input throughput measurement.
State hashing adds overhead. The new CPU build overlaps only the final part of
this correctness sequence, not the six CLI performance runs above.

The first acceptance attempt stopped because its test incorrectly demanded
8198 reused tokens instead of the real 8192-token checkpoint boundary. The
check was corrected to require the last full chunk boundary and the complete
pair was rerun. The failed attempt remains in local logs, with no engine failure.

The desktop `rtx5090-262k-mtp2.json` now points to `strata-efficiency2.exe` and
enables only `STRATA_NATIVE_CLOSE_FILES=1`. The inherited/global default stays
off. No public server was started or user process modified. Wildcard Host,
262144 context, INT8 KV, dedicated 8192 workspace, BF16 projections and MTP2
remain the operator settings.

CPU split, batched DMA, kernel copying, exact workspace pricing and elastic KV
are separate experiments. No doubled decode or selection is claimed before
their runs. The staged CPU row-unroll experiment is opt-in; its microbenchmark
is not an inference-throughput result.

Reproduce a pair with `tools/native_cli_bench.py`, P9 `details-requests.json`,
case `details-8199`, the same executable, and
`--env STRATA_NATIVE_CLOSE_FILES=0` / `=1`. Use a fresh output folder each time.
