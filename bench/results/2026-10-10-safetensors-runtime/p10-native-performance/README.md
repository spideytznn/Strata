# Native runtime performance follow-up, 2026-10-10

RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, Windows, NVIDIA driver 617.14.
Original NVIDIA NVFP4 safetensors, 262144 context, INT8 KV, dedicated 8192-token
prefill workspace, BF16 projections, native MTP2, adaptive expert cache, 2048 MiB
reserve. Only the startup handle release changes in the first comparison.

## Selected desktop profile

The desktop now selects efficiency11 with CPU75 cold experts, kernel copying,
40 pool tasks, canonical FP32 expert arithmetic, row unroll, GPU weight reuse,
exact workspace pricing, elastic KV and ordinary single-token commit. Original
BF16 MTP2, INT8 KV, 262144 total context, dedicated 8192 prefill and wildcard
Host remain. Global/inherited arithmetic defaults are unchanged; this is the
measured profile for this machine.

Three alternating final-binary pairs against the preceding native desktop
measure prefill median **2814.53 -> 2869.39 tok/s**, decode
**45.28 -> 81.59 tok/s (1.802x)**. All six 128-token sequences are exactly equal.
The selected range is 64.57..85.16 decode; the slow run is retained. Both sides
use two drafts. See [all final pairs](delivered-pairs/README.md), including raw
commands, telemetry and the host-wait outlier. The earlier efficiency10 pairs
are separately retained in [paired-runtime-r10](paired-runtime-r10/README.md),
and are not substituted for the final-binary median.

The final profile passes [complete config/state acceptance](stable-full-config/README.md),
[64K automatic KV tests](stable-kv-auto-64k/README.md), and
[262000-token input with all 262144 KV cells mapped](capacity-262k/README.md).
Cold/warm/disk-restored main states agree; all 19 resident-content/table/alias
audits in each capacity run have zero errors. Expert/MTP source bytes remain
zero after residency. The capacity fixtures repeat documentation and are not
throughput or long-distance retrieval-quality tests.

[2048-token greedy stability](draft-stability-2048/README.md) and
[512-token fixed-seed sampled stability](sampled-stability/README.md) compare
English, Chinese and code, cold/warm, across draft modes. The rejected prior
warm-code mismatch remains archived; ordinary T1 commit passes the full rerun.
These are finite regression checks, not universal speculative stability.
[Session identity](session-identity/README.md) rejects incompatible arithmetic
before restoring payload, while same-config SAVE/RESTORE passes.

The fully unloaded `rtx5090-262k-no-mtp.json` alternative passes
[cache/state checks](no-mtp-cache/README.md) and sampled comparisons; its
[single screen](no-mtp-screen/README.md) measures 3100.40 prefill / 50.50 decode.
CPU25/50 profiles inherit the selected flags but lack full combined speed
measurements. The preceding desktop is retained as `rtx5090-262k-gpu-reference.json`.
No HTTP service was started; restart the existing desktop launcher to load the
profile. CUDA SM120 was built and tested; HIP/SYCL were unavailable and untested.

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

The handle-release milestone selected `strata-efficiency2.exe` for the desktop
`rtx5090-262k-mtp2.json` and
enabled only `STRATA_NATIVE_CLOSE_FILES=1`. The inherited/global default stays
off. No public server was started or user process modified. Wildcard Host,
262144 context, INT8 KV, dedicated 8192 workspace, BF16 projections and MTP2
remain the operator settings.

CPU split, batched DMA, kernel copying, exact workspace pricing and elastic KV
were screened separately before the combined selection above. CPU row-unroll
microbenchmarks are kept separate from inference-throughput results.

## CPU/GPU arithmetic and long-output checks

`arithmetic-consistency/` records independent FP64 expert checks, CPU/GPU
bit comparisons and 234 teacher-forced full-logit positions. Canonical arithmetic
matches CPU75 and all-GPU logits bit-for-bit on those positions. It is a numerical
change from the original GPU arithmetic, not a claim of unquantized-model parity.
`cpu-row-unroll/`, `gpu-weight-reuse/` and `screening/` separate kernel measurements
from single-run model screens. Those screens alone do not select a default.

`draft-stability-512/` passes no-draft/one-draft/two-draft cold/warm outputs.
The longer three-task test then found a code warm mismatch at output index 1061;
the failed candidate is retained in `draft-stability-2048-rejected/`. A control
retaining the ordinary commit graph for one-token windows, followed by the full
same-history rerun, passes all 12 requests and 24576 output tokens. See
`onecommit-diagnostic/` and `draft-stability-2048/`. This is finite regression
evidence; no particular faulty kernel instruction or universal stability is claimed.

`full-config-acceptance/` and `kv-auto-64k/` hold the preceding cache/state gates.
The latter validates 65543-token input, automatic expert-slot lending, KV trim,
disk restore and equal committed states, with zero resident-content audit errors.
It is a capacity/cache fixture, not a throughput result or a full 262K input test.

Reproduce a pair with `tools/native_cli_bench.py`, P9 `details-requests.json`,
case `details-8199`, the same executable, and
`--env STRATA_NATIVE_CLOSE_FILES=0` / `=1`. Use a fresh output folder each time.
