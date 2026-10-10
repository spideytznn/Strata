# MTP rollback and runtime diagnosis

2026-10-10, Windows, RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM, driver 617.14.
No model forward, server start, restart or stop was performed for this diagnosis.
The interactive timings were collected from the user's previous MTP4/Q8 service
log, not a new controlled benchmark. Prompt text and responses are not included.

## Desktop rollback

`START-NATIVE-262K.bat` again selects `rtx5090-262k-mtp2.json`: two drafts,
original BF16 MTP projections, BF16 draft batching off. Context 262144, prefill
8192, port 8880, INT8 KV, dedicated prefill workspace, FP32 NVFP4 decode,
all-host-header acceptance and resident experts remain configured. The desktop
shortcut already calls that repository launcher. It now selects the newly built
`strata-diagnostics.exe`; the SHA-256 is in `summary.json`. No running process is
replaced. Optional MTP4 profiles remain available for comparison.

## Observed long-request timings

The previous request reported 24332 fresh input tokens / 21580 ms (1127.5 tok/s),
then 193 generated tokens / 5938 ms (32.5 tok/s). Four chunks spent:

| Chunk start | Rows | Main + wait ms | Gather future wait ms | MTP ms | Checkpoint ms |
| --- | --- | --- | --- | --- | --- |
| 0 | 8192 | 8908.549 | 6102.3 | 17.412 | 0.001 |
| 8192 | 8192 | 5287.191 | 3124.6 | 8.306 | 24.372 |
| 16384 | 4793 | 3791.703 | 2284.5 | 4.045 | 0.001 |
| 21177 | 3150 | 3295.092 | 2025.6 | 3.112 | 0.001 |

MTP callbacks total 32.875 ms, checkpoints 24.375 ms, and gather waits 13537 ms.
The latter includes the entire async row preparation, not just SSD service.
The cumulative ngram `blocked` metric also includes cache/callback processing,
earlier requests and decode; it cannot be assigned entirely to this prefill.
`interactive-timing-excerpt.log` preserves the relevant original diagnostic lines.
Expert-file reads remained zero. In the long request 82996 / 149280 expert routes
(55.6%) were outside the hot GPU cache; the startup held about 3197 slots and
4585892608 bytes of dedicated prefill workspace.

## Idle raw-row I/O comparison

`ngram_io_bench` is CPU-only. It uses `PleReader` and the metadata offsets in
`ngram-io-spec.json`, never writes the original weights, and performs no model
forward, GPU work, FP8 decode or row-cache processing. Invocation:

```powershell
.\build-safetensors\ngram_io_bench.exe `
  bench\results\2026-10-10-safetensors-runtime\p8-diagnosis\ngram-io-spec.json 131072
```

The original D: shard has 128 logical FP8 segments. The existing G: FP8 GGUF
sidecar is contiguous. Both use 160-byte rows, 256 outstanding reads, 8 readers,
zero row cache, and identical row IDs (`mt19937`, seed 42). Every requested byte
matched in all six runs. This is a sampled row comparison, not a full-table hash.
Order alternates D/G, G/D, D/G; retain the first round rather than discarding it.

| Round | D: original ms | G: existing sidecar ms |
| --- | --- | --- |
| 0 | 335.627 | 842.713 |
| 1 | 119.720 | 984.212 |
| 2 | 114.476 | 914.799 |

Raw JSON includes read counts, bytes and submission-to-reap latency percentiles.
The reads are unbuffered, but drive-controller caches were not flushed. The PC
was not running the model, so this does not isolate behavior under GPU/DRAM/PCIe
load. It provides no evidence for moving the native default to G:. No sidecar
or mirror was created, and the shared existing FP8 file remains unchanged.

## Changes and remaining checks

`STRATA_PLE_TRACE=1` splits batch preparation into call-lock wait, cache lookup,
page planning, group wait, cache insertion, callback/decode and reader drain.
Those phases are caller wall time, not independent hardware latencies: reader
work overlaps callbacks and scheduling is included. Gather tracing also reports
raw-buffer resize and available Windows physical RAM. Numerical execution is
unchanged, and trace-off skips the new timing calls. The segment test verifies
that callbacks expose only completed row prefixes, including crossings,
duplicates, out-of-range zeros and an all-cached second pass with zero disk reads.

`STRATA_NATIVE_KV_GROW=1` enables native eligibility for the existing elastic-KV
scheduler. Native startup owns all experts, including mirrors of hot GPU slots;
the old resident-complement guard therefore need not forbid this path. The
desktop explicitly leaves it off. The separate
`config/native/rtx5090-262k-mtp2-elastic-test.json` enables it and
`STRATA_KVG_CHECK=1`, without changing context capacity, weight/KV precision or
dedicated workspace. All other original eligibility guards still apply.

Five CPU native tests pass; the segment test also passes with tracing enabled.
The SM120 build and existing VMM map/unmap/chunk-transfer test pass, as does the
existing PLE selftest. Logs are included here. These do not establish native
elastic-KV correctness for full prefill, adaptive swaps, MTP, conversation cache
restore or cancellation. Those model checks, followed by same-condition timing,
are required before enabling it in the desktop default. HIP/SYCL were not built.
The prefill/decode performance target has not been restored or claimed met.
