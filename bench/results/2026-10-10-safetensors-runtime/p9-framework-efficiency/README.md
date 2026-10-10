# Framework efficiency investigation, paused for the user's reboot

RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13.0, driver 617.14.
Offline CLI inference only: no HTTP or web service was started. The original
checkpoint and Q4XL deployment were read-only. Desktop defaults are unchanged.

## Measured startup-handle fix

Native startup retained buffered safetensors file handles after all weights had
become resident. `STRATA_NATIVE_CLOSE_FILES=1` now closes those handles and seals
all subsequent resident-payload reads. PLE keeps its independent DirectFile
handles. Tensor metadata, stable resident weights and I/O counters remain valid.
This experiment changes neither the ngram file nor its codes, offsets or scale.
The precise Windows cache/coherency mechanism has not been independently traced.

The fixture uses the first part of repository `docs/DETAILS.md`, with the original
HF tokenizer and a chat wrapper: 8199 input tokens, 8198 processed in prefill,
128 greedy output tokens. The first chunk actually executes 8192 tokens. Context
262144, INT8 KV, dedicated prefill workspace, MTP depth 2, BF16 projections,
DMA expert copies, 2048 MiB reserve and the same expert profile are retained.
The source text SHA-256 is
`7c23c234ae5a84337da24403d1c0f8b26bcdf03b93525f981bc46c3f95cbc55b`.

| Run | Startup handles | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| details-fullkv-r1, initial binary | retained | 822.78 | 44.37 |
| details-close-r1, staged binary | closed | 2485.34 | 44.70 |
| details-off-r2, same staged binary | retained | 791.23 | 42.81 |

The first closed-handle run's PLE gather took 473.916 ms for 131072 rows / 87252
pages, compared with 7258.773 ms in the initial reference. Time waiting at layer
1 fell from 6643.3 ms to zero. All completed document runs generated exactly the
same 128 tokens. Every completed engine reports post-residency expert and MTP
source bytes = 0. These are requested-source counters, not physical SSD counters.
The staged binary's hash is in each manifest. GPU clock/power samples are present
for the closed run and same-binary control; the earlier exploratory runs predate
the telemetry addition.

This is one successful candidate run and two reference runs, including a
same-binary control. It is **not** three-round acceptance and **not** a 2x decode
result. `details-close-r2` was stopped during startup solely because the user
requested reboot; its manifest explicitly records interruption and no throughput.

## Other experiments

- Elastic native KV increases the expert cache from 3174 to 4377 slots. The
  repeated-text fixture goes from 44.66 to 52.55 decode tok/s, with identical
  tokens. The document fixture goes from 44.37 to 49.50. It does not fix the PLE
  stall: document prefill stays at 811.42 tok/s. Its startup overlapped compilation
  for part of loading; treat it as exploratory, not acceptance.
- The repeated-text fixture touches only 582 pages for its first 131072 PLE rows.
  Its 2009.43/3154.85 prefill numbers do not represent a varied long document.
- Four-output-row BF16 single-token reuse is rejected. All 19 synthetic shapes
  pass bitwise parity, but K/N 2560/2560 is 4.421 vs 5.227 us, 2560/6144 is
  7.720 vs 8.770 us, and 6144/2560 is 7.186 vs 9.238 us (five-round medians).
  The explicit benchmark entry point remains for reproduction; the production
  entry point always uses the original one-row kernel. No new tuning flag selects it.
- Inspection found native safetensors bypasses the inherited async-arena branch
  that starts GEMM warmup. `STRATA_NATIVE_PREWARM=1` reuses that warmup while
  experts load. It is compiled but has **no end-to-end measurement yet** and is off.

## Validation and restart point

CUDA SM120 build succeeded; all five CPU safetensors CTests pass, including
source guards after closing resident handles. HIP/SYCL were not built. There is
no session-cache or >16384-cell native elastic-KV acceptance from this phase.
No API service or user's process was stopped. Only the exact owned benchmark
process for `details-close-r2` was terminated.

Optional profile: `config/native/rtx5090-262k-io-fix-test.json` points to the
staged `strata-efficiency2.exe`, enables only the handle fix, and keeps wildcard
Host, INT8 KV, 262144 context, 8192 prefill and BF16/MTP2. The desktop still uses
the previous `rtx5090-262k-mtp2.json`; nothing starts automatically after reboot.

Next: finish alternating three-run same-binary handle controls, then test a
24583-token document for real inter-chunk waits. Independently measure existing
`STRATA_DMA_BATCH=1`, kernel copying, exact owned-workspace pricing and elastic
KV for decode; include setup/adaptation cost, output equality and zero expert
reads. Test long-context growth and cache restore before selecting elastic KV.
Do not enable the rejected BF16 row experiment or claim decode has doubled.

Example offline continuation from the repository root (GPU must be idle):

```powershell
.\.venv-native\Scripts\python.exe tools/native_cli_bench.py --exe build-native-engine/strata-efficiency2.exe --fixture bench/results/2026-10-10-safetensors-runtime/p9-framework-efficiency/details-requests.json --case details-8199 --out logs/efficiency/reboot-close-r2 --env STRATA_NATIVE_CLOSE_FILES=1 --env STRATA_PREFILL_TIMING=1
```

Use a fresh output directory per run. Harness guards refuse another running
Strata process or a busy GPU. Raw command, environment, executable/token hashes,
stdout/stderr and GPU samples are retained. Process duration includes startup;
reported inference timings are separate. This CLI does not exercise conversation
reuse or the HTTP path, and its first-token time includes first graph capture.

Archived build logs and the microbenchmark JSONL normalize text line endings and trailing whitespace; local raw originals remain in `logs/`.
