# Native safetensors concurrency

The CUDA native backend can decode two text requests in one target-model window
with `"parallel": 2` in the server config (`--batch 2` / `--slots 2` in the
engine). This is opt-in. The original desktop launcher and its local settings
are unchanged.

## Local launch

Use `START-NATIVE-262K-PARALLEL.bat`, or the desktop
`Start-Strata-Safetensors-Parallel.bat`. Both use
`config/native/rtx5090-262k-parallel.json` and port 8880. Stop any other server
using that port before launching this one. The profile keeps original safetensors
weights, INT8 KV, 262144 context, a dedicated 4096-row prefill workspace, the
existing external GPU vision encoder and unrestricted HTTP Host headers.
It uses `strata-concurrency3.exe` and a 1536 MiB explicit reserve for runtime
graph allocations. The profile contains this PC's paths; adapt them elsewhere.

For a generic local config, `tools/native_config.py --parallel 2` adds the
server setting; its default is still one request. The measured desktop profile
contains additional arithmetic, memory and scheduling settings, so a generic
config is not a reproduction of its measurements.

The desktop parallel profile leaves MTP entirely disabled: no `--mtp`, no draft
weights loaded, `--spec 2` and `--suffix-draft 0`. Every request, including one
running alone, uses target-only decode. Removing MTP also removes its draft
weight/state allocation; the earlier acceptance profile reported 2620 MiB for
that allocation. Actual reclaimed VRAM and resulting cache sizing have not been
remeasured with this setting.

When another text request arrives, the server drains the solo generation and
resumes it in an independent batch slot. Slots verify one target token each per
window. A custom config can still load native MTP for solo requests, as in the
initial acceptance runs below; concurrent slots remain target-only.
`--batch-mtp` is explicitly refused for native weights until its combined state,
sampling and performance are validated. No target weights or projections are
requantized by concurrency.

More than two requests wait for a free slot. Admissions share one prompt reader;
active slots receive decode windows between prompt chunks. This is batch decode
with chunk-level admission scheduling, not simultaneous prompt GEMMs. The
maximum wait between those decode opportunities includes a full prefill chunk.
Disk `/slots/0` save/restore remains unavailable while batch mode is enabled.
In-memory live-slot, checkpoint and parked-conversation reuse remain available.

## State, sampling and memory

### Adaptive exchange correction staged after initial acceptance

Inspection after the initial runs found that `strata-concurrency2.exe` accumulates
routing heat in batch windows but does not periodically schedule new adaptive
rounds there. Solo generation still exchanges experts. Initial logs' cumulative
promotion counts therefore do not prove promotions during two-row decode.

The correction is built separately as `strata-concurrency3.exe`; the desktop
config now selects it for the user's next-start validation. No running process
was stopped or replaced. The initial acceptance data below use `concurrency2`;
the correction has not yet passed full GPU acceptance.
The native batch boundary now reuses the existing hot/cold swap policy every
`--adapt-every` windows, with at most `--adapt-swaps` promotions per round (4 and
96 in this profile). Heat includes routed entries from all active rows. Copies
complete before the next window; exchanges are deferred while prefill borrows
expert-cache slots. The dedicated workspace profile does not take that loan.
The first correction uses a synchronized boundary; its speed is unmeasured.

CUDA SM120 compilation passed. GPU parity, real two-row promotion and source-read
acceptance remain pending because the user's existing service occupies the GPU.
The acceptance harness's new `--adaptive` option retains configured adaptation
and additionally requires nonzero promotions from logged `active_rows=2` rounds:

```powershell
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-concurrency3.exe --output bench/results/local-batch-adaptive --context 262144 --vision --limit 128 --adaptive
```

### Per-request state

Each slot has its own GDN recurrence/conv history, PLE history and previous token
pair, QSA KV/indexer and committed token list. The target verifier's batch commit
now applies the same opt-in canonical QSA spare-row cleanup as its solo commit.
Native windows stage a separate penalty tail for each row, including greedy
repetition/frequency/presence penalties. Temperature, top_p, top_k, min_p, seed
and Philox position are taken from that row's request. Legacy batch callers that
do not supply row histories retain their existing sampler path.

The native elastic KV registry can include the main session and both slots.
It initially maps 16384 cells, grows all those pools to their common required
length, and recycles physical pages from the expert cache. A short admission
must retain the longest active, yielded or cached slot's cells. Growth is also
checked before each batch window. Virtual addresses and graph pointers remain
stable; the original resident RAM expert mirrors remain available when a GPU
expert slot is surrendered. This first implementation maps all slot pools to
the same length, rather than growing each slot independently.

On the recorded 262K two-slot run, the combined elastic KV pools initially used
0.72 GiB against 9.54 GiB at full capacity. The two sessions' non-KV/initial
allocations were reported as 1.12 GiB total. These are overlapping accounting
views, not amounts to add together. There is still less room for GPU experts
than in the serial profile; concurrency can reduce single-request speed.
The dedicated prefill workspace stays allocated before expert-cache sizing.

Vision kernels use a shared mutable M-RoPE position table. The server therefore
allows text requests to share the engine but gives image requests exclusive
generation, after existing text slots and cancellation drains finish. Waiting
images take precedence over newly arriving text. The engine also refuses a
native image admission that would overlap active batch slots, including direct
stdin callers. The external encoder can prepare the image before generation
gets its exclusive permit. This does not establish parallel image generation.

## Validation on this PC

RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, native Windows CUDA SM120,
2026-10-11. These initial runs loaded MTP for solo requests, before it was
disabled in the desktop parallel profile. Raw data and commands are in
[`bench/results/2026-10-11-native-concurrency`](../bench/results/2026-10-11-native-concurrency/README.md).

- Raw protocol checks compare solo and two-slot outputs exactly for English and
  Chinese/code prompts, greedy decoding, different seeded temperatures/top-k/
  top-p, and repetition/frequency/presence penalties. Short output limits,
  cancellation/resume and subsequent turns are exercised.
- The 262K configuration passes those checks with the vision encoder loaded.
  A 22777-token fixture grows the common KV mapping to 24576 cells; its output
  and later continuation match solo after a short request is admitted.
- Two near-capacity prompts of 261000 and 261007 tokens each generate 256
  tokens matching solo, with overlapping output intervals (6.488–24.715 s and
  16.532–27.110 s from admission). The common KV mapping reaches 262144 cells
  and 9.54 GiB; the GPU expert cache falls to 128 slots. These repeated-text
  fixtures establish tested capacity/state coexistence, not retrieval quality.
- Real HTTP SSE checks cover one, two and three simultaneous clients over the
  OpenAI route, an Anthropic request, client disconnection while another request
  runs, recovery afterwards, and a red image between text requests. Text matches
  the corresponding serial responses; the image returns red. HTTP usage retains
  cached input tokens. Each round's answer cap is 96 target tokens.
- Runtime diagnostics report zero expert file bytes; engine shutdown reports
  zero post-residency expert and MTP source bytes. These are logical source
  counters, not physical-disk telemetry. Ngram SSD reads remain allowed.

These are finite regression tests against the same NVFP4 target, not a broad
quality benchmark or proof of equivalence to unquantized BF16. Native HIP/SYCL,
other GPUs, multi-GPU and more than two native slots have not been validated.

## Latency and throughput screen

Before disabling MTP in the desktop parallel profile, the same executable was
run serially and with two slots, using 262K INT8 KV,
4096 dedicated prefill, vision loaded, native MTP for solo requests, 1536 MiB
reserve, greedy sampling and adaptive expert exchange. Three rounds per client
count followed reference warmups. Each client requested 96 tokens. Values below
are medians; throughput includes prompt/admission and HTTP time.

| Simultaneous clients | Serial aggregate tok/s | Two-slot aggregate tok/s | Serial last visible first-content latency | Two-slot last visible first-content latency |
|---|---:|---:|---:|---:|
| 1 | 79.19 | 61.25 | 0.131 s | 0.144 s |
| 2 | 71.34 | 48.05 | 1.438 s | 0.455 s |
| 3 | 76.94 | 71.65 | 2.519 s | 2.744 s |

The two-client last first-content wait fell in this screen, while aggregate
throughput and the one-client rate fell. Three clients did not improve their
last first-content wait. These were sequential profile runs, not interleaved
paired trials; cache state adapted and GPU clocks were not recorded. They are
functional/latency screening evidence, not a stable peak-throughput claim.
The serial desktop entry remains the choice for one user's maximum speed.
