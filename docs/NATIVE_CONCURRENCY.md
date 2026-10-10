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
It uses `strata-concurrency5.exe`, enables independent session KV and direct
device slot copies, and keeps a 1536 MiB explicit reserve for runtime graph
allocations. The profile contains this PC's paths; adapt them elsewhere.

For a generic local config, `tools/native_config.py --parallel 2` adds the
server setting; its default is still one request. The measured desktop profile
contains additional arithmetic, memory and scheduling settings, so a generic
config is not a reproduction of its measurements.

The desktop parallel profile leaves MTP entirely disabled: no `--mtp`, no draft
weights loaded, `--spec 2` and `--suffix-draft 0`. Every request, including one
running alone, uses target-only decode. Removing MTP also removes its draft
weight/state allocation; the earlier acceptance profile reported 2620 MiB for
that allocation. The target-only acceptance below measures the resulting KV and
cache sizing; WDDM allocation variation still affects the available cache.

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

### Adaptive exchange correction

Inspection after the initial runs found that `strata-concurrency2.exe` accumulates
routing heat in batch windows but does not periodically schedule new adaptive
rounds there. Solo generation still exchanges experts. Initial logs' cumulative
promotion counts therefore do not prove promotions during two-row decode.

The correction was built separately as `strata-concurrency3.exe` and accepted
on this PC on 2026-10-11 with MTP entirely unloaded and vision loaded. The initial
acceptance data below use `concurrency2`; the new records are separate.
The native batch boundary now reuses the existing hot/cold swap policy every
`--adapt-every` windows, with at most `--adapt-swaps` promotions per round (4 and
96 in this profile). Heat includes routed entries from all active rows. Copies
complete before the next window; exchanges are deferred while prefill borrows
expert-cache slots. The dedicated workspace profile does not take that loan.
The correction uses a synchronized boundary.

CUDA SM120 compilation and GPU parity passed, including 128 tokens per sampling
pair, cancellation/resume, continuation, limits and EOS. The run logged 98 actual
two-row adaptive rounds and 9367 promotions in those rounds. Post-residency expert
and MTP source bytes were zero. The acceptance harness's `--adaptive` option
retains configured adaptation and requires nonzero promotions from logged
`active_rows=2` rounds:

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
The original path initially maps 16384 cells, grows all those pools to their common required
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

### Independent KV and direct slot copies

Two optional native, target-only paths are implemented in `strata-concurrency5.exe`:

- `STRATA_NATIVE_KV_INDEPENDENT=1` grows and shrinks only the selected session's
  QSA pools. The main prompt reader and each slot keep their own mapped extent.
  A destination grows before cached KV is copied into it; a replaced slot can
  release pages beyond its new conversation. Other live/cached slots retain
  their pages. Growth is checked before every batch window. Physical pages still
  come from the existing expert-cache VMM allocator; graph addresses remain stable.
- `STRATA_NATIVE_SLOT_COPY=1` transfers the fully resident KV, GDN and PLE state
  directly between device allocations. It avoids temporary host snapshots when
  migrating a request into/out of a slot. An earlier checkpoint still restores
  its saved running state, then copies the authoritative KV from the live slot.
  RAM conversation snapshots and their file format are unchanged.

Both flags default off and require native batching with MTP unloaded. Direct
copies require fully resident device KV (mode 0); they do not implement streaming
or draft-ring migration. No projection, expert arithmetic, quantization or
attention kernel changes are involved. Source and destination remain idle until
copies finish. Copy failure uses the existing admission refusal/slot invalidation
or full-prompt recomputation path; partial destination state is not committed.

The combined paths passed raw-protocol GPU acceptance with the vision encoder
loaded: greedy/seeded/penalized sampling, cancellation/resume, followups, limits,
EOS, two overlapping 261000/261007-token slots with 256 output tokens each, and
a 124135-token conversation beside a short request. Live-slot continuation
matched solo; after both slots were replaced by short conversations, the long
request was admitted again and matched solo. That last admission reread its
prompt rather than hitting the RAM cache, so it does not prove a RAM-cache hit.
175 actual two-row adaptive rounds promoted 15279 experts; post-residency
expert and MTP source bytes stayed zero.

At full capacity, the three target KV groups mapped 9504 MiB (9.28 GiB) with MTP
unloaded. With just the main group full and two initial 16384-cell slots, they
mapped 3648 MiB. With one 131072-cell slot and the main/other slot shrunk to
8192, they mapped 1872 MiB (1.83 GiB); replacing that slot with a short one
returned the total to 432 MiB. These are mapped KV bytes, not total GPU memory.
Two full-length slots still need their full KV; the savings apply to unequal
lengths and unused state. The historical 9.54 GiB figure above included draft KV.

4219 CUDA snapshot checks passed, including byte equality of direct copies
against the snapshot path at page/block boundaries, different KV formats,
rotation mismatch refusal, and another session's size/bytes surviving selected
growth and shrink. Server/gate/config tests also passed. CUDA SM120 was built;
HIP/SYCL were not built or validated on this PC.

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

## Target-only latency and throughput screen

After the adaptive correction, `strata-concurrency3.exe` was tested with MTP
entirely unloaded, INT8 262K KV, 4096 dedicated prefill, vision loaded, reserve
1536 MiB and adapt 4/96. Only the server's parallel setting differed. Each client
requested 96 greedy target tokens; three rounds followed reference warmups.
These medians include admission and HTTP time.

| Simultaneous clients | Serial aggregate tok/s | Two-slot aggregate tok/s | Serial last first-content latency | Two-slot last first-content latency |
|---|---:|---:|---:|---:|
| 1 | 64.67 | 61.43 | 0.136 s | 0.141 s |
| 2 | 58.10 | 73.27 | 1.640 s | 0.420 s |
| 3 | 55.19 | 62.24 | 3.511 s | 2.866 s |

Two-client aggregate throughput rose 26.1% and three-client throughput 12.8%;
the one-client rate fell 5.0%. The measured two-client first-content wait fell
74.4%. All 24 shared completed HTTP responses matched across configurations.
Both runs passed disconnect/recovery and Anthropic checks; the parallel run also
passed text/image exclusion. Expert source bytes stayed zero. Raw records are
`http3-no-mtp-serial`, `http3-no-mtp-parallel` and
`no-mtp3-http-summary.json` in the evidence directory.

These sequential, warm-cache screens do not control GPU clocks or establish a
peak rate. Startup varied from 70.878 to 482.566 seconds as source-read time
changed; startup is excluded from the request measurements.

## Independent-KV mixed-length HTTP screen

The same `strata-concurrency5.exe` was run with both new flags off, then both
on, with two slots, MTP unloaded and the same profile above. The first fixture
has 124135 formatted tokens of repeated text; the other two are short code/tree
questions. Each request asks for 96 greedy tokens. Three measured rounds follow
reference warmups; repeated long prompts reuse 124128 input tokens. These are
warm conversation measurements, not fresh 124K prefill measurements.

| Clients | Flags off aggregate tok/s | Flags on aggregate tok/s | Flags off last first-content latency | Flags on last first-content latency |
|---|---:|---:|---:|---:|
| 1 | 42.47 | 47.11 | 0.587 s | 0.367 s |
| 2 | 44.33 | 51.97 | 1.260 s | 1.273 s |
| 3 | 42.21 | 49.84 | 4.776 s | 3.865 s |

Measured aggregate gains were 10.9%, 17.2% and 18.1%. The two-client last
first-content wait was essentially unchanged (+1.0%); the three-client wait
fell 19.1%. All 27 shared completed responses matched, including disconnect
recovery, image exclusion and Anthropic. Both shutdown logs report zero expert
and MTP source bytes. The same binary SHA256 is recorded in both results.

Common KV growth mapped 4.64 GiB at 131072 cells in the flags-off run.
With flags on, one 131072-cell group plus two 8192-cell groups mapped 1872 MiB;
two long groups plus a short one mapped 3312 MiB. Main/slot copies can temporarily
require both long groups. Freed pages return to the GPU expert cache.

Direct copies do not eliminate allocation work. The median logged slot
admission rose from 56 to 107 ms because admissions also grow/shrink their own
pools and adjust the expert cache. Median slot restoration fell from 41.9 to
22.05 ms. End-to-end gains above include those costs.

Raw records are `http5-long-baseline`, `http5-long-optimized` and
`optimized5-long-http-summary.json`. Runs were sequential; clocks were not
controlled, cache placement adapted and startup (104.458/61.728 s) is excluded.
These figures do not establish general speed or retrieval quality.

The same off/on comparison also passed with all three prompts short. For
1/2/3 clients, median aggregate rates were 65.26/75.00/66.02 tok/s off and
66.18/78.43/69.75 on (+1.4%/+4.6%/+5.6%). Last first-content latency was
0.134/0.429/2.679 s off and 0.124/0.376/2.574 s on. Median slot admission was
63.4 versus 22.8 ms; restoration was 35.1 versus 27.9 ms. All 27 completed
responses matched, and the same HTTP/residency checks passed. Raw records are
`http5-short-baseline`, `http5-short-optimized` and
`optimized5-short-http-summary.json`. The same measurement limits apply; the
small short-request differences are not a guaranteed gain.

The separate local parallel desktop profile now opts into both paths. Their
engine defaults remain off; the original serial launcher is unchanged.

## Historical latency screen with solo MTP

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
