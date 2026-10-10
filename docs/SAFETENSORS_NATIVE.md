# Native safetensors backend

## Current runtime

`strata --model <original HF directory>` now runs text inference directly from
NVIDIA safetensors. `--safetensors` is an alias. The independent branch remains
`codex/safetensors-native`, based on `d167eb89301a02e0d6299f49d35494ba5096bc10`.
The original checkpoint and Q4XL deployments are read-only inputs.

The loader validates the fixed Qwen3.8-Flash-Next geometry and binds the existing
Strata forward, scheduler, cache and prefill code. It preserves NVFP4 weight
codes and scales, ordinary BF16 weights, FP8 ngram data and original FP8 MTP
experts. There is no required GGUF, expert pack or converted MTP artifact.
A 63.282 GiB owned expert arena is physically locked in system RAM, including
copies of experts currently in VRAM. Failure to lock the whole arena aborts
startup. The initial path uses a separate 337.5 MiB pinned staging arena.
`STRATA_NATIVE_ALLOC_PINNED=1` instead allocates the whole arena with
`cudaHostAllocMapped`, eliminating that host staging copy. The generated quality
config now selects this measured path; a bare CLI keeps the initial reference
defaults. Model expert and MTP source reads are sealed after loading;
only the segmented ngram table remains demand-read during inference. Source
counters measure requested bytes, not physical SSD activity; physical locking
prevents the owned expert arena from being paged out.

Native CPU experts use FP32 activations and intermediates with the unchanged
NVFP4 codes. The default GPU path uses the inherited FP32-activation reference
kernels. `STRATA_NVFP4_TC=1`, `2`, or `3` selects the existing SM120 FP4 MMA path
with one, two or three activation terms. These are explicit test choices until
end-to-end measurements establish a quality and latency case for a default.
The original GGUF/pack path retains its defaults.

`--mtp native` loads this checkpoint's 29 BF16/F32 draft tensors and 512 FP8
experts. The draft reuses Strata's attention, rollback, sampling and verification;
the main model verifies every proposed token. Draft attention currently follows
Strata's dense/window path; the saved MTP indexer weights are validated and kept
but are not used for sparse draft selection. Multi-GPU, batching, pipeline >1,
cross-model drafts and MTP requantization are refused by this native path.
The desktop profile enables the inherited `--vision` / GENI embedding path with
the installed external BF16 image encoder, including image rotary positions
and image-aware conversation reuse. Main-model weights remain safetensors;
the read-only visual sidecar uses the existing BF16 mmproj GGUF and vocabulary
metadata GGUF, without converting the main model. It uses the GPU and caps
each image at 1024 tokens. Default sampling is the official thinking preset
(temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0,
repetition_penalty 1). Explicit requests and shared web settings override it.
The updated desktop binary is `strata-efficiency12.exe`; this sampling/vision
configuration has not been throughput-benchmarked at the user's request.
Earlier efficiency11 measurements below remain text-only greedy results.
The [private image smoke check](../bench/results/2026-10-10-safetensors-runtime/p11-native-vision/README.md)
passes solid-color recognition, warm image reuse, A/B/A, return to text and
a capped request with official thinking sampling. This is not a visual quality
suite, image-session disk-restore check or a performance comparison.
The HF tokenizer and all added tokens are used directly. The supplied froggeric
v22.5 template is retained in `config/native/froggeric-v22.5.jinja`.

## Runtime correctness evidence

The measured text-only desktop profile used `strata-efficiency11.exe`, 262144 total
context, INT8 KV, dedicated 8192 prefill and original BF16 MTP2. Cold decode
experts use CPU75 (`--pcie-frac 0.25`), kernel copying and 40 pool tasks, with
15 workers plus the host on 16 physical cores. Hot experts stay on the GPU.
`STRATA_PREFILL_CPU_SHARE=0` keeps prefill on the GPU. CPU fractions describe
cold experts, not total model work or utilization targets.

The opt-in profile enables CPU row unroll, GPU weight reuse, canonical FP32
expert arithmetic, exact workspace pricing and elastic KV. It closes startup
weight handles and uses `STRATA_ONE_TOKEN_COMMIT=0` to retain the ordinary commit
graph. BF16 MTP batching, Q8 projections and MTP4 stay off. These choices are
local profile settings; the inherited/global arithmetic defaults are unchanged.
Canonical arithmetic is a rounding change from the original GPU path: the
234-position teacher comparison gives 232 equal argmaxes and mean KL
0.0000999431, not unquantized-model equivalence. Canonical GPU versus CPU75,
however, matches every complete 248320-logit row bit-for-bit in that suite.
Independent real-expert FP64 and provider-bit tests are retained separately.

Three alternating final-binary CLI pairs on the varied 8199-token document
measure prefill median 2869.39 tok/s and decode 45.28 -> 81.59 tok/s (1.802x),
with the same two drafts and all 128 output tokens equal. Candidate decode
range is 64.57..85.16; the slow run is retained. Original handle release
separately measured prefill 822 -> 2465 in same-binary/event-enabled pairs;
those settings must not be mixed with the final event-free timings.
Complete commands, counters, clocks and CPU samples are in
[`p10-native-performance`](../bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md).
CLI first-token timing includes initialization beyond the printed prefill
phase and is not a warm HTTP latency measurement.

The final candidate passes 27 full-config 8K/24K, prefix, A/B/A and disk-restore
requests. Original-profile states remain equal to the preceding binary;
canonical all-GPU and CPU75 committed states match exactly. All own cold/warm
and restored states match. Automatic KV/cache checks also pass at 65543 and
262000 input tokens, actually mapping all 262144 cells in the latter. Long
cold/warm/disk-restored states agree, and all 19 resident-content/table/alias
audits in each capacity run have zero errors. Expert/MTP source bytes remain
zero. The repeated-document capacity fixtures are not throughput or
long-distance retrieval-quality tests. The limit counts input and output together.

The longer stability gate initially found a code warm mismatch at output
index 1061. Its failure is retained. The full-history rerun with ordinary T1
commit matches every output across no-draft/MTP2 at 2048 greedy tokens for
English, Chinese and code, each cold/warm. Temperature 0.7 / seed 9950 also
passes 18 requests of 512 tokens across no-draft with weights retained, MTP2
and MTP fully unloaded. This supports the tested mode; it does not isolate a
particular faulty instruction or prove all speculative workloads stable.
Native disk-session identity binds the new arithmetic/commit choices and
refuses incompatible older files before payload restoration. Same-config
SAVE/RESTORE passes, and the engine stays usable after a refused restore.

`rtx5090-262k-no-mtp.json` fully unloads MTP as an alternative. All nine cache
requests have the same complete main states as MTP2, and the sampled gate passes.
Its one 8K screen measures 3100.40 prefill / 50.50 decode tok/s; this is not a
repeated speed comparison. Run it with `tools/run_safetensors.ps1 -Config
config/native/rtx5090-262k-no-mtp.json -Port 8880`. The preceding native desktop
is retained in `rtx5090-262k-gpu-reference.json` for rollback. CPU25/50/75 test
profiles inherit the current flags; only CPU75 has the complete combined
acceptance/performance evidence. Neither an existing user engine nor HTTP
service was started or modified by these private-pipe/CLI tests. Native GEMM
prewarm remains a separate opt-in without an end-to-end selection result.

These are correctness checks on the RTX 5090 / Ryzen 9950X3D / 96 GB machine.
Repeated throughput measurements are recorded separately below. Raw summaries are under
`bench/results/2026-10-10-safetensors-runtime/`.

- All 1,079 dense bindings, 4,947,698,560 values, were compared with the same
  checkpoint's fidelity GGUF and exact BF16 sidecar. Source bits and prescribed
  layout/norm transformations match, except 11 `-exp(A_log)` values differing
  by at most two FP32 ULPs between implementations.
- Full forward first-token diagnostics: all 48 router top-10 ID lists match.
  Layer 0 residual and expert output are bit-identical. Final 248,320 logits
  have maximum absolute difference 0.00377691, RMS 0.00058715 and cosine
  0.9999999425. Argmax and all 32 continuation tokens match. This does not claim
  full-model bit equality or equivalence to an unquantized BF16 checkpoint.
- A/B/A with conversation caching disabled versus enabled produces identical
  tokens and byte-identical committed main-model state. Prefix reuse was
  249 and 264 tokens for a 271-token continuation. Expert file reads remain zero.
- Original MTP FP8 GPU kernels against a CPU FP64 oracle, real experts 0 and
  511: relative L2 error 2.84e-7 over 10,240 outputs. CPU NVFP4 FP32 activation
  tests give relative L2 below 3.5e-7 with AVX-512 and 1.4e-6 with the generic
  fallback; one- and four-token grouping is bit-identical within each backend.
- Fixed disk session save/restore for a main model with no loaded MTP draft.
  GPU snapshot tests and CPU validation tests pass. An exhaustive 792-case
  visible-output boundary test covers accepted drafts, output caps and EOS.

The native MTP matrix now passes at draft depths 0, 1, 2 and 4: 68 completed
requests cover English, Chinese, code, arithmetic, EOS, seeded sampling, caps
1/2/3/7/9, A/B/A, disk restore and cancel/resume. Completed main-model states
match bit-for-bit across draft depths, including the QSA spare row and block
position; the deliberately interrupted request itself may end at different
window boundaries and is excluded, but its resumed continuation must match.
Raw logs and tokens are in `mtp-cache/`. CPU/GPU scheduling, adaptive swaps
and repeated comparative timings are recorded in the P5 runs below.

`p5-reference/` records the next completed gate, using 4,499 hot GPU experts,
16,384 context, FP16 KV, no MTP, FP32 decode activations and BF16x2/FP16 prefill.
The engine actually executed `tokens=8192 max_chunk=8192`; the full request also
includes seven assistant-header tokens. Cold prompt processing was 5,108.1 ms,
and replay reused 8,192 tokens, processing only the seven header tokens in
368.6 ms. This is one cold/warm long-prompt pair, not a repeated speed claim.
Three short prompt pairs had cold TTFT 1.64–1.95 s and warm TTFT 0.425–0.429 s.
All eight requests generated 128 tokens with zero expert file reads. Whole
process peak working set was 73,361,530,880 bytes; raw GPU clocks, memory and
power samples are retained. Windows total page-fault counts include soft faults
and are not presented as SSD reads.

The same private engine passed OpenAI nonstreaming/SSE parity, Anthropic text,
forced tool arguments and tool-result continuation, plus explicit image and
oversized-context rejection over real localhost HTTP. The earlier HTTP attempt
incorrectly left Anthropic thinking enabled while expecting a short answer;
the corrected test explicitly disables thinking. No engine change was needed.
These results establish the reference path. The subsequent FP4, mixed CPU/GPU,
adaptive, MTP timing and longer-context quality results follow below.

`p5-pinned/` repeats the same eight requests with full `cudaHostAllocMapped`
residency. All first-token logits and all 128-token continuations are bit-identical
to `p5-reference/`; HTTP checks and executed 8192 prefill also pass. This changes
allocation and data movement, not the expert representation or arithmetic. The
same requests are in `p5-reference/requests.json`; binary hashes and raw GPU
samples are included with each run. Default selection uses the repeated runs below.

The manual `native-registration-probe` target reproduces the allocation order
without any model files (it uses 63.282 GiB host memory and 20 GiB VRAM). On this
machine, registering a prefix of an already locked allocation succeeds at
16/24 GiB but fails at 32/40 GiB when a later GPU operation synchronizes, despite
the allocation calls returning success. Reversing the order and splitting the
whole registration into 16 GiB pieces also fail. Directly allocating the whole
arena with `cudaHostAllocMapped` passes, including device reads and copies.
These are observations on this driver, not a general CUDA size limit. Raw logs
and the original 40 GiB model-start failure are in `registration/`. The probe is
excluded from ordinary builds and CTest; run it with no inference engine alive.

`p5-pinned-combinations/` adds seven completed eight-request runs: adaptive hot
experts, 50%/100% CPU execution of cold experts, original MTP depths 1/2/4 and
three-term FP4 decode. Hot cache hits still run on the GPU in the CPU cases.
Adaptive exchange completed 18,165 promotions with 50,222,882,640 payload bytes;
this counts promoted weights, not measured physical PCIe traffic. All first
logits and continuations for the allocation/exchange-only path match the
reference exactly. The three short warm runs decoded 128 tokens in
2,572.3 / 2,314.9 / 2,157.9 ms. They benefit from both prefix reuse and learned
expert placement; these are not repeated measurements of a universal speed.
MTP depths 1/2/4 also match all eight 128-token reference continuations,
including after actual 8192-token prefill. Expert source reads remain zero.
FP4 decode has not shown an end-to-end advantage in these measurements.

`p5-long-context/` uses the original directory through `--model`, a balanced
expert cache without a profile file, 4,499 GPU slots, 32,768 context, full pinned
expert RAM, FP16 KV and a 4 GiB conversation cache. All 20 retrieval, follow-up
and A/B/A requests at 2,048 / 8,192 / 25,000 / 32,000 input tokens return the
expected code. Returning to the 32K A conversation reuses 32,025 tokens, reads
seven new tokens and has 0.458 s TTFT. The first 32K request takes 13.952 s of
prompt processing; the other 32K request automatically reuses a 16,384-token
common prefix and is **not** a full cold-prefill timing. Peak process working
set is 76,716,089,344 bytes. All expert source counters stay zero.
The extended HTTP checks preserve reasoning history (89 of 90 prior tokens
reused), accept the template's later system message, and pass the existing
OpenAI, Anthropic, tool, streaming and rejection checks. These small retrieval
tasks do not establish general long-context reasoning quality.

`p5-staging-compute/` retains the earlier FP4/CPU comparison with host staging.
Its adaptive case was deliberately interrupted during startup before the full
pinned experiments; the archive is marked `interrupted_between_cases`, not a
completed adaptive result. Each evidence archive includes a file hash manifest.
Hardware and firmware-reported memory settings are in `hardware.json`.

`p5-teacher/` contains 234 teacher-forced positions (68 English, 82 Chinese,
84 code), using 6,999 hot slots, 4,096 context and 256-token prefill. Each target
is scored against the full 248,320 logits. The same-checkpoint fidelity backend
uses fully pinned resident experts and performs no CPU Q8 fallback. Relative
to native FP32 activations, its average KL is 2.5864e-6, top-1 agreement is
234/234, and mean NLL difference is -0.00028689 nats/token. Three-term FP4 gives
KL 0.00030822, agreement 233/234 and NLL difference -0.00002183; running all cold
experts on CPU gives KL 5.3595e-10, agreement 234/234 and NLL difference
-0.00000154. Native reference mean NLL is 1.18476296. These authored, small
samples check numerical and integration behavior; they are not a broad model
benchmark or a comparison with unquantized BF16 weights.

`p5-prefill/` compares prefill arithmetic with full pinned residency and the
same eight requests. For the actual 8192-token segment plus assistant header,
prompt processing is 4,966.7 ms for BF16x2/FP16, 4,237.4 ms for W4A8,
3,074.9 ms for two-term FP4 and 2,917.9 ms for single-term FP4. Relative first
logit KL on that long request is respectively 0 / 0.0012513 / 0.0020589 /
0.0020638. W4A8 preserves this 128-token continuation; both FP4 prefill choices
change it. These are single cold/warm pairs, so faster candidates stay opt-in
while the quality configuration retains BF16x2/FP16.

`p5-fp4-numerical/` records rebuilt SM120 diagnostics on real checkpoint
weights. For layer 0 expert 0, the one/two/three-term GPU results match the
independent activation quantizer and FP64 product reference (gate/up relative
error at most 5.31e-7 and down at most 1.48e-7); uneven groups and permuted
destinations also pass. Error versus FP32 activations is a separate quantity:
13.82% / 1.36% / 0.116% on these synthetic activations. Grouped prefill tests
cover 508 rows across 20 experts, including partial groups, and all three
candidate paths pass their numerical checks. Trace messages are emitted from
the actual FP4 dispatch wrappers; SASS evidence is retained separately. These
kernel diagnostics read the already-validated fidelity pack as an oracle;
the inference runtime still loads only the original safetensors directory.

`p5-adaptive-mtp/` combines mapped RAM and hot promotions with draft depths
1/2/4 and, separately, 50% CPU execution of cold experts. All four cases match
all eight reference continuations and keep expert file reads at zero. MTP
acceptance totals are 474/542, 636/760 and 746/1,080 offered tokens. Each case
also performs real promotions (11,827 / 8,973 / 6,624 for MTP, 18,163 for the
CPU split). Native MTP now enforces the full checkpoint vocabulary in code,
independently of the diagnostic environment variable or any stray subset file.
These results still use fixed 4,499-slot caches; automatic-cache and repeated
comparisons are recorded separately.

## Repeated comparison on this machine

`p5-three-rounds/` contains three fresh processes per backend, each with a
128-token unmeasured warmup followed by the same eight 128-token requests.
All 72 measured requests pass, with zero expert file reads and no CPU expert
execution. Each backend uses 32,768 context, 8,192 prefill with borrowing,
FP16 KV, automatic expert cache sizing, 2,048 MiB VRAM reserve, adaptive
promotions every four rounds and MTP disabled. The comparison uses a 2 GiB,
two-slot conversation cache; the separately tested operator profiles use 4 GiB
and four slots. Request IDs, commands, raw logs, telemetry and min/median/max
summaries are archived. No slow samples are removed.

The table reports medians across the three processes. Decode rates exclude
the first output token. A warm prompt reuses its prefix and benefits from prior
expert placement. A cold prompt is a new prefix in an already loaded, warmed
process, not an empty OS disk cache. Startup is excluded from these timings.

| Measurement | Native safetensors | Fidelity NVFP4 | Existing Q4XL |
| --- | ---: | ---: | ---: |
| Short warm 0: total seconds / decode tok/s | 2.030 / 67.22 | 2.065 / 65.59 | 1.666 / 82.74 |
| Short warm 1: total seconds / decode tok/s | 1.907 / 71.82 | 1.907 / 71.09 | 1.563 / 88.10 |
| Short warm 2: total seconds / decode tok/s | 1.905 / 71.73 | 1.898 / 71.28 | 1.624 / 83.96 |
| 8192 segment + header: prompt processing seconds | 5.056 | 3.777 | 2.753 |
| Long cold: total seconds / decode tok/s | 7.563 / 50.80 | 6.553 / 46.43 | 5.147 / 53.73 |
| Long warm: total seconds / decode tok/s | 2.072 / 67.66 | 2.072 / 66.58 | 1.736 / 80.71 |
| Peak process working set GiB | 67.95 | 51.50 | 84.61 |
| Whole-device peak used VRAM MiB | 31,330 | 31,135 | 31,027 |

This is a deployment comparison, not isolation of one kernel. Q4XL uses
different quantized weights and has more GPU expert slots because its dense
weights occupy less VRAM. Native retains RAM copies of all experts, including
GPU residents; legacy keeps the resident complement. Native enables FP32 GDN
activations, while the inherited fidelity preset does not. Native ngram data
resides on D: (BIWIN X570 PRO), legacy ngram data on G: (Fanxiang S790); both are
NVMe drives. These differences are retained in the commands and hardware
record. The slower native cold prefill remains an optimization opportunity;
these measurements do not establish a single cause.

One native short-cold sample takes 5.333 seconds instead of approximately 2.95
seconds. Its ngram counters record an additional 2.481 seconds of blocked wait,
with a reported p99 read latency of 80,763 microseconds. Expert source reads
remain zero, and the warm replay returns to approximately 1.9 seconds. The
sample remains in the ranges. This observation does not identify a hardware
fault or measure physical expert SSD traffic.

Recorded measured-request SM clock ranges are 2707-2790 MHz native,
2692-2760 MHz fidelity and 2715-2790 MHz Q4XL; all report 13,801 MHz memory
clock. GPU telemetry includes the desktop. Hardware is RTX 5090 with driver
617.14, Ryzen 9950X3D and 96 GiB RAM. DIMM firmware reports 6600 MT/s; that is
not a measured memory clock. See `hardware.json` and each `summary.json`.

## Native MTP and selected operator defaults

`p5-mtp2-three-rounds/` repeats the native comparison in three fresh processes
with two draft tokens. All 24 continuations are exactly equal to the matching
no-MTP continuations, and all expert source reads remain zero. The main model
still verifies every draft token; FP32 decode activations and BF16x2/FP16
prefill are unchanged. The draft consumes VRAM: automatic cache sizes are
6,119 / 6,177 / 6,111 experts instead of the no-MTP 7,143 / 7,154 / 7,161.

| Request | No MTP median total seconds | MTP2 median total seconds | MTP2 min-max seconds |
| --- | ---: | ---: | ---: |
| Short 0 cold | 3.014 | 2.468 | 2.385-2.500 |
| Short 0 warm | 2.030 | 1.564 | 1.537-1.632 |
| Short 1 cold | 2.949 | 2.317 | 2.310-2.332 |
| Short 1 warm | 1.907 | 1.460 | 1.445-1.554 |
| Short 2 cold | 3.428 | 2.370 | 2.325-2.613 |
| Short 2 warm | 1.905 | 1.419 | 1.407-1.515 |
| Long cold | 7.563 | 6.607 | 6.513-11.563 |
| Long warm | 2.072 | 1.669 | 1.662-1.693 |

MTP2 reduces total-time medians by 12.6-30.9% across these eight requests.
Short warm decode medians are 88.57 / 95.65 / 98.90 tok/s; these exclude TTFT
and are not end-to-end rates. One long cold run spends 9.556 seconds processing
the prompt, versus 4.516 and 4.590 seconds in the other runs. Its ngram blocked
counter increases by only 0.215 seconds; the full delay is not explained by
that counter. The MTP runs include low-clock GPU samples (540 MHz SM and
7001 MHz memory minimum); their causal relation to the outlier is unproven.
The sample remains in the report. Startup takes 93.74-107.52 seconds, median
99.41; peak process working set is 68.00 GiB and whole-device VRAM 31,326 MiB.

`START-NATIVE.bat` and newly generated configs therefore select MTP2 for this
machine. `--mtp 0` generates the no-MTP reference. This decision uses measured
request latency, not MTP acceptance or FP4 kernel throughput alone. Both
operator configs were also tested without benchmark overrides:
`p5-operator-auto/` and `p5-operator-auto-mtp2/` pass actual 8192 prefill, all
eight continuation comparisons and real HTTP acceptance, including native
image rejection. Their conversation cache is 4 GiB with four slots.

The completed scope is Windows single-GPU SM120 text inference from this
original checkpoint, tested through 32K context. HIP and SYCL builds are not
validated. Sparse MTP-indexer selection and multi-GPU remain outside
this implementation; unsupported native modes fail explicitly. The small
quality suite establishes migration fidelity and cache/MTP consistency, not
general model quality or equivalence to an unquantized BF16 model. Improving
native cold prefill and explaining latency tails remain optimization work.

## Startup loading

The original native loader reads and packs 24,576 experts one at a time and
copies each temporary blob into the final 63.282 GiB locked arena. A GGUF/pack
has already arranged those bytes before startup; native safetensors adapts them
in RAM on each start. Reading the original files, physically locking the arena,
packing and GPU setup all contribute; the existing logs did not time them
separately, so they do not establish which accounts for the full startup gap.

The opt-in startup loader now sorts a bounded batch's reads through the existing
source, packs directly into disjoint final arena spans and reuses host scratch
between batches. Source I/O stays serial; CPU workers only permute bytes and
retain FP32 scales. No conversion cache is written, no ngram data is pulled into
the batch, and the existing expert/MTP source-read barrier remains in force.
`STRATA_NATIVE_LOAD_BATCH=1` keeps the original loader. Batch sizes 2..128 use
the new path; `STRATA_NATIVE_LOAD_WORKERS=1..16` sets CPU packing workers.
The desktop profile opts into batch 64 / four workers. Its raw scratch is
168.75 MiB plus the source's at-most-8-MiB coalescing buffer and small metadata;
scratch is freed when expert startup finishes, before inference.

On 2026-10-10, a CPU-only sample on the Ryzen 9950X3D / 96 GB Windows machine
read 64 layer-0 experts from the original checkpoint on D: (BIWIN X570 PRO).
After warming the sample, three rounds alternating mode order measured medians
84.8902 ms for the legacy loader and 34.7159 ms for batch 64 / four workers
with reusable scratch. The source's read calls fell from 192 to 141 per mode
invocation, with the same 176,948,736 expert bytes and zero ngram bytes. All
destination bytes and input scales match in every round; reads after sealing
are refused. The test ran alongside the user's service, allocated no GPU
context and did not restart it. These are warm bounded-sample results, not
cold SSD throughput or a full-engine startup speedup. Raw evidence is in
`bench/results/2026-10-10-safetensors-runtime/p6-startup-sample/`.

The five CPU CTest checks also pass with new tests for batch/serial equality,
signed FP4 zero, tiny FP32 scales, destination canaries, worker counts 1/4/16,
overlap refusal and worker-error propagation. The staged CUDA executable
builds with the unchanged GPU kernels; HIP/SYCL remain unbuilt here.
The desktop executable is now `strata-startup.exe`, retaining the previous
`strata-int8.exe` for comparison. A user restart is required. Full cold startup
has not been measured with the new loader. Logs now time arena allocation,
reading, packing and total engine startup; `INFO startup_ms` measures main-entry
to ready, excluding Python setup and process/DLL loading. Read/pack values of
-1 on the reference loader mean unmeasured. All-expert residency is still
required before ready; the optimization does not trade startup time for expert
SSD reads during inference.

```powershell
.\tools\build_safetensors.ps1
.\build-safetensors\safetensors_batch_bench.exe D:\迅雷下载\Qwen3.8-flash-next-nvfp4 0 64 4
```

## Build the runtime

The desktop `Start-Strata-Safetensors.bat` calls this checkout's
`START-NATIVE-262K.bat`, selecting `config/native/rtx5090-262k-mtp2.json` and
`build-native-engine/strata-efficiency12.exe`. Settings are 262144 total context,
8192 dedicated prefill, INT8 KV, original BF16 MTP2, CPU75 cold decode experts,
kernel copying, elastic KV, canonical FP32 arithmetic and ordinary T1 commit.
FP32 decode activations, BF16x2 dense prefill, W4A8 expert prefill and closed
startup handles remain enabled. It serves `127.0.0.1:8880` in the foreground;
Ctrl+C stops it. Wildcard Host accepts any HTTP Host header. Restarting loads
the new profile; tests do not start the public server. Logs append to
`logs/native-262k-int8.log`, whose directory the launcher creates.

Build this exact binary with:

```powershell
.\tools\build_safetensors_engine.ps1 -Jobs 2 -OutputName strata-efficiency12 -Targets @('strata')
```

The preceding text-only efficiency11 acceptance evidence includes near-capacity
262000-token input and
full 262144-cell KV mapping, warm reuse, A/B/A, disk restore, sampled/greedy
stability and three alternating final-binary speed pairs. The original main
model and deployment remain separate. MTP4/Q8 were rolled back at the user's
request; the desktop keeps original BF16 projections and BF16 MTP batching off.
The fully unloaded MTP alternative is `rtx5090-262k-no-mtp.json`; CPU25/50 full
combined throughput is unmeasured. CUDA SM120 builds and runs successfully;
HIP/SYCL have not been built or validated on this machine.

### Optional Q8 MTP projections and chunk diagnostics

The optional `rtx5090-262k-mtp4.json` comparison profile selects `--spec 5 --mtp-max-t 5`
(one verifier token plus four drafts) and
`STRATA_MTP_NATIVE_PROJECTIONS=q8_0`. With this variable unset or `bf16`, native
MTP keeps the original BF16 projection path. Invalid values are rejected at load.
The config generator exposes `--mtp 4 --mtp-projections q8_0` and keeps BF16 as
its default. `rtx5090-262k-mtp2.json` is again the desktop profile.

The adapter quantizes the same ten matrices as the existing Q8 MTP pack:
`fc_embedding`, `fc_hidden`, attention Q/K/V/O, indexer `index_qk_proj` and the
shared expert's gate/up/down projections. It streams BF16 rows directly from
safetensors through ggml's Q8_0 reference quantizer at load, without writing a
converted model. Norms remain FP32, HC mixers/router remain BF16, and all 512 MTP
experts keep their original FP8 codes and block scales. The original main-model
NVFP4 weights and output head are unchanged. Projection quantization is lossy;
draft acceptance, output parity and end-to-end speed require separate GPU checks.

The inherited `Prefill::draft_kv` batch path required Q8 front projections. The
BF16 native profile therefore fell back to small token groups after each main
prefill chunk. Q8 makes that batch path eligible on this single-GPU profile;
runtime scratch/KV conditions can still cause fallback. The CUDA build includes
the existing Q8 MMQ kernels. `STRATA_MTP_BATCH=0` retains the token-group path for
comparison. Q8 loading alone does not establish that the batch path ran or that
it improved speed.

An additional opt-in path, `STRATA_MTP_BATCH_BF16=1`, now accepts the four original
BF16 front projections directly. This is a code-interface extension, not a GPU
format restriction. It reuses `Gemm::bf16_f32` (three BF16 components of each FP32
activation, FP32 accumulated output), the existing batched norms, pinned HC
postops and KV appends. HC down/up also retain FP32 intermediate activations.
The existing scratch is reused; no persistent matrix copy or extra workspace is
allocated. GEMM accumulation order differs from per-token execution, so this
is not a byte-identity claim. The engine default remains the old BF16 fallback
until GPU parity/acceptance and timing are measured. Generate a comparison
profile with `--mtp 4 --mtp-batch-bf16`; the local
`rtx5090-262k-mtp4-bf16.json` has the same desktop settings with BF16 projections
and this batch opt-in. Neither MTP4 profile is selected by the desktop launcher.

On 2026-10-10, the user's RTX 5090 / 9950X3D / 96 GB service with INT8 KV and BF16
MTP reported 24,332 fresh input tokens in 22,577 ms (1,077.7 tok/s), below the
2,000-2,500 target. This is an interactive observation, not a controlled A/B.
The later Q8/MTP4 interactive run reported 24,332 fresh tokens in 21,580 ms
(1,127.5 tok/s), and 193 generated tokens in 5,938 ms (32.5 tok/s).
These are different interactive requests, not a controlled A/B. Chunk tracing reports
main compute plus waits, MTP callback path/time and conversation-checkpoint time.
The desktop profile also enables the existing `STRATA_PLE_TRACE` read/wait trace
to distinguish ngram preparation waits from MTP and checkpoint pauses. Cumulative ngram
blocked time includes decode and earlier requests and cannot be assigned to one
prefill. No second engine was started during the user's active session.

CPU tests cover row slices, tile boundaries, exact scale/code/tie rounding,
zero blocks, canaries and rejection of nonfinite inputs/scales and invalid
bindings. A read-only pass over all ten real matrices converted 138,936,320 BF16
bytes to 73,809,920 Q8 bytes (62.1 MiB less), with no main-expert or ngram reads.
The CUDA engine builds for SM120. The user ran the Q8/four-draft profile, but a
controlled quality/cache-restore comparison of that combination remains pending.
HIP/SYCL were not built.

### Runtime diagnosis after the MTP4/Q8 rollback

Evidence is in `bench/results/2026-10-10-safetensors-runtime/p8-diagnosis/`.
On this RTX 5090 / 9950X3D / 96 GB Windows machine, the four long-request chunks
spent 13,537 ms waiting for the async PLE gather to finish, versus 32.875 ms in
MTP callbacks and 24.375 ms in conversation checkpoints. The gather wait includes
row reads, row-cache operations, FP8 decoding and scheduling. It is not a disk
service-time measurement. The cumulative `blocked` counter also includes batch
cache/decode processing, decode requests and earlier requests.

A CPU-only, idle comparison read the same 131,072 row IDs from the original D:
safetensors shard and the existing G: FP8 GGUF sidecar. Eight readers, 256 in-flight
requests, zero row cache and three alternating rounds gave D: 335.627 / 119.720 /
114.476 ms and G: 842.713 / 984.212 / 914.799 ms. All requested bytes matched.
These numbers exclude FP8 decoding and model activity; controller caches were
not flushed. They give no basis for moving the default table to G:. The native
path continues to read the original table, with no new sidecar or mirror.

With `STRATA_PLE_TRACE=1`, `strata ple batch phases` now reports call-lock wait,
cache lookup, page planning, group wait, cache insertion, callback decoding and
reader drain separately. Those are caller wall-time phases: reader work overlaps
callbacks, and group wait includes thread scheduling. `strata ple gather phases`
adds raw-buffer resize time and available physical host RAM on Windows (0 means
unavailable on other platforms). Default numerical paths are unchanged.

The native adapter also inherited a legacy guard that disables elastic KV for
`resident_cpu_experts`. Legacy resident mode keeps a complement of GPU experts;
native safetensors keeps all experts, including hot-slot RAM mirrors, so it can
use the existing VMM grow/trim scheduler. `STRATA_NATIVE_KV_GROW=1` opts into that
integration. The separate `rtx5090-262k-mtp2-elastic-test.json` profile enables it
and residency auditing; the desktop does not. The usual `STRATA_KV_GROW=0` still
disables it. Context capacity remains 262,144; physical KV pages grow as needed.
This can leave more room for hot experts at short contexts, but no native
end-to-end speed improvement is claimed before long-context/cache-restore tests.
The existing VMM map/unmap/transfer unit test passed on SM120; it is not an
acceptance test of the model, adaptive swaps, conversations or MTP.

The observed profile allocated 4,585,892,608 bytes of dedicated prefill workspace
and kept approximately 3,197 GPU expert slots. INT8 KV for the full 262K context
is also allocated up front when native elastic KV is off. In the long request,
55.6% of expert routes were outside the hot GPU cache, with zero expert-file
bytes. Original BF16 dense/head storage, dedicated workspace and KV allocation
must be considered alongside kernel timing when comparing against converted
GGUF deployments. A container-format-only explanation is not established.

The native loader previously replaced every explicit `--kv` with FP16. It now
honours `--kv int8` and keeps FP16 when unspecified. Native Q4/hybrid KV are
rejected rather than silently replaced. INT8 reuses the existing rotated-KV
prefill, decode, MTP and conversation-cache paths; FP8 KV is not implemented.
`--no-prefill-borrow` reserves separate buffers before automatic expert cache
sizing, so prefill does not evict and refill the expert-cache tail for workspace.
The conservative existing reservation for 8,192 rows is 5,600 MiB, in addition
to the 2,048 MiB reserve. It is an allocation budget, not a measured workspace
size. Startup traces report actual allocated bytes and effective chunk capacity;
`INFO` also exposes `prefill_chunk` and `prefill_workspace`. The existing fallback
can reduce the chunk if other VRAM use prevents it fitting; check these fields
before claiming an 8,192-row run.

The pre-change user-service metrics recorded a cold 24,115-token request at
27,792.4 ms (867.7 tok/s), with zero expert source reads. This is evidence of the
reported slowdown, not a benchmark of the new profile. Earlier W4A8 results
above used FP16 KV and different context/cache conditions; they do not establish
the speed or quality of this combined INT8 profile. Repeat cold prefill, cached
follow-up, cache restore, MTP parity and expert-source-I/O checks after switching.

On this machine, `START-NATIVE.bat` starts the independent MTP2 quality configuration
at `127.0.0.1:8097`. It uses this checkout's Python environment and engine;
the original model and other deployments are unchanged. The server runs in the
foreground and Ctrl+C stops it. This older 32K startup profile accepts text only;
the updated 262K desktop profile above enables the external image encoder.

`config/native/rtx5090-quality.json` selects 32,768 context, 8,192 prefill,
FP16 KV, full mapped expert RAM, automatic VRAM cache sizing with 2,048 MiB
reserve, the bundled initial expert profile, 96 promotions every four rounds,
and a 4 GiB conversation cache. MTP stays off in this reference config;
`config/native/rtx5090-mtp2.json` is the startup profile with native MTP2 enabled.
`--staging`, `--adapt-every 0`, `--balanced-experts`, `--pcie-frac 0|0.5|1` and
`--mtp 0|1|2|4` on the generator expose the measured alternatives explicitly.
`--kv fp16|int8`, `--prefill-mode fp16|w4a8|w4a4x2|w4a4`,
`--dedicated-prefill` and `--exe` select precision, workspace ownership and a
staged executable. Generator defaults retain the FP16 reference settings.
`--load-batch 64 --load-workers 4` selects the measured startup sample's loader;
both generator defaults are 1, retaining the reference startup path.
CPU fractions apply only to cold experts. The smaller reference diagnostic
config remains in `config/native/reference.json`.

`p5-operator-auto/` tests these quality settings without benchmark overrides:
7,102 actual GPU slots, 1,670 temporarily borrowed for prefill, all eight
128-token continuations identical to the reference, actual 8192 prefill,
HTTP acceptance and zero expert source reads. Slot counts depend on other VRAM
use at startup and are not hardcoded. This run's peak working set is
73,460,129,792 bytes; startup is 90.48 seconds with in-memory repacking.

```powershell
cd G:\Strata\Strata-Safetensors
.\tools\build_safetensors_engine.ps1
.\.venv-native\Scripts\python.exe tools\native_config.py `
  --model D:\迅雷下载\Qwen3.8-flash-next-nvfp4 --output build-native-engine\native.json
.\.venv-native\Scripts\python.exe -m serve.server --engine strata `
  --config build-native-engine\native.json --host 127.0.0.1 --port 8097
```

To rebuild the desktop profile without overwriting the reference executable:

```powershell
.\tools\build_safetensors_engine.ps1 -Jobs 2 -OutputName strata-startup
```

The build script refuses to overwrite a running engine with the selected name.
It configures the existing build directory and reuses unchanged CUDA libraries.
On 2026-10-10 the earlier INT8 staged SM120 CUDA build succeeded. The five CPU safetensors
CTest checks and 16 offline conversation-cache harness tests passed. Explicit
INT8/default FP16 config generation, overwrite refusal, desktop profile and
port settings, PowerShell parsing and `--help` with GPUs hidden passed. The
reference `strata.exe` hash remains
`841835c525a2598e8d9f7d12f4bc85a264e04cb7ff5df0ab12c7de9780fa7a53`;
the staged `strata-int8.exe` hash is
`9cb530d67529e076013e58d9ff89184f417c0432eaa1c9f2222edef4ebf910d6`.
These checks do not validate INT8 GPU inference. No GPU kernel sources changed;
HIP and SYCL builds remain untested on this Windows CUDA setup.

The config generator refuses writes inside the model directory and refuses to
overwrite an existing config. Its CUDA DLL paths refer to the installed toolkit;
`--cuda-bin` overrides that location. The server uses a private port and its own
child engine. Install the native Python requirements into this checkout's own
virtual environment. Do not invoke the inherited installer to set up this path.

## Historical P0/P1 record

On 2026-10-10 the user retired the local fidelity GGUF directory, its
`Strata-data/packs/nvidia-nvfp4-fidelity` pack, the `Strata-NVFP4-Fusion`
checkout and the old NVFP4 desktop launcher, freeing approximately 156.5 GiB
of file bytes. Original safetensors, Q4XL, shared ngram/MTP files and this
independent checkout remain. The native repository was fetched completely
from GitHub and passed Git connectivity checks before the old checkout was
removed. Archived comparison results remain available; rerunning diagnostics
that name those retired GGUF/pack oracles requires rebuilding the oracles.

The remaining sections document the first committed adapter milestone. Their
"not yet" statements apply to that milestone only; current runtime status is above.

This independent checkout starts at fidelity commit
`d167eb89301a02e0d6299f49d35494ba5096bc10`, on branch
`codex/safetensors-native`. It adds a CPU-only C++20 library and diagnostic CLI.
It does not yet start the full model with `strata --model <HF directory>`.

The development target is Windows, RTX 5090 32 GB, Ryzen 9950X3D and 96 GB RAM.
The original model is read-only at
`D:/迅雷下载/Qwen3.8-flash-next-nvfp4`. The main deployment at
`G:/Strata/Strata` and the verified checkout at
`G:/Strata/Strata-NVFP4-Fusion` are left in place. No service is started, stopped
or replaced by the new build or validation scripts. No model is downloaded.

## What is implemented

- `SafetensorsSource` cross-checks the shard index and every actual file header.
  It rejects duplicate JSON keys, malformed UTF-8, excessive nesting, unsupported
  dtypes, invalid ranks, negative/nonintegral sizes, multiplication overflow,
  truncated files, overlapping ranges, holes, unindexed trailing bytes, missing
  tensors and unindexed shards. Header and metadata JSON limits are 100 MB each;
  rank is limited to 16. Only basename shard references are accepted. Absolute
  paths, separators, Windows ADS syntax and resolved paths outside the root are
  rejected. Windows opens files read-only with wide paths and denied writes.
- `WeightSource` separates source descriptions from consumers. Each descriptor
  carries source file, absolute offset, byte count, dtype, physical shape and
  family. Descriptors live for the source's lifetime; output buffers belong to
  the caller. Reads are synchronous and serial. The source keeps file handles,
  not a second copy of the model or a mapped whole-file cache.
- Batched requests are sorted by file and offset. Adjacent ranges in the same
  family are coalesced with at most 8 MiB of staging. Larger contiguous requests
  go directly to the destination in bounded chunks. Gaps are not read merely to
  reduce the number of calls. Dense, expert, ngram, MTP and vision read bytes have
  separate counters. These count requested source bytes, **not physical SSD or
  OS page-cache activity**.
- `Qwen4Config` rejects unsupported architecture and geometry. The binding plan
  validates all 24,576 routed experts / 73,728 four-part projections in this
  checkpoint and the 128 PLE segments. It records 36 linear-attention and 12
  full-attention layers and the config-to-weight PLE layer numbering.
- `Nvfp4Projection` owns references to weight, micro scale, FP32 global weight
  scale and FP32 input scale together. The logical shape is `[output,input]`,
  separately from packed physical shape. Expert canonical names are centralized
  in `qwen4.cpp`. The adapter keeps all three input scales, including unequal
  gate/up scales; it does not silently select one for a shared-scale kernel.
- The expert adapter rearranges bytes into the existing `NativeExpertLayout`:
  gate, up, down, then `{s_gate,s_up,s_down,0}`. Adjacent HF FP4 nibbles become
  ggml's 16-value sub-block order; four E4M3 bytes accompany each 64 values.
  No weight is requantized, transposed or rounded. Negative/NaN micro scales are
  refused because the existing UE4M3 kernels cannot preserve their meaning.
  Global/input scales must be positive finite FP32. Zero micro scales and signed
  FP4 zero are preserved. The input scales remain FP32, outside the legacy blob.
- The PLE plan uses a logical-row segment table rather than assuming physical
  adjacency. It reads the saved head offsets, vocabulary sizes and BF16 global
  scale. Row reads use the real tensor byte offsets, including a row crossing a
  4 KiB boundary. There is no asynchronous IOCP or row cache yet.
- `seal_expert_reads()` prohibits subsequent expert reads. Tests exercise this
  guard and reuse a small owned sample without source reads. This is not yet a
  full model RAM-residency, swap, paging or zero-expert-SSD acceptance result.

The adapter currently loads one expert at a time with bounded raw and packed
buffers. A full arena loader must scatter by shard order before it is used for
startup of all experts; looping this sample adapter over the full model is not
the planned production loading strategy.

## Build and run

The standalone build does not enable or alter the inherited CUDA/HIP/SYCL build.
It requires MSVC 2022, CMake, Ninja and Python for the adversarial tests. The local
script defaults to the existing tools, without modifying that environment:

```powershell
cd G:\Strata\Strata-Safetensors
.\tools\build_safetensors.ps1 -Clean
.\build-safetensors\strata-safetensors.exe inspect D:\迅雷下载\Qwen3.8-flash-next-nvfp4
```

`headers MODEL` only validates the container and index. `inspect MODEL` additionally
validates the targeted model plan and reads 258 bytes of PLE constants; it does not
read expert payloads. `verify MODEL LAYER EXPERT` verifies one expert's complete
byte roundtrip. `verify-set MODEL` checks six fixed experts and eight PLE rows.
An optional output file/directory writes small diagnostic expert blobs only;
none of these files is a required model format. Output under the original model
directory and overwriting an existing test blob are refused.

The equivalent CMake entry point is `cmake -S native_safetensors -B build-safetensors`
with the compiler environment configured. `STRATA_SAFETENSORS_TRACE=1` prints
shard progress to stderr. Structured results go to stdout. This build corrects
the specific double-decoded Chinese MSVC include prefix observed locally, so
Ninja tracks header dependencies; it leaves other compiler prefixes alone.

For the independent real-checkpoint test, NumPy is used only by the test oracle:

```powershell
G:\Strata\Strata\.venv\Scripts\python.exe tools\validate_safetensors_checkpoint.py `
  --model D:\迅雷下载\Qwen3.8-flash-next-nvfp4 `
  --exe build-safetensors\strata-safetensors.exe `
  --output build-safetensors\my-new-validation-run `
  --oracle-pack G:\Strata\Strata-data\packs\nvidia-nvfp4-fidelity\experts.bin `
  --gpu-parity G:\Strata\Strata-NVFP4-Fusion\build-fidelity\nvfp4_expert_gpu_parity.exe `
  --cuda-bin G:\Strata\Strata\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64
```

The output directory must be new. `--oracle-pack` and `--gpu-parity` are optional
test oracles. The CLI/library never needs the old pack. The GPU oracle is the
previously verified executable from the baseline, with `STRATA_NVFP4_F32=1` and
`STRATA_NVFP4_TC=0`; it runs the existing high-precision expert kernel, not FP4 MMA.
Each exported blob is passed to its single-blob test as index `0/0`, so its printed
layer/expert label is local to that test file. The report records the actual IDs.

## Validation and its limits

Evidence is under `bench/results/2026-10-09-safetensors-p1/`. On the machine above,
Windows, MSVC 19.44.35228 and NVIDIA driver 617.14:

- All 299,545 tensor headers in 11 files match the original index. Payload sizes
  match dtype and shape. Header SHA-256 values are recorded; these are not hashes
  of all 123.568 GiB of source payloads.
- The six complete experts are `0/0`, `0/7`, `12/100`, `24/300`, `40/5`, `47/511`.
  Every packed blob is byte-identical to its previous fidelity-pack entry. An
  independent NumPy FP64 decoder gives exactly equal weights, including zero
  signs, and equal three-row SwiGLU outputs. The existing CUDA reference test
  uses eight activation rows per expert and reports at most 0.000027% relative
  output error against its FP64 reference.
- Synthetic layout tests cover all finite nonnegative micro-scale codes, signed
  FP4 zero, FP32 scales too small for FP16, unequal activation scales, invalid
  scales and wrong matrix direction. Source tests cover coalescing, ownership,
  family counters, prevalidation of an entire read batch, and the read guard.
  Python fixtures cover malformed headers/indexes, Unicode paths, a sparse file
  read beyond 4 GiB and invalid read ranges. The file-symlink escape test is
  skipped when Windows does not grant symlink creation; that limitation is
  recorded in the test log rather than counted as a pass.
- Eight real PLE rows are independently checked by source offsets, including
  logical segment transitions, a 4 KiB crossing and the final row.
- The planned expert blob arena is 67,948,118,016 bytes (63.281616 GiB), plus
  294,912 bytes of input scales. This is a **plan**, not committed or locked RAM.
  No full-arena allocation or GPU cache sizing was performed.

No end-to-end text quality, session cache, all-expert residency, CPU/GPU scheduling,
MTP, 8192-token executed prefill, FP4 Tensor Core dispatch or throughput is claimed
by these tests. No comparison against BF16 model quality is implied by matching
the same NVFP4 checkpoint. Only metadata and selected expert/PLE payloads have
been read. The inherited forward implementation, kernels and runtime defaults
have no changes in this milestone; Git diff against the baseline verifies that.

## Next implementation gates

1. P2: bind ordinary BF16/F32 weights, embedding, output head and model constants
   into the existing forward. Centralize concat/transpose/norm transformations;
   preserve source bits unless a recorded model transformation requires arithmetic.
   Add layer hidden-state, router top-k, expert output and teacher-forced logits
   comparisons to the old fidelity backend, with MTP and complex caches disabled.
2. P3: load all experts once by shard order into stable owned RAM; add a physical
   residency policy, budget checks and OS paging observations. Connect hot GPU
   slots and CPU/GPU cost decisions without disk fallback. Add async segmented
   PLE reads and bounded row caching, with independent I/O metrics.
3. P4: use the same checkpoint's MTP/FP8 weights. Validate KV, GDN, convolution,
   PLE, HC, QSA/indexer and token boundaries on cache restore and speculation
   commit/rollback, including EOS, cancel and interleaved sessions. Preserve the
   supplied froggeric v22.5 template and reasoning history.
4. P5: measure SM120 FP4 paths and effective 8192 prefill only after those gates.
   Run same-condition, at-least-three-run end-to-end comparisons with the old
   NVFP4 reference and Q4XL; choose defaults from quality and total session time.

## Provenance

Strata, this branch and the reused kernels retain their original MIT notices.
The layout contract follows the existing `tools/nvfp4_codec.py`,
`src/kernels/cuda/iq_kernels.cu` and ggml `block_nvfp4`; no claim of original
authorship is made for those parts. ggml notices remain in `third_party/ggml/`.
The new library vendors nlohmann/json 3.12.0 with its MIT license and SHA-256 in
`third_party/nlohmann/README.md`. The original safetensors format specification is
https://github.com/huggingface/safetensors#format. No CUTLASS code was added in P1;
the inherited dependency notices still apply when its kernels are built.
Model licensing is separate from engine licensing. No model weights are tracked.

The implementation was guided by the complete local handoff:
`C:/Users/spideytznn/Documents/Codex/2026-10-09/d-qwen3-8-flash-next-nvfp4/outputs/Strata-Safetensors-项目交接.txt`.
Open safetensors issues/PRs were searched before work; upstream issue 1387 requests
higher-precision support on larger hardware, but no matching native-loader PR was
found by those searches. No issue or PR was posted by this milestone.
