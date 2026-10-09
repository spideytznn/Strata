# Native safetensors backend

## Current runtime

`strata --model <original HF directory>` now runs text inference directly from
NVIDIA safetensors. `--safetensors` is an alias. The independent branch remains
`codex/safetensors-native`, based on `d167eb89301a02e0d6299f49d35494ba5096bc10`.
The original checkpoint and existing deployments are read-only inputs.

The loader validates the fixed Qwen3.8-Flash-Next geometry and binds the existing
Strata forward, scheduler, cache and prefill code. It preserves NVFP4 weight
codes and scales, ordinary BF16 weights, FP8 ngram data and original FP8 MTP
experts. There is no required GGUF, expert pack or converted MTP artifact.
A 63.282 GiB owned expert arena is physically locked in system RAM, including
copies of experts currently in VRAM. Failure to lock the whole arena aborts
startup. The initial path uses a separate 337.5 MiB pinned staging arena.
`STRATA_NATIVE_ALLOC_PINNED=1` instead allocates the whole arena with
`cudaHostAllocMapped`, eliminating that host staging copy. It remains explicit
while end-to-end default selection is in progress. Model expert and MTP source reads are sealed after loading;
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
vision, cross-model drafts and MTP requantization are refused by this native path.
The HF tokenizer and all added tokens are used directly. The supplied froggeric
v22.5 template is retained in `config/native/froggeric-v22.5.jinja`.

## Runtime evidence recorded so far

These are correctness checks on the RTX 5090 / Ryzen 9950X3D / 96 GB machine,
not a completed throughput comparison. Raw summaries are under
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
and repeated comparative timings are still being validated.

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
These results establish the reference path only. The FP4, mixed CPU/GPU,
adaptive, MTP timing and longer-context quality matrix remains in progress.

`p5-pinned/` repeats the same eight requests with full `cudaHostAllocMapped`
residency. All first-token logits and all 128-token continuations are bit-identical
to `p5-reference/`; HTTP checks and executed 8192 prefill also pass. This changes
allocation and data movement, not the expert representation or arithmetic. The
same requests are in `p5-reference/requests.json`; binary hashes and raw GPU
samples are included with each run. Repeated default selection remains pending.

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

## Build the runtime

```powershell
cd G:\Strata\Strata-Safetensors
.\tools\build_safetensors_engine.ps1
.\.venv-native\Scripts\python.exe tools\native_config.py `
  --model D:\迅雷下载\Qwen3.8-flash-next-nvfp4 --output build-native-engine\native.json
.\.venv-native\Scripts\python.exe -m serve.server --engine strata `
  --config build-native-engine\native.json --host 127.0.0.1 --port 8097
```

The config generator refuses writes inside the model directory and refuses to
overwrite an existing config. Its CUDA DLL paths refer to the installed toolkit;
`--cuda-bin` overrides that location. The server uses a private port and its own
child engine. Install the native Python requirements into this checkout's own
virtual environment. Do not invoke the inherited installer to set up this path.

## Historical P0/P1 record

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
