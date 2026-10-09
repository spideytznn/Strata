# Native safetensors backend: P0/P1

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
