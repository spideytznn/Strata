# strata-safetensors

**Run Qwen3.8-Flash-Next directly from the original NVIDIA NVFP4 safetensors on one GPU and system RAM.**

[中文说明](README.zh-CN.md) · [Runtime details](docs/SAFETENSORS_NATIVE.md) · [Measurements](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) · [Upstream Strata](https://github.com/Niko1221/Strata)

strata-safetensors adapts Strata's forward pass, expert scheduler and session cache
to the original Hugging Face checkpoint. It keeps the checkpoint's expert codes
and scales, loads experts into locked RAM, caches hot experts in VRAM, and reads
ngram rows from SSD on demand. No main-model GGUF conversion or persistent
converted expert pack is required.

The validated deployment is **Windows + RTX 5090 32 GB + Ryzen 9950X3D + 96 GB RAM**.
The active branch is `main`; the executable and CLI retain the name `strata`.

## What it does

| Capability | Current native implementation |
|---|---|
| Original weights | Loads NVIDIA NVFP4 safetensors, BF16 dense weights, FP8 ngram data and native MTP weights directly. Expert repacking changes layout, preserving codes and scales. |
| Expert residency | All main experts have locked RAM copies, including experts cached on the GPU. Expert and loaded MTP source reads are sealed after loading. |
| CPU/GPU scheduling | Hot experts execute on GPU. Cold experts can execute on CPU with AVX-512 or be transferred to GPU; adaptive swaps update the hot set. |
| Prefill | Dedicated prompt workspace, W4A8 expert prefill, BF16 dense projections with activation-remainder compensation, and prefill-first allocation in the tuned desktop profile. |
| Context | FP16 or INT8 KV; the desktop uses INT8 and grows KV as needed. A 262,000-token input has passed capacity and session-state checks within a 262,144-token total window. |
| MTP | Uses the checkpoint's native draft weights. The main model verifies proposals; draft length and confidence gating are configurable. MTP can be fully unloaded. |
| Session reuse | Prefix reuse, multiple parked conversations, A/B/A switching, and disk save/restore of KV and linear-attention state. |
| Vision | Image input through Strata's existing external BF16 encoder. The visual sidecar uses mmproj/vocabulary GGUF files; the main model stays in safetensors. |
| Serving | Web chat, OpenAI-compatible and Anthropic-compatible APIs, streaming and tool calling. Native generation is serial; requests queue. |

This is a model-specific backend, not a general safetensors runner. See
[scope and validation](docs/SAFETENSORS_NATIVE.md) for the supported geometry,
numerical comparisons and remaining limits.

## How the optimizations work

- **Keep experts resident.** The full expert arena is CUDA-pinned and GPU-mapped,
  avoiding an extra host staging copy. Runtime SSD reads are for ngram rows,
  plus session files when explicitly saving or restoring.
- **Use both processors.** The tuned profile runs about 75% of cold experts on
  CPU and 25% on GPU. These fractions apply to cache misses, not all model work.
  AVX-512 row-loop specialization and GPU weight reuse reduce repeated work.
- **Allocate prefill space first.** Allocate and write the requested workspace
  before sizing the expert cache. Budget late allocations, including MTP, and
  refuse an insufficient startup allocation instead of silently reducing capacity.
- **Reduce runtime waits.** Release buffered startup weight handles once resident
  reads are sealed. This produced a measured prefill improvement on Windows;
  the specific Windows cache/driver mechanism has not been established.
- **Reuse conversation state.** Keep both KV and linear-attention state so a
  resumed session processes its new suffix rather than its entire history.

SM120 FP4 Tensor Core kernels are implemented and available as opt-in paths.
They are **not the selected desktop default**: measured FP4 decode did not
establish an end-to-end advantage, and FP4 prefill changed outputs. The current
desktop uses NVFP4 weights with FP32 decode activations, W4A8 expert prefill and
original BF16 projections. Format preservation does not imply byte-identical
arithmetic or equivalence to an unquantized BF16 model.

## Measured performance

All numbers below were measured on **RTX 5090 32 GB / Ryzen 9950X3D / 96 GB RAM,
Windows, 2026-10-10**, using the original NVIDIA checkpoint. Rows describe
different historical comparisons; their gains must not be multiplied.

| Comparison | Before → after | Conditions and evidence |
|---|---|---|
| Startup-handle release, prefill | 822 → 2,465 tok/s | Same binary, about 8K input, three alternating pairs; complete 128-token outputs equal. [P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md) |
| CPU/GPU and kernel profile, decode | 45.28 → 81.59 tok/s | Three alternating pairs; both profiles use two drafts; complete 128-token outputs equal. Candidate range 64.57–85.16. [P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md) |
| Later 4096-row profile, text prefill | 2,968 → 3,331 tok/s | About 24K text, visual encoder loaded, three-start medians. First about-8K text prefill instead falls 2,688 → 2,558. [P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) |
| Later profile, image-request prefill | 2,741 → 3,089 tok/s | About 8K including 1,024 image tokens, three-start medians; total wall 4.51 → 3.94 s. [P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) |

A separate P16 screen measured warm aggregate decode of **91.93 → 100.82 tok/s**
over English, Chinese and code. It used one fresh engine per profile and is
not a repeated throughput estimate. That candidate used four drafts, a 0.7
confidence gate and a two-draft floor.

**The current temperature-0.7 / two-draft / gate-0.5 desktop trial has not been
remeasured.** These figures are evidence for the archived configurations, not
a speed promise for every prompt or the current trial.

## Build and run on Windows

The following is a source-build route for an RTX 5090. Install Git, Python 3.12+,
Visual Studio 2022 C++ Build Tools, CUDA 13, CMake 3.24+ and Ninja. Run the commands
from PowerShell with the **x64 MSVC development environment** activated and those
tools on PATH. Model weights
must already be downloaded outside this repository.

```powershell
git clone https://github.com/spideytznn/strata-safetensors.git
cd strata-safetensors
python -m venv .venv-native
.\.venv-native\Scripts\python.exe -m pip install -r requirements-native.txt

cmake -S . -B build-native-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NVFP4_TC=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CXX_FLAGS=/utf-8 /EHsc"
cmake --build build-native-engine --target strata --parallel 2

.\.venv-native\Scripts\python.exe tools\native_config.py --model "D:\Models\Qwen3.8-Flash-Next-NVFP4" --output strata-native-local.json --exe build-native-engine\strata.exe --context 262144 --prefill 4096 --kv int8 --dedicated-prefill --mtp 2 --spec-min-p 0.5 --load-batch 64 --load-workers 4
```

CMake fetches its pinned ggml dependency unless `STRATA_GGML_DIR` is supplied.
The generator validates model metadata, refuses to overwrite an existing config,
and refuses output inside the model directory. Its generated config keeps
reference FP16 prefill and does **not** reproduce every tuned desktop setting.
Before starting, add this top-level `sampling` field to `strata-native-local.json`
for Qwen's official thinking baseline:

```json
"sampling": {
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0
}
```

Then start the server. Vision requires separately configured encoder assets.

```powershell
.\tools\run_safetensors.ps1 -Config .\strata-native-local.json -Port 8880
```

Open **http://127.0.0.1:8880** for chat; clients use
**http://127.0.0.1:8880/v1**. Check `/health` and `/v1/models` for readiness and
the served model ID. Ctrl+C stops the foreground server. Keep a local bind, or
configure `--api-key` before exposing the API beyond localhost.

The original model directory is read-only input. The measured expert RAM arena
alone occupies **63.282 GiB**, before other weights, sessions, runtime buffers and
the OS. 96 GB RAM is the validated configuration; a 64 GB native-resident setup
has not been established.

### Existing desktop deployment

`START-NATIVE-262K.bat` uses
[`config/native/rtx5090-262k-fast.json`](config/native/rtx5090-262k-fast.json).
This checked-in profile and the local build helper contain deployment-specific
Windows paths; adapt them before use on another PC. The generic build above
avoids those fixed paths.

| Setting | Current desktop trial |
|---|---|
| Context / dedicated prefill | 262,144 total tokens / 4,096 rows |
| KV / vision | INT8 with elastic growth / external GPU encoder, at most 1,024 tokens per image |
| MTP | `--mtp native --spec 3 --spec-min-p 0.5`; at most two drafts, no forced minimum |
| Target sampling | temperature **0.7**, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1 |
| Template | Froggeric v22.5; thinking on, effort defaults to medium when not specified |
| Endpoint | `127.0.0.1:8880`; HTTP Host headers unrestricted |
| Output budget | `fit_max_tokens: true` fits the requested output cap to remaining context; input is not truncated |

The temperature is a user-selected trial. [Qwen's official thinking preset](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices) uses
**1.0 / 0.95 / 20 / 0 / 0 / 1**, and its template defaults to xhigh.
Explicit client parameters and persisted shared web settings can override
server defaults. Output-budget fitting does not change a client's compaction
policy; set the client context to 262,144 as well.

The fully unloaded MTP alternative is
[`rtx5090-262k-no-mtp.json`](config/native/rtx5090-262k-no-mtp.json).
`START-NATIVE-262K-STABLE.bat` retains the earlier efficiency14 profile.

`START-NATIVE-262K-PARALLEL.bat` selects an opt-in **two-request text batch**,
with the same 262K context and dedicated 4096-row prefill. Images run exclusively;
this profile leaves MTP weights unloaded and uses target-only decode for all requests.
Native target-only batching also offers independent per-session elastic KV and
direct device slot migration; the measured local profile enables both.
See [native concurrency](docs/NATIVE_CONCURRENCY.md) for tests, memory costs and
latency/throughput measurements. The original desktop launcher is unchanged.

## Validation and limits

- Weight-layout roundtrips, real-expert FP64 checks, full-logit comparisons,
  seeded sampling, output caps/EOS and cancellation/resume have recorded evidence.
- Final efficiency17 acceptance covers 80 completed requests and four
  cancellations across zero/two/four/six drafts, including output/main-state,
  session switching and disk-restore checks. These are finite regression tests.
- The 262K fixture proves capacity and tested state reuse. It does not establish
  general long-context reasoning quality, retrieval accuracy or 262K throughput.
- Native text requests can decode together with `"parallel": 2`; admissions
  share the prompt reader. Native multi-GPU, batched MTP, pipeline parallelism
  and external draft models are unsupported. Images require exclusive generation.
- Native CUDA SM120 is validated locally. Linux, HIP, SYCL and other hardware
  have not been validated for this backend. Inherited backend support is not
  evidence for the new native path.
- The main model requires no GGUF. Vision still uses an external GGUF sidecar.
  No native vision-tower implementation or automatic native installer is claimed.

## Documentation and provenance

- [Native implementation, diagnostics and acceptance](docs/SAFETENSORS_NATIVE.md)
- [Earlier CPU/GPU performance comparisons](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md)
- [Workspace and speculation measurements](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md)
- [Inherited NVFP4/GGUF documentation](README.legacy-nvfp4.md) and [original Strata README](README.upstream.md)

Built on [Niko1221/Strata](https://github.com/Niko1221/Strata) and
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4), starting from the
NVFP4 fidelity commit `d167eb89301a02e0d6299f49d35494ba5096bc10`.
Their authorship and licenses are retained. The previous personal GGUF main
branch is preserved as `archive/gguf-main-20261011`.

The [upstream proposal](https://github.com/Niko1221/Strata/issues/1858) and
[standalone safetensors-source PR](https://github.com/Niko1221/Strata/pull/1859)
are open as of 2026-10-11. The complete native runtime has not been merged
into upstream Strata.

Engine source is [MIT-licensed](LICENSE). Model weights and visual assets retain
their respective licenses; weights are not distributed in this repository.
