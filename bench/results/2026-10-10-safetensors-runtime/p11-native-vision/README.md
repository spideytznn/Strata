# Native image path and thinking sampling

RTX 5090 32 GB / Ryzen 9950X3D / 96 GB, Windows, 2026-10-10.
The user requested official thinking sampling and vision, without benchmarking.
The selected desktop config is `config/native/rtx5090-262k-mtp2.json`, now using
`strata-efficiency12.exe` and `--vision`. Context 262144, prefill 8192, INT8 KV,
CPU75 cold experts, two MTP drafts and port 8880 remain configured.

Default sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0,
presence_penalty 0, repetition_penalty 1. Source:
[Qwen's official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices).
The existing config-level sampling merge preserves explicit request overrides
and shared web settings; the template already defaults to thinking mode.

The native option guard now permits the inherited image path. No forward or
CUDA kernel was changed. It reuses GENI embedding rows, M-RoPE positions,
image digests in conversation caching and the native PLE reader. The external
GPU encoder uses the installed BF16 mmproj and metadata vocabulary GGUF
read-only; the main language model still loads original NVFP4 safetensors.
The visual files and Q4XL deployment were not modified. The encoder starts
before the main engine chooses its cache size. Image cap: 1024 tokens.
Pillow 12.3.0 was installed in this project's venv for WebP/TIFF conversion.

## Functional check only

```powershell
.\tools\build_safetensors_engine.ps1 -Jobs 2 -OutputName strata-efficiency12 -Targets @('strata')
.\.venv-native\Scripts\python.exe tools/native_vision_smoke.py --config config/native/rtx5090-262k-mtp2.json --output logs/vision-smoke-r12
```

Use a fresh output directory to repeat. The check starts a private stdin
encoder/engine, never HTTP, then closes both. It makes red and blue 224x224
PNG fixtures locally; each encodes to 49 image tokens. All six greedy image
requests identify the correct color, warm copies match exactly, and returning
to the red picture after blue reuses the red prefix. Text after images returns
ZEBRA-417. The eighth request accepts all official sampling fields with
thinking enabled (32 generated tokens, still in reasoning; no assertion of a
completed final answer). Every request reports zero expert source reads.
These checks establish this small functional path, not general visual quality
or throughput, long-image context, image-session disk restore or exhaustive
MTP sampling equivalence. No performance comparison was run. Prior efficiency11
throughput results are text-only greedy and do not describe this new profile.

`results.json` records config, deployed binary SHA256 and token outputs.
Its config SHA256 refers to the deployed local bytes; archived text uses UTF8/LF.
`engine.log` is the functional check's runtime log (normal engine timing lines
are diagnostic, not a benchmark). Python syntax checks and CUDA SM120 build
passed. HIP/SYCL were unavailable and were not built or tested.
