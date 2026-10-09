# MTP projection adapters, 2026-10-10

Machine: RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM; Windows, CUDA 13.0,
MSVC 2022. Original model: `D:\迅雷下载\Qwen3.8-flash-next-nvfp4`.
These are build/CPU checks, not an inference speed or quality benchmark.

The opt-in Q8_0 loader streams the ten existing MTP projection matrices from
the original safetensors through ggml's reference quantizer. A CPU-only pass
converted all ten: 138,936,320 input BF16 bytes, 73,809,920 output Q8 bytes.
It read no main experts or ngram weights. Synthetic tests check known Q8 scale
and signed code values, half-way rounding, row slices, tile offsets, canaries,
zero blocks and invalid/nonfinite input rejection.

Commands completed:

```powershell
tools/build_safetensors_engine.ps1 -Jobs 2 -OutputName strata-prefill `
  -Targets @('strata','mtp_q8_test','conv_cache_test','message_boundary_test')
build-native-engine/mtp_q8_test.exe 'D:\迅雷下载\Qwen3.8-flash-next-nvfp4'
tools/build_safetensors.ps1
G:/Strata/Strata/.venv/Scripts/ctest.exe --test-dir build-native-engine `
  -R 'mtp_q8_stream|conv_cache_test|message_boundary_test' --output-on-failure
```

The SM120 CUDA engine compiled successfully. All five native-reader/layout
CTest cases and all three selected engine CPU cases passed; their CTest logs
are included here. Generator checks also passed: BF16 defaults are unchanged,
MTP4 produces `--spec 5 --mtp-max-t 5`, Q8 and BF16-batch options are explicit,
invalid combinations are rejected, and both desktop profiles retain INT8 KV,
262,144 context, 8,192 prefill, dedicated workspace and wildcard Host handling.

Staged `strata-prefill.exe` SHA256:
`f30affdf953a92ec581ee1448d90a57e682192acd46443cb61b739e883020805`.

The user started the new Q8 profile during this work. Its startup log confirmed
ten Q8_0 projections and 512 original FP8 experts, 2,511.0 MiB of MTP weights,
2,823 MiB total MTP VRAM and 2.31 s MTP startup. This confirms load/upload, not
GPU output correctness or improved prefill throughput. The assistant did not
start or restart a model process or submit generation requests.

An additional BF16 batch path is compiled but opt-in through
`STRATA_MTP_BATCH_BF16=1`; the matching BF16/MTP4 profile is separate from the
desktop Q8 profile. It reuses existing three-component BF16 GEMM for FP32
activations and the existing workspace. Neither Q8/MTP4 nor BF16-batch/MTP4 has
completed GPU parity, draft acceptance, conversation-cache restore, zero
post-load expert-I/O checks or controlled end-to-end comparison. HIP/SYCL were
not built. Preserve the existing BF16 fallback until that evidence exists.
