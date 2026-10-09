# Optional float weights for the NVIDIA NVFP4 checkpoint

This local branch adds a native BF16 projection path so an NVFP4 checkpoint does
not need Q8_0 attention, shared-expert, embedding or output weights. Existing
quantized models keep their existing paths. CUDA on an RTX 5090 is the tested
backend; HIP and SYCL have not been built or validated for these additions.

This branch is based on `sergqwer/strata-nvfp4` at `2f65abf6`, with the
additional changes described below. It is published as a separate experimental
branch; it does not replace the main Strata deployment.

Build the CUDA engine and numerical checks (CUDA 13.0 and an SM120 GPU were
used for validation):

```
cmake -S . -B build-fidelity -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_NVFP4_TC=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp
cmake --build build-fidelity --target strata fidelity_projection_test \
  nvfp4_avx512_parity nvfp4_expert_gpu_parity nvfp4_tensor_test
```

On Windows, run these commands from an x64 Visual Studio developer shell and
set the CUDA compiler path if CMake does not find it. The FP4 library targets
`120a` independently of the ordinary CUDA kernels. Leave `STRATA_NVFP4_TC` OFF
when that architecture is unavailable.

Prepare a previously converted F32/NVFP4 GGUF:

```
python tools/fidelity_dense_pack.py --gguf model-F32.gguf --out dense-BF16-exact.gguf
python tools/iq_pack.py --gguf model-F32.gguf --out pack/nvfp4-fidelity --fidelity
```

The sidecar retains the source metadata verbatim and refuses a matrix whose F32
values are not exactly BF16. It reads every output tensor back and compares its
bits against the input. Norms and transformed F32 values stay F32 in the pack.
The fidelity pack also keeps the PLE conv weights as BF16: some source values
are below FP16's normal range and would otherwise be rounded.

Run with the main GGUF for experts, the sidecar for ordinary matrices and the
exact FP8 PLE table:

```
strata --pack pack/nvfp4-fidelity --native model-F32.gguf \
  --native-dense-gguf dense-BF16-exact.gguf --native-head-gguf dense-BF16-exact.gguf \
  --embd-gguf dense-BF16-exact.gguf --ple-gguf ple-fp8.gguf \
  --expert-profile expert-profile.bin --expert-cache auto --pcie-frac 1 \
  --prefill 512 --spec 2 --kv fp16 --max-context 8192
```

The ordinary BF16 projections read FP32 activations during decode. Their prefill
GEMMs split FP32 activations into three BF16 components, accumulate in FP32 and
never convert the weights to FP16. This is a numerical approximation to an FP32
dot product, not a promise of identical floating-point accumulation order.

`STRATA_NVFP4_F32=1` selects reference GPU expert decode with unquantized FP32
activations and SwiGLU intermediates. It reads the original NVFP4 blocks and all
three global scale tails; it does not use FP4 Tensor Cores. Use `--pcie-frac 1`
to keep misses on this GPU path. CPU expert calculations are unchanged.
Omitting the variable keeps the fork's Q8_1 expert decode.

For the higher-precision prefill reference, set `STRATA_PREFILL_NVFP4=fp16`,
`STRATA_PREFILL_CPU_SHARE=0`, and `STRATA_PREFILL_BF16X2=1`. The routed-expert
prefill reference still uses the fork's FP16 GEMMs. Its optional `w4a4x2` mode
remains available for FP4 Tensor Cores; its accuracy/performance tradeoff is not
established by these local adaptations.

For faster prompt reading without changing the stored weights, the local user
configuration now selects `STRATA_PREFILL_NVFP4=w4a8` (INT8 activations), measured
CPU sharing, and `--prefill auto:8192`. It uses FP16 KV, a 262144-token capacity
with 32768 GPU KV cells, a 4 GiB conversation cache, GPU vision and the existing
Qwen MTP draft. These are runtime choices, separate from the lossless pack.

The user's external froggeric v22.5 template is restored in both local model
packs. The web history serializer now includes reasoning alongside answer text,
so the template can reproduce the previous assistant turn's token prefix.
`STRATA_SPEC_STOP_BOUNDARY=1` additionally caps a serial speculative window's
commit at EOS or the output token limit. Without it, accepted lookahead beyond
the visible reply can make the next turn rewind to its prompt checkpoint.
The variable is opt-in; the default commit path is unchanged. Pipelined and
multi-slot decoding have not been adapted or tested for this flag.

Validation targets:

```
fidelity_projection_test
nvfp4_avx512_parity
nvfp4_expert_gpu_parity pack/nvfp4-fidelity/experts.bin 0 0 17 123 47 511
```

The first test compares decode and prefill projections with independent FP64 dot
products, including token tiling and output strides. It also checks that GDN's
optional FP32 output includes normalization and gating, while its default FP16
output remains bit-identical. The expert test compares
real expert blobs with a double-precision expert, and uses a stricter tolerance
when `STRATA_NVFP4_F32` is set. Model quality still requires a separate benchmark
against NVIDIA's reference implementation.

## Optional SM120 FP4 expert decode

Build with `-DSTRATA_NVFP4_TC=ON` (default OFF). This adds a CUDA static library
compiled for `120a`; the ordinary kernels retain the build's original architecture.
The implementation requires SM120 and the 2560/640 NVFP4 expert geometry. It uses
warp-level block-scaled `mma.sync` and preserves the GGUF weight bytes, FP8 block
scales and three FP32 weight-scale tails. Unset/zero `STRATA_NVFP4_TC` preserves
the previous expert implementation. With the switch enabled it takes precedence
over `STRATA_NVFP4_F32`, which remains available as a reference.

Extract the activation scales using the same `gguf` Python module as the converter:

```
python tools/extract_nvfp4_input_scales.py --gguf model-F32.gguf --out input-scales.bin
```

The extractor checks all 73728 original scales: each layer/projection must have
512 equal, positive finite values, and gate/up must agree. The sidecar contains
48 little-endian FP32 gate/up,down pairs. Missing/nonuniform scales fail rather
than being silently replaced with a guess.

Set `STRATA_NVFP4_INPUT_SCALES` to that file and `STRATA_NVFP4_TC=3`. Keep
`--pcie-frac 1` to put both resident experts and GPU-staged misses through this
path. Hot expert exchange, MTP, FP16 KV, the template and ordinary BF16 weights
are retained. This feature changes expert decode; batched prefill remains on the
configured W4A8 path. Multi-GPU, pipeline and other GPU architectures are untested.

Mode 1 quantizes activations to one E2M1 term with E4M3 block scales and the
checkpoint global scale. Modes 2/3 also quantize the residual once/twice and sum
two/three FP4 MMA products in FP32. Residual modes extend the row's global scale
by powers of two if its largest value exceeds the checkpoint's finite range.
These compensation modes are local adaptations, not NVIDIA's standard W4A4
recipe, and are not bit-identical to FP32 activation arithmetic.

`nvfp4_tensor_test experts.bin input-scales.bin [layer expert [input_stddev]]`
compares activation packing with an independent CPU quantizer and both matrix
products with double-precision dot products of those quantized inputs. It tests
uneven groups, reordered token/output indices, group striding, zero groups and
CUDA graph replay. Benchmarks use 10-80 distinct actual expert blobs; they exclude
PCIe transfers, routing, attention and the rest of the model.

Measured on RTX 5090, Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13.0:

- Six checkpoint experts with eight Gaussian input rows: three terms gave
  0.116-0.138% relative output error versus NVFP4 weights with FP32 activations.
  A separate high-amplitude outlier test gave 0.124%.
- Two same-prompt long generations with FP16 KV and MTP: reference 60.1/61.1
  tok/s, compensated FP4 68.1/69.3 tok/s. Outputs differ, and the second pair's
  lengths differ (1360 vs 1536), so these are observed session rates rather than
  a controlled fixed-token throughput claim. The historical repetition-bug
  result of 171.6 tok/s is excluded.
- A small teacher-forced comparison over 12 texts / 659 positions gave 99.70%
  top-token agreement, mean absolute target log-probability delta 0.07768, and
  mean NLL change +0.01225 nats/token (+1.23% relative perplexity). This is not
  a general quality benchmark or an NVIDIA/BF16 equivalence result.
- Disassembly contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, and the model log
  confirms the three-term dispatch. Nsight Compute hardware counters could not
  be read because the driver returned `ERR_NVGPUCTRPERM`.

The desktop launcher runs Python in the foreground, opens the browser when ready,
and retains the console after exit. Its `-Reference` option selects the saved
FP32 activation configuration. No additional download or weight conversion is
required.

Before publication on 2026-10-09, the CUDA targets were rebuilt and the float
projection/GDN, BF16 GEMV, CPU NVFP4 parity, three checkpoint expert reference,
and FP4 Tensor Core tests passed on the machine above. The web reasoning-history
regression check and Python syntax checks for the pack/scale tools also passed.
HIP and SYCL toolchains were unavailable, so this is a CUDA experimental branch,
not a claim of validation on all Strata backends.
