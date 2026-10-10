# CPU/GPU arithmetic consistency

RTX 5090 SM120, Ryzen 9950X3D, 96 GB RAM, Windows, driver 617.14.
Original NVIDIA NVFP4 codes and scales, FP32 expert activations/intermediates.
No activation requantization or draft tokens in the teacher tests. Each teacher
case uses 4499 GPU expert slots, 4096 context, 256 prefill and fixed residency;
this is a correctness suite, not the 262K operator throughput configuration.

## Loop optimization versus provider arithmetic

The old 16-lane CPU rolled and unrolled loops produce byte-identical complete
248320-element logits at all 234 teacher positions (English 68, Chinese 82,
code 84). The unroll itself introduces no observed numerical change.

The inherited CPU/GPU arithmetic differs: CPU uses a 16-lane accumulation and
host expf, GPU a 32-lane accumulation and device expf. Against the original GPU
teacher result, CPU75/unroll has 233/234 equal argmaxes, mean KL 0.0001267584
and mean target NLL delta -0.00155146. A 32-lane CPU reduction alone improves
same-input expert agreement but does not eliminate full-model differences
(232/234 argmaxes, mean KL 0.0000916880, NLL delta +0.000601979).
Small floating differences can change discrete router selections. Neither
experiment is evidence of weight loss, nor a justification to hide differences.

## Opt-in canonical expert arithmetic

`STRATA_NVFP4_F32_CANONICAL=1` gives CPU experts the GPU's 32-lane FMA and
XOR reduction order. Both providers evaluate exp in double, round to float,
then retain the original float SwiGLU operations. Weights remain unchanged.
The normal CPU and GPU arithmetic remains the global default. This flag also
selects the unrolled CPU loop. Canonical GPU reuse uses four-entry tiles, or
one-entry tiles when reuse is off; the noncanonical tile selector is separate.

Six complete real expert exports pass independent FP64 output checks. With
identical random inputs, all 15360 CPU/GPU hidden values and 61440 output values
match bit for bit. All 24 canonical GPU layout cases pass, including 1..8
entries, empty/uneven groups, permuted destinations, strided grids and canaries.
`canonical-expert-parity/results.json` records commands and binary/blob hashes.
The blobs and large logit arrays remain local; no model weights are committed.

At all 234 full-model teacher positions, canonical GPU and CPU75 have **exactly
equal complete logits**: mean KL 0, argmax agreement 100%, target NLL delta 0.
The provider change therefore does not alter these sampled main-model outputs.
This does not assert bit equality across all hardware or all possible inputs.

Canonical GPU versus the original GPU is an arithmetic change: argmax matches
232/234 positions, mean KL 0.0000999431, target NLL improves by 0.000500850 on
average (1.18857620 -> 1.18807535). This small fixed teacher suite is not a
general text-quality evaluation or a BF16 checkpoint comparison. Long-output,
speculation, prefix and disk-cache checks are separate acceptance gates.

## Reproduction and limits

Build `strata`, `nvfp4_cpu_f32_test`, `nvfp4_expert_gpu_parity` and
`nvfp4_gpu_reuse_bench` with `tools/build_safetensors_engine.ps1` for SM120.
Run `tools/native_expert_provider_parity.py --help` for the real-export test.
The full-model command uses `tools/native_quality_suite.py --teacher --cases
reference,cpu75-unroll` with the canonical candidate config recorded here.

The standalone loader's five CTests pass in 7.47 seconds, including the existing
visible speculation-boundary test. CUDA was built and exercised locally; HIP
and SYCL were not built or exercised. No desktop profile is selected by these
arithmetic-only tests, and no speedup is inferred from exactness.
