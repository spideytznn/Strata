# Opt-in FP32 AVX-512 row loop

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, MSVC 19.44.35228. The expert
blocks, micro scales, global scales, FP32 activations and intermediates stay
unchanged. `STRATA_NVFP4_F32_UNROLL=1` specializes token counts 1 through 8
and extracts decoded code vectors directly, preserving each lane's FMA order.
Without the flag, the original loop runs.

The CPU microbenchmark covers both real row geometries (640 x 2560 and
2560 x 1280), all eight token counts, output canaries and an independent FP64
oracle. All 16 cases are bit-equal to the old loop; worst sampled relative
L2 error is below 2.4e-7. Seven alternating rounds of three calls report
medians. The single-thread hot-weight speedups are about 2 to 3.6 times;
these are **not** model throughput measurements.

The complete Gate/Up, SwiGLU and Down test also passes on real exports 0/0
and 47/511, with separate flag-off/on processes producing identical hidden
and output binary hashes. The original one-/four-token grouping and FP64
checks remain enabled. Logs and hashes are in this directory; output arrays
remain local, not in Git.

The first end-to-end screen with 50% of cold experts on CPU measured
61.86 tok/s without unrolling and 64.10 with it. This is one run each,
not a repeated acceptance result. Full-model improvements are limited by
memory bandwidth, GPU work and scheduling. See the parent report for final
paired measurements and selected settings.

CUDA SM120 and the five CPU safetensors CTests build/pass on this machine.
HIP and SYCL toolchains are unavailable and were not built or validated.
