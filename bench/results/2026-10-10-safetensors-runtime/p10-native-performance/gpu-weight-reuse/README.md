# Opt-in NVFP4 FP32 weight reuse

`STRATA_NVFP4_F32_REUSE=1` changes the existing FP32 expert loop's traversal:
decode a weight once, apply it to several activation rows, then continue.
Weight codes/scales, per-lane FMA order, warp reductions, global scales and
SwiGLU remain unchanged. This is a software FP32 path; it is not an FP4
Tensor Core speed claim. The inherited FP4 Tensor Core path remains separate.
The original loop remains the global default.

The initial `strata-efficiency4.exe` uses an eight-entry tile. Its 24-case
benchmark covers 1/16/64 distinct physical blobs, 1..8 entries per group,
permuted destinations/tokens, empty/uneven groups, strided grids and canaries.
Every candidate output matches the original loop bit for bit. Six complete
real experts also pass the independent FP64 SwiGLU/Down oracle, worst
relative L2 error 2.7e-7. `gpu-reuse-bench.jsonl`, the six oracle logs and
binary hashes record that build. Printed oracle expert IDs are local 0/0;
the filenames record the actual checkpoint IDs.

Five alternating CUDA-graph timing rounds show faster multi-entry groups,
but slower single-entry groups. Resource inspection reports Gate/Up registers
38 -> 64, Down 35 -> 48, with no stack spills. The first complete CPU75 /
elastic-KV comparison measures only 77.74 -> 79.01 tok/s (one run each).
Microbenchmark ratios must not be read as model throughput ratios.

The subsequent build adds tile sizes 3, 4 and 8, selected by
`STRATA_NVFP4_F32_REUSE_TILE` (3 if unspecified). Each tile keeps independent
accumulators and the original arithmetic order; larger groups use multiple
tiles. The flag remains opt-in. Results for these builds are recorded separately.
All 72 tile/shape cases pass bit equality. The tile3 FP64 checks also pass on
all six real experts. Tile3 uses 40 Gate/Up and 39 Down registers; tile4 uses
40 in both kernels. Neither spills. Single-entry groups remain slower in the
microbenchmark, so smaller register counts alone do not establish a benefit.
The full CPU75/kernel-copy screens measure 81.60 tok/s with tile3 and 82.77
with tile4; this small difference has not yet been established by repeated pairs.
CUDA SM120 is built locally; HIP and SYCL are not validated on this machine.
