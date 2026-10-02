# Local retained changes (2026-10-02)

This branch starts at upstream v0.1.35, commit
`d9ab8435f654c368c586340d490915f6addf56a3`.
It keeps two small memory planning changes for the local Windows RTX 5090
(32 GiB VRAM, 48 GiB system RAM). It does not change the model, kernels,
FP8 PLE reader, HTTP server or conversation snapshot format.

## Independent prefill buffers

On a single GPU, after filling the expert cache, the planner prices independent
prefill buffers against the remaining free VRAM. It preserves the configured
VRAM reserve and chooses an independent buffer only if its batch is at least as
large as upstream's borrowed buffer. It does not shrink the expert cache to make
room. With the normal auto-sized cache there may be no suitable free space; the
upstream borrowing path then remains active. Multi-GPU and remote-cache paths
remain upstream's. `STRATA_PREFILL_OWN_AUTO=0` disables this local choice for A/B
comparison. An explicit `--no-prefill-borrow` keeps its upstream meaning.

## RAM budget with upstream conversation caching

When RAM-resident experts and conversation caching are enabled, expert headroom
is at least the conversation budget plus the physical RAM floor plus 256 MiB.
A larger explicit headroom is respected. Disabling conversation caching keeps
upstream expert headroom unchanged. This is an allocation budget, not a
guarantee against other applications allocating RAM later.

The local launch configuration uses upstream's `--conversation-cache-mib 2048`,
`--conversation-cache-slots 2`, and `--conversation-cache-min-free-mib 2560`.
These imply at least 4.75 GiB of expert headroom. Snapshot accounting, prefix
checkpoints, restoration, growth, isolation, and eviction all remain upstream's.
Two slots are an upper bound, not a guarantee that two arbitrarily large
conversations fit within 2 GiB. Snapshots are temporary RAM state, lost when the
engine exits. No SSD session cache is implemented.

## Validation

The local planner's 11 boundary checks and 30 upstream conversation-cache Python
tests pass. The engine builds with CUDA 13 / MSVC Release for sm_120.
The upstream real-model A/B/A parity harness passes on IQ3_S, int8 KV and FP8 PLE:
output and the restored main-model state hashes match with caching off/on;
one 984-token checkpoint restored in approximately 26 ms.

A sequential direct-engine comparison with the same 2 GiB cache, 4.75 GiB
headroom, auto expert cache and 262144-token limit read a fresh 17944-token prompt
at approximately 2925 tok/s (official) and 2919 tok/s (local), with warm files.
Returning to the same prompt reused 17937 tokens. One 512-token decode measured
121.2 tok/s (official) and 112.6 tok/s (local); one pair cannot establish a stable
speed difference. The normal auto-sized cache used borrowing in both arms.
Three further decode runs in reverse order measured 107.3/146.0/136.7 tok/s
(local) and 95.7/94.2/128.2 tok/s (official). The order-dependent spread reinforces
that these short measurements do not establish a steady decode speed change.
The Python engine harness does not launch the separate vision helper, so these
measurements are not a prediction of the complete desktop service's speed.
The local changes are memory planning choices, not a claim of general speedup.

An isolated same-binary comparison capped the expert-cache byte budget at
5400 MiB (7056 sized expert slots, 13.39 GiB actual VRAM allocation). Both arms
used 8192-token prefill batches. Borrowing read the warm 17944-token prompt in
7762.6 ms (2311.6 tok/s); independent 3805 MiB buffers read it in 6183.9 ms
(2901.7 tok/s), retaining a 1536 MiB VRAM reserve. Arithmetic and the expected
`OK` answers matched. This was one sequential pair with warm-file confounding,
so it supports retaining the option rather than predicting a general speedup.
The default expert cache remains `auto`; it is not capped to obtain this result.

The exact benchmark results are kept with the local deployment report.
