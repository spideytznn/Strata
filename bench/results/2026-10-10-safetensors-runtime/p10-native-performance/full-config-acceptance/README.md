# Complete 262K operator configurations, 8K/24K acceptance

RTX 5090 / 9950X3D / 96 GB, Windows, driver 617.14. Three sequential private
stdin engines use efficiency10: the delivered legacy GPU profile, canonical
GPU with elastic KV/exact workspace pricing, and the same canonical profile
with CPU75 cold-expert scheduling/kernel copying/40 pool tasks. All retain
INT8 KV, dedicated 8192 prefill, original BF16 projections, MTP2, adaptive cache
and 262144 configured capacity. Commands and binary hashes are in the result.

Each config runs nine requests: 8199- and 24583-token initial/warm documents,
A/B/A, then short disk SAVE/RESTORE. All 27 output sequences match the earlier
delivered-profile acceptance reference. Warm reuse is 8192/24576; A/B/A reuse
is 28, short disk restore 43. Every own warm or restored continuation also has
the same committed main-model state as its own original continuation.
Expert and MTP post-loading source bytes are zero in all three engines.

The legacy config's nine fingerprints match the earlier efficiency2 reference,
showing the debug fingerprint batching preserves byte order and the original
arithmetic remains unchanged when new flags are off. Canonical GPU's nine
committed main-model fingerprints exactly match canonical CPU75's, including
all warm and restored states. This exercises provider changes with the full
operator context/workspace, speculation and adaptive-cache settings.

Canonical versus legacy state fingerprints intentionally differ because of
the arithmetic choice documented in `../arithmetic-consistency/README.md`.
The raw multi-config tool therefore uses `--state-comparison report` for that
cross-arithmetic comparison. The companion `state-assertions.json` strictly
checks legacy/reference equality, canonical GPU/CPU equality and all own
warm/restore equalities. These comparisons are not inferred from equal tokens.
MTP scratch/spare state is excluded by the existing committed-main-state parser.

Reproduce with `tools/native_config_pair.py`, the three recorded configs, P9
`details-requests.json` and the earlier handle-fix acceptance as reference, then
apply the recorded state assertions. Fingerprints add overhead; this is not
a throughput run. Contexts above 24K are covered separately by the 64K suite.
CUDA SM120 was built and run. HIP/SYCL were not built on this machine.
