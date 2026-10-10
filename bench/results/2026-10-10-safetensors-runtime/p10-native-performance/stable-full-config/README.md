# Complete profile acceptance with ordinary single-token commit

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14, efficiency11.
262144 capacity, INT8 KV, dedicated 8192 prefill, MTP2/BF16 projections,
adaptive cache and session caching. Private stdin, state hashes enabled.

All 27 requests pass across the preceding desktop profile, canonical all-GPU,
and canonical CPU75 configurations. Every generated token equals the preceding
efficiency2 reference. Every original-profile committed fingerprint equals that
reference, preserving the original default path. Canonical all-GPU and CPU75
complete fingerprints match bit-for-bit, with ordinary single-token commit.
Each config's own cold/warm 8K/24K and A+/disk-restored states also match exactly.
The harness now requires these self-state checks by default. The separate
state-assertions.json additionally gates the intended cross-config comparisons;
legacy versus canonical arithmetic is deliberately not required to match states.

Prefix reuse is 8192/24576, A/B/A reuses 28, and disk restore reuses 43 tokens.
All requests have zero expert/MTP file blobs and bytes; shutdown source counters
also remain zero. Logs prove executed 8192 batches, not just configured capacity.
These fingerprinted requests are correctness checks, not throughput measurements.

Efficiency11 adds disk-session identity fields for opt-in canonical/GPU-order
expert arithmetic and disabled single-token self-commit. Original unset/default
choices retain prior identity fields. The native original forward is unchanged;
these new choices must not silently accept sessions saved under other choices.
Positive same-config SAVE/RESTORE passes here; negative old-file rejection is
recorded separately. CUDA SM120 builds successfully; HIP/SYCL are unavailable
and have not been built or validated.

Use tools/native_config_pair.py with the embedded configs, P9 details-requests.json,
the io-fix-pair-acceptance reference, and --state-comparison report. Then require
original committed_state_equal and canonical run 1 states == run 2 states,
as recorded in state-assertions.json. The configs and executable hashes are in
results.json; binary session files and model weights are not committed.
