# Fully unloaded MTP cache acceptance

Efficiency11 on RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14.
The full candidate settings are retained, except MTP weights/KV are completely
absent and suffix drafts remain disabled. All nine 8K/24K cold/warm, A/B/A and
disk-restored requests match the preceding desktop token reference. Own warm
and restored committed states match exactly. A separate strict comparison
also confirms all nine complete committed main states equal the loaded-MTP2
candidate in ../stable-full-config/, despite different expert VRAM capacity.
state-assertions.json records this comparison.

Prefix reuse, executed 8192 batches, zero expert/MTP source reads and positive
same-config SAVE/RESTORE all pass. State hashes add overhead, so these are not
throughput runs. The separate unloaded-MTP screen is one run (50.50 decode /
3100.40 prefill tok/s), not a repeated selection result. Seeded sampled parity
is separately checked in ../sampled-stability/. Reproduce with the embedded
config and tools/native_config_pair.py, P9 details-requests.json and the
io-fix-pair-acceptance token reference with --state-comparison report; then
require the nine states equal the canonical CPU75 MTP2 run as recorded here.
