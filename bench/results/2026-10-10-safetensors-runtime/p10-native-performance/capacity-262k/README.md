# Near-262K input, full KV capacity and disk restore

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14, efficiency11.
Original safetensors, 262144 total context capacity, INT8 KV, CPU75/canonical
FP32 expert arithmetic, BF16 MTP2, ordinary T1 commit, adaptive experts and
dedicated 8192 prefill. Private stdin with state hashes and STRATA_KVG_CHECK=2.
The normal launch profile disables those diagnostics.

All eight requests pass. The long input is exactly 262000 tokens, leaving room
for the requested 128-token output cap and verifier padding. Cold, warm and
saved-then-restored long requests all emit the exact ZEBRA-262000 access code
and EOS (11 tokens); all three complete main-state fingerprints match bit-for-bit.
Warm/disk restore reuse 261993 prompt tokens. A/B/A and the short disk restore
also match their own output/state exactly.

KV actually maps all 262144 cells. Growth lends 1204 of 4765 expert slots,
leaving 3561; short requests trim to 8192 cells and refill all 4765 slots.
Restoring the long snapshot grows to the full capacity again. All 19 complete
resident-content/table/alias audits have zero errors, including after restore.
Expert and MTP source bytes remain zero after residency. Ngram SSD reads remain
allowed and appear separately in the logs. This proves near-capacity operation
and full KV mapping, not an input of 262144 plus additional generated tokens.
The configured limit counts input and output together.

The frozen input repeats documentation and ends with an exact access code.
It is a capacity/cache test, not a throughput or long-distance retrieval-quality
benchmark. Its source-byte SHA and construction details are in provenance;
use the frozen IDs across platforms/newline styles. Full configs, binary hash,
state comparisons and telemetry are in results.json. Large binary session
snapshots remain local, and no model weights are committed.

Reproduce with tools/native_config_pair.py --configs CONFIG --fixture
capacity-262k-fixture.json --cases capacity-262000 --long-restore --output NEW_DIR,
using the embedded diagnostic config. Runtime identity and ordinary T1 commit
are the same as the separately accepted desktop candidate.
