# 64K automatic KV acceptance with the selected commit path

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14, efficiency11.
Full 262144-context candidate, INT8 KV, CPU75/canonical arithmetic, MTP2,
original BF16 projections, ordinary single-token commit, 8192 dedicated prefill.
STRATA_KVG_CHECK=2 and state hashing are diagnostic-only; neither belongs in
the normal launch profile.

All eight requests pass, including the 65543-token cold, warm and disk-restored
long session. The exact access code ZEBRA-65543 and EOS agree in all three;
warm/disk restore reuse 65536 tokens. All three complete committed states
match bit-for-bit. Short A/B/A and its disk restore also preserve output/state.
The strengthened harness requires these self-state comparisons.

Automatic KV grows from 16384 to 73728 cells, lending 277 of 4765 expert slots.
Short prompts trim to 8192 cells and refill all 4765 slots. Long disk restore
then grows again and lends 316 slots. All 19 full content/table/alias audits
have zero errors. Expert and MTP source bytes stay zero after loading.
The native model's session-model fingerprint is prepared at startup and reused;
SAVE/RESTORE do not resample expert-weight files at request time.

The frozen repeated-document fixture and generator are in ../kv-auto-64k/.
This is a cache/capacity check, not a prefill throughput fixture. It validates
64K input under a 262K configured capacity, not an actual 262K input run.
Reproduce with tools/native_config_pair.py --long-restore and the embedded
config, using ../kv-auto-64k/growth-fixture.json. Full config/executable hashes,
logs and telemetry are retained; binary session snapshots remain local.
