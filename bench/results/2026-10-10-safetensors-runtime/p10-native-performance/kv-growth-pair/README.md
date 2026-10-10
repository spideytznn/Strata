# Fixed-residency KV growth and loop comparison

RTX 5090 / 9950X3D / 96 GB, Windows, driver 617.14; original safetensors,
INT8 KV, dedicated 8192 prefill, original BF16 projections, MTP2, CPU75.
Both configs request 3174 cache slots (3173 actual), retaining the same expert
assignments. Grow0 uses full KV and original CPU/GPU loops; grow1 uses elastic
KV, CPU unroll and tile4 GPU weight reuse. Canonical arithmetic is off in both.

All nine requests per config pass, including 8K/24K initial/warm documents,
A/B/A and disk SAVE/RESTORE. Full output sequences and committed main-model
state fingerprints match across configs. Warm outputs equal their own cold
outputs. Expert and MTP source bytes after loading are zero. Full resident
expert-content and alias audits report no errors during 16K -> 32K KV growth.

Own warm/restored fingerprints differ in these legacy mixed CPU/GPU cases,
despite equal tokens, because changing provider assignments changes rounding.
They are not evidence of provider-independent main state. The canonical
full-config and 64K suites later require exact own warm/restore state as well.
Reproducing this older experiment with the current harness requires the
explicit `--self-state-comparison report` option; default checks are stricter.

This fixes `STRATA_KV_GROW_FLOOR=3174`: growth uses new VRAM instead of lending
expert slots. It does not validate cache lending, automatic cache sizing, or
trim/refill. The later 64K automatic-cache test covers those separately.
State hashing and full content audits add overhead; these are not throughput
measurements. No model weights or disk session binaries are committed.
