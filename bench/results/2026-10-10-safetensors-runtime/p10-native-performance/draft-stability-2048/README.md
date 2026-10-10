# Full-history 2048-token draft stability rerun

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14. Full 262144-context
operator configs, INT8 KV, CPU75 canonical FP32 experts, dedicated 8192 prefill,
original BF16 MTP projections. Private stdin engines. The no-draft reference
retains the MTP weights to separate drafting from VRAM capacity changes.

All 12 requests reach the 2048-token cap. English, Chinese and code cold/warm
outputs match exactly across no-draft and two-draft configurations. All 24576
output token IDs are retained. All requests have zero expert/MTP source reads.
The candidate sets `STRATA_ONE_TOKEN_COMMIT=0`; it retains the ordinary commit
graph for one-token windows. The formerly failing code warm request now matches
all 2048 reference tokens, after the same preceding English/Chinese requests.
The rejected attempt and focused controls remain separately archived. This is
a bounded regression check, not proof that all speculative workloads are stable.

| Task | Candidate cold tok/s | Candidate warm tok/s | Draft acceptance |
| --- | ---: | ---: | ---: |
| english | 101.57 | 103.48 | 89.63% |
| chinese | 112.14 | 112.50 | 90.01% |
| code | 101.74 | 103.64 | 91.37% |

These are one cold/warm pair per task, not the deployment speed comparison.
Baseline uses efficiency10; candidate uses efficiency11, whose additional change
only binds new arithmetic/commit choices into disk-session identity. The
identity-only build overlapped part of the no-draft English baseline, so those
baseline timings are not used for a speed claim. The candidate ran after build
completion. Executable hashes, full configs, sampling and input hashes are in
results.json. The final alternating CLI comparison is recorded separately.

Reproduce with `tools/native_operator_bench.py --new 2048`, using the embedded
configurations. State hashing and logit dumps are disabled in these runs.
