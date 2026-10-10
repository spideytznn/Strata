# Rejected 2048-token MTP2 candidate

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14, efficiency10.
Full candidate configs, 262144 capacity, INT8 KV, CPU75 canonical arithmetic,
8192 dedicated prefill, original BF16 MTP projections. Private stdin engines.

All six no-draft English, Chinese and code requests produce 2048 tokens and
have identical own cold/warm outputs. MTP2 English cold/warm, Chinese cold/warm,
and code cold each match the no-draft reference for all 2048 tokens. Code warm
first differs at zero-based output index 1061. The strict harness rejects this
candidate. It must not be described as passing or promoted on these results.
Both engines report zero post-residency expert and MTP source bytes.

Reproduce:

```powershell
.venv-native/Scripts/python.exe tools/native_operator_bench.py --configs logs/efficiency/candidate-configs/canonical-operator-mtp0-r10.json logs/efficiency/candidate-configs/canonical-operator-mtp2-r10.json --new 2048 --output logs/efficiency/long-reproduction
```

The complete configurations and executable SHA256 are embedded in results.json.
These finite checks do not identify the cause of the mismatch. Later diagnostic
runs and any passing replacement are recorded separately.
