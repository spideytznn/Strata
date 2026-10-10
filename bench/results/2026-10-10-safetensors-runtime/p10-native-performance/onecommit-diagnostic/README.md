# Single-token commit control

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows, driver 617.14, efficiency10.
Original safetensors, full 262144-context operator configs, INT8 KV, CPU75,
canonical FP32 experts, 8192 dedicated prefill and BF16 MTP projections.

The preceding full English/Chinese/code 2048-token test rejected MTP2: the warm
code request first differed from cold/no-draft at output index 1061. See
`../draft-stability-2048-rejected/`. Disabling `STRATA_ONE_TOKEN_COMMIT` retains
the ordinary commit graph for single-token windows. The focused code-only run
in `code-2048/` passes: both cold/warm 2048-token sequences equal no-draft.
Measured decode is 100.50/103.44 tok/s versus 61.55/62.07 without drafts on this
code task. These are one pair per config, not the final deployment comparison.
The focused run also has a different preceding request history from the rejected
three-task run, so this alone does not prove that the commit flag caused the fix.

The separate `first-token/` diagnostic uses output cap 1 and state/logit readback.
All four first-token logits arrays (248320 values each) and committed main states
match bit-for-bit: original fast commit cold/warm and ordinary commit cold/warm.
This finds no initial prefix-restoration mismatch. It does not establish the
cause of the later divergence or measure normal throughput. Binary logit dumps
remain local; complete configs, hashes, logs and derived comparisons are here.

Reproduce with `tools/native_operator_state_check.py --new 1`, or use
`tools/native_operator_bench.py --categories code --new 2048`. Configs are embedded
in results.json. Both engines report zero post-residency expert and MTP source
bytes. A full three-task rerun and cache acceptance are required before selection;
their outcomes are recorded separately.

The subsequent full-history three-task rerun passes all 12 requests and all
24576 token IDs; see `../draft-stability-2048/`. This supports selecting the
ordinary commit path for the tested profile. It does not isolate a particular
kernel instruction as the cause of the original failure.
