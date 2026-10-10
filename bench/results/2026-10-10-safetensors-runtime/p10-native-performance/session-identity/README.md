# Session identity refuses incompatible arithmetic

Efficiency11, RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14.
The candidate loads the unchanged checkpoint and attempts RESTORE of a session
created by the preceding efficiency10 canonical config, which did not bind the
new arithmetic/commit choices into the config fingerprint. RESTORE is refused
as invalid, with config fingerprint differs, before payload restoration.

The same engine then serves cold/warm code prompts successfully: their output,
complete committed main state and all 248320 first logits agree bit-for-bit.
This diagnostic caps output at 1 and enables state/logit readback; it is not
a performance run. Binary logit arrays and the old session remain local.
The rejected session SHA, executable SHA, full config and refusal are in
results.json; the engine log records zero expert/MTP source bytes.

Reproduce with tools/native_operator_state_check.py --reject-session PATH --new 1,
using an older same-model file whose fingerprint lacks the new mode fields.
New-mode same-config SAVE/RESTORE already passes stable-full-config and
stable-kv-auto-64k. Original unset/default arithmetic retains its prior identity
fields. This change does not delete session files or change forward arithmetic.
