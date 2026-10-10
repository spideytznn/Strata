# Automatic KV lending, trim and long-session restore

RTX 5090 / 9950X3D / 96 GB, Windows, driver 617.14. Original safetensors,
262144 configured context, INT8 KV, dedicated 8192 prefill, MTP2, CPU75,
kernel copying, automatic adaptive expert cache, canonical FP32 expert math.
`canonical-growth-64k-r10b/results.json` is the complete passing run.

The frozen fixture has 65543 tokens, repeated documentation and a final exact
access code. It is a capacity/cache test, **not a prefill throughput fixture**.
The binary's 4765-slot cache initially maps 16384 KV cells. The long request
grows to 73728 cells (0.96 GiB), giving up 277 expert slots. Short A/B/A requests
trim KV to 8192 cells and refill all 4765 expert slots. Restoring the saved long
session grows KV to 73728 again, giving up 316 slots from the smaller starting
KV. All 19 full resident-content/table/alias audits report zero errors.

Cold, warm and disk-restored long requests return exactly `ZEBRA-65543` followed
by EOS. Warm and restored requests reuse 65536 tokens. The complete committed
main-model fingerprints are identical across all three, despite the intervening
KV resize and different expert assignments. A/B/A and the short disk restore
also pass. Post-loading expert and MTP source bytes remain zero. Disk session
files are test outputs, not expert-weight reads, and are not committed.

## Diagnostic failures and repair

The initial efficiency8 attempt completed all prefill chunks and the last
verify window, but the debug fingerprint's many tiny blocking D2H reads
triggered the 60-second watchdog. The normal desktop profile does not enable
fingerprinting. The repair coalesces complete page/head spans without changing
byte order, keeps partial boundary pages separate, names the fingerprint stage,
and beats the watchdog after actual copied/hashed chunks. It does not disable
the watchdog or alter forward computation.

The first efficiency10 run completed all eight requests, matching recall and
long cold/warm/restored fingerprints, then failed a test-script assertion that
required total prefill to equal 8192. Its actual trace is `tokens=65536
max_chunk=8192 capacity=8192`. The harness now accepts any total of at least
8192 with an executed maximum and capacity of exactly 8192. The entire suite
was rerun in a fresh process and passes. Both failed attempts are retained here.

A separate original-profile regression compares efficiency10 with the earlier
efficiency2 acceptance reference: all nine outputs and committed fingerprints
match, verifying the batched hash retains the original hash order. The CUDA
build and this source change's validation are recorded with the full-config
evidence. Normal throughput benchmarks have fingerprints disabled.

Reproduction uses `tools/native_config_pair.py --long-restore` and the passing
run's config plus `growth-fixture.json`. The generator and original source hash
are included; use the frozen IDs for exact reproduction across newline styles.
Only metadata, logs, telemetry and token fixtures are tracked. This validates
64K operation and 262K configuration, not a complete 262K input run.
