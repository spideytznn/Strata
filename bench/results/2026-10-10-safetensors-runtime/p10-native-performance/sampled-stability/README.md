# Fixed-seed sampled output stability

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14, efficiency11.
Temperature 0.7, seed 9950; English, Chinese and code, each cold/warm, output
cap 512. Original checkpoint, 262144 capacity, INT8 KV, dedicated 8192 prefill,
CPU75 canonical arithmetic and ordinary single-token commit.

All 18 requests reach 512 tokens. Every cold/warm output equals its corresponding
output across these three configurations: no drafts with MTP weights retained,
two drafts with original BF16 projections, and MTP completely unloaded. All
9216 token IDs, full configs, input/executable hashes and sampling values are
retained. Prefix reuse is positive for every warm request; expert and MTP source
bytes remain zero. No state/logit diagnostic readback affects these runs.

This checks sampled verification and replay as well as the greedy 2048-token
suite. It is one seed, three prompts and finite output lengths, not proof of
all speculative workloads or a repeated throughput comparison. The unloaded
config has more expert VRAM than the retained-weight reference, so its speed
must not be interpreted as an isolated draft-depth effect.

Reproduce with tools/native_operator_bench.py --new 512 --temperature 0.7
--seed 9950, using the embedded configurations. The private engines never
open HTTP or connect to a running user service.
