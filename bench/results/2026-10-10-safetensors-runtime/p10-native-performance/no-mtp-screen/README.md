# Fully unloaded MTP screen

RTX 5090 / Ryzen 9950X3D / 96 GB, Windows/driver 617.14, efficiency11.
Same varied P9 8199-token input and all 128 output IDs equal the preceding
desktop comparison. CPU75, canonical FP32 experts, INT8 KV, exact/dedicated
8192 prefill, elastic KV, adaptive cache and ordinary single-token commit.
Unlike the no-draft operator reference, this config removes --mtp entirely:
all MTP weight and KV allocations are absent. The initial expert cache grows
to 5770 slots / 14.86 GiB versus 4765 / 12.27 GiB with MTP loaded.

One fresh CLI run measures prefill 3100.40 and decode 50.50 tok/s, with zero
post-residency expert/MTP source bytes. This is a single-run screen, not a
three-round default-selection result. It improves prefill but is substantially
slower on this decode fixture than the paired two-draft candidate. Both the
no-draft (weights retained) and fully unloaded configurations have separate
seeded-output and cache checks. This screen alone does not prove either mode
stable on all workloads. Full command, binary/input hashes, tokens and telemetry
are retained; manifest hashes are over original local bytes, while archived
text is normalized to UTF8/LF. Input tokens.txt bytes are preserved exactly.
