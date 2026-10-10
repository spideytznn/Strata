# Visual desktop prefill allocation correction

RTX 5090 32 GB / Ryzen 9950X3D / 96 GB, Windows, 2026-10-10.
The user reported lower prefill performance after enabling images. No speed
comparison was run, respecting the request to measure performance themselves.

The active desktop configuration previously asked for 4096 prompt rows with
a 2048 MiB VRAM reserve. Startup logs show only 2994 MiB free when the 2819
MiB workspace was about to allocate. The allocator also requires 512 MiB
headroom, so it reduced the prompt path to 3072 rows. After verifier buffers
and all remaining startup allocations, only 22 MiB remained and the engine
printed its LOW warning. `before-allocation.json` preserves those specific
lines and the original local log hash; it excludes conversation contents.

Only the desktop `--vram-reserve-mib` changed, from 2048 to 3072. Configured
prefill remains 4096. GPU vision, official thinking sampling, INT8 KV, native
MTP2, CPU75, the 262144 context and port 8880 are unchanged. No engine
source, arithmetic or visual model file changed. Experts remain resident
in the locked RAM arena; fewer initial expert slots are allocated in VRAM.

## Startup check

A private stdin check loaded the configured `Vision` GPU encoder first,
then `StrataEngine`, using `child_env` / `vision_env` exactly as the server.
The encoder warmed at its configured maximum 1024 image tokens. It asserted
`engine.info['prefill_chunk'] == 4096` and at least 512 MiB free in the final
startup log, then closed both processes. No HTTP or generation request ran.

Result: actual dedicated prefill capacity **4096**, **746 MiB** free with
everything loaded, **4998** initial expert cache slots, and zero post-residency
expert/MTP source bytes. `results.json` records the complete config, binary
SHA256 and INFO fields. Raw startup logs are included, normalized UTF8/LF.
No rebuild was needed for this config-only adjustment. JSON and diff checks
passed. Original Q4XL and model files were not modified.

This confirms the allocation and headroom correction under this desktop
state. It does not establish a prefill/decode speed improvement, all-request
VRAM sufficiency, or the sole cause of every slow request. Prior throughput
measurements used other settings. The user will test after restarting the
desktop launcher; short cached continuations have different fixed overhead
from long cold prompts and should be reported separately.
