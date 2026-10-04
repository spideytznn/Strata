# Local SSD expert promotion (2026-10-04)

This opt-in extension adds a dynamic RAM tier to the existing SSD / RAM / VRAM
expert path. It is supported on a single GPU with a resident RAM budget and
speculative decoding, and leaves the upstream GPU adaptive tier enabled.

The Q4 launch configuration enables:

```
--ram-hot-experts --ram-hot-every 16 --ram-hot-swaps 16
```

At a decode-window boundary, after pending GPU exchanges have committed and
expert workers have finished, the extension compares the adaptive tier's decayed
routing counts. A file-tier expert must have a count of at least 2 and exceed a
RAM victim's count by at least 1.5. Up to 16 candidates are promoted every 16
decode rounds, in descending heat-gain order. Counts include speculative routed
tokens. Prompt processing does not trigger this RAM replacement policy.

Replacements stay within the same layer: native GGUF experts can have different
byte sizes in different layers. The promoted expert reuses the victim's byte
range in the original resident arena. GPU readers are synchronized before any
range is overwritten. A complete file/staging read precedes replacement, so a
failed read leaves that victim intact. An evicted RAM expert falls back to its
original GGUF/experts.bin bytes. A pending RAM/VRAM exchange blocks promotion.
The router lookahead's RAM-residency checks share a mutex with offset updates.

No second RAM arena is allocated. The configured 31.5 GiB expert arena, original
CUDA registration boundaries, INT8 KV / 262144 context, and 8192 prefill batch
remain unchanged. The existing staging buffers and small planner vectors are
still used. A promoted expert in a registered RAM range can use the existing
PCIe GPU path and can later enter VRAM through the upstream adaptive policy.
Placement changes can change CPU/GPU rounding; expert weight bytes are unchanged.

The engine prints `hot RAM ON` at startup and, after each request, `hot RAM: N
file-tier experts promoted, M MB copied ...` (cumulative). `N` counts admissions,
not unique experts. The copied MB are logical promotion bytes; they are not a
measurement of physical SSD traffic. Disabled builds/configurations retain the
original static RAM behavior. Remove `--ram-hot-experts` from the launch args to
disable the extension. The RAM placement is not persisted across restarts.

Validation is recorded in the deployment report in the current Codex workspace.
The full pre-change source, engine, launch files, expert profile, and maintainer
Git history are backed up at:

`D:\Strata\backups\before-ssd-hot-ram-20261004-140119`
