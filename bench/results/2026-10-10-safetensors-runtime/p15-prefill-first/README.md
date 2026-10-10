# Allocate the prompt workspace before the expert cache

Requested after the real desktop fallback in P14. The initial implementation
commit `f72d339d` recorded build checks while the user's service was running.
After the user released resources, private GPU acceptance completed below.
No HTTP request was sent to the user's server.

## Allocation order

With `STRATA_NATIVE_PREFILL_FIRST=1`:

1. After the original weights, locked expert RAM and initial session are ready,
   allocate a 512 MiB startup guard and the requested owned prompt workspace.
   Write the workspace buffers and guard, then synchronize before cache sizing.
2. Size the automatic expert cache from the remaining device memory. Charge
   only the existing 64 MiB allocator margin for prefill in this calculation;
   charging the full workspace estimate again would double-count its memory.
   The configured 3072 MiB reserve still covers other late startup allocations.
3. After filling the expert cache and constructing the residency table, attach
   that table to the already initialized Prefill object. Do not allocate a
   second workspace or run the ordinary smaller-chunk fallback.
4. Finish allocating the verifier, MTP and other startup buffers, then release
   the 512 MiB guard. Refuse startup if final free memory is below 512 MiB.

The requested chunk must remain equal to the allocated capacity. Failure to
allocate it aborts startup with a diagnostic. This protects the capacity at
startup; it cannot prevent another application consuming VRAM later. Real
request chunks may still be shorter at cache boundaries or the prompt's end.

This initial mode requires native safetensors, single-GPU serve, automatic
expert caching, an explicit prefill chunk and no borrowing, batching, peer,
remote caches, pipeline or segmented elastic-cache mode. Unsupported combinations fail
explicitly. The existing elastic KV mechanism remains enabled and still
takes space from the expert cache as context grows. All experts retain their
locked RAM copies; the cache uses the existing CPU/GPU providers. Weights,
KV precision, forward arithmetic, sampling and checkpoint logic are unchanged.

The environment switch defaults off. The user's desktop profile opts in and
selects `strata-efficiency14.exe`; its other values remain unchanged, including
262144 context, 4096 prefill, INT8 KV, two MTP drafts, GPU vision, official
thinking sampling, wildcard Host headers and loopback port 8880. The desktop
shortcut still points at `START-NATIVE-262K.bat`. The active process keeps the
configuration it loaded before this change.

## Initial build checks

- Release CUDA SM120 / MSVC engine builds successfully. HIP and SYCL toolchains
  are unavailable and were not built.
- Binary `--help` exits 0; Python benchmark tools compile; source diff checks pass.
- Desktop config differs from its preceding version only in the executable
  and the new opt-in environment variable. Referenced executable exists.
- The updated audit reproduces P13's existing 57-request audited summary exactly.
  This checks the audit tool, not the new GPU allocation path.
- The benchmark's existing busy-process guard refuses to load a model while
  the user's Strata processes are active. The expected refusal is recorded
  separately from inference acceptance.

`validation.json` records hashes, config checks and the pending runtime status.
That initial file is retained as build-time history. The subsequent GPU pass
is recorded in `runtime/validation.json` and `runtime/audited-summary.json`.

## Private acceptance procedure

Executed after the user explicitly released resources (actual output directory:
`logs/p15-prefill-first-validated`):

```powershell
.\.venv-native\Scripts\python.exe tools/native_visual_performance.py --config config/native/rtx5090-262k-mtp2.json --prefill-first --output logs/p15-prefill-first-runtime --rounds 3 --new 128
.\.venv-native\Scripts\python.exe tools/native_visual_performance_summary.py logs/p15-prefill-first-runtime/results.json --out logs/p15-prefill-first-runtime/audited-summary.json
```

This uses the same efficiency14 binary for reserve-only 3072, reserve-only
3584, and prefill-first 3072 profiles. Each starts its GPU visual encoder first,
as the desktop server does. The three rounds include cold/warm 8K and 24K,
short conversation continuation and 8K including an image: 63 requests total.
The audit requires 4096 capacity in the 3584 control and prefill-first runs,
at least 512 MiB final headroom in prefill-first, zero expert source reads,
exact repeat/own-cache outputs, and exact outputs between the two 4096 modes.
Reserve-only 3072's actual capacity is recorded rather than assumed.
Image preparation, prefill, decode and TTFT are reported separately. This
finite suite does not replace the earlier full 262K/cache-restore acceptance;
long-context acceptance of the changed cache sizing remains pending too.

## Measured acceptance

Three fresh engines per profile, RTX 5090 32 GB / Ryzen 9950X3D / 96 GB,
Windows / CUDA 13; same efficiency14 SHA256 in every run. GPU vision loaded
first, official thinking sampling, seed 9950, 128 generated tokens per request.
Startup is excluded. All 63 requests met their output limit; own-cache and
repeat outputs match, and outputs match across the allocation order. Every
post-residency expert/MTP source counter is zero. All profiles actually used
4096 capacity; the prefill-first profile ended with 3286 MiB free each time.

| Allocation / reserve | 8K text prefill | 24K text prefill | Image 8K prefill | 24K decode |
| --- | ---: | ---: | ---: | ---: |
| Reserve only / 3072 MiB | 2744 | 3095 | 2839 | 114.2 |
| Reserve only / 3584 MiB | 2733 | 3068 | 2824 | 113.5 |
| Prefill first / 3072 MiB | 2661 | 2979 | 2761 | 104.8 |

Values are three-run medians in tokens/s for the cold requests. The fixed
workspace is correct, but the old reserve is conservative after the allocation
order changes: cache slots fall from 4998 to 4055, with about 3.2 GiB still free.
This is a correctness/capacity pass, **not a speedup**. A smaller reserve needs
its own measured acceptance before changing the desktop profile. Long-context
KV growth and disk restore on this allocation policy remain pending here.
