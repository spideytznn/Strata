# Three-round visual desktop performance

RTX 5090 32 GB / Ryzen 9950X3D / 96 GB, Windows, 2026-10-10. The user
authorized this benchmark after the allocation-only correction. Every run uses
`strata-efficiency12.exe`, original safetensors, INT8 KV, CPU75, dedicated
4096 prefill, two native MTP drafts, and official thinking sampling:
**temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0,
repetition_penalty 1**, with seed 9950. Each request generates 128 tokens.
The desktop configuration was not changed during this benchmark. No HTTP starts.

## Controlled profiles

- `vision2048`: prior visual profile with 2048 MiB VRAM reserve.
- `vision3072`: current desktop visual profile with 3072 MiB reserve.
- `text3072`: current profile with just the image encoder and --vision removed.

Three fresh engines per profile run in orders old/new/text, text/new/old,
new/text/old. Each visual encoder starts and warms at 1024 image tokens before
expert-cache sizing, matching the server. Input fixture and all sampling
parameters are fixed. Startup and process loading are excluded from request
latencies. The first 8K request includes first-request work; the later 24K
request has a cold conversation prefix but already-used kernels. All raw
samples, telemetry, token outputs, binary SHA256 and config snapshots remain.

## Prefill medians

Rates are **fresh prompt tokens / engine prompt milliseconds**. Image preparation
(encoding, tokenization and embedding-file assembly) is recorded separately.
The image case contains 1024 image tokens plus text, not 8K text plus an image.

| Workload | Input tokens | Visual 2048 reserve | Visual 3072 reserve | Change | Text control 3072 |
|---|---:|---:|---:|---:|---:|
| ~8K text | 8052 | 1990.31 | 2727.37 | 37.0% | 2413.45 |
| ~24K text | 24057 | 2396.04 | 3109.26 | 29.8% | 3096.58 |
| ~8K including image | 8087 | 2066.70 | 2850.55 | 37.9% | n/a |

Actual capacity: old **3072/3072/3072**, current **4096/4096/4096**,
control **4096/4096/4096**. Final startup free VRAM was old **165/137/155 MiB**,
current **782/782/771 MiB**, control **2356/2356/2326 MiB**. In the original
user log the old profile had only 22 MiB free; available desktop memory differs
between runs. The prior profile consistently falls back because its workspace
plus allocator headroom does not fit. This result supports the larger reserve
for the tested visual workload, without proving physical WDDM paging or all
possible causes of slow requests. The text control does not show a meaningful
24K throughput penalty from enabling vision in the corrected profile. The
first 8K samples also include initial request work and are not isolated kernel
measurements; a faster visual sample does not establish that vision speeds text.

Current individual cold samples (tok/s):

- doc8k-cold: 2743.25, 2679.27, 2727.37
- doc24k-cold: 3106.73, 3115.87, 3109.26
- image8k-cold: 2866.41, 2850.55, 2832.38

## Latency, decode and correctness

- doc8k-cold: median prompt 2952.3 ms, prepare 7.3 ms, TTFT 2980.3 ms, decode 90.10 tok/s; 8052 fresh / 0 reused tokens.
- doc24k-cold: median prompt 7737.2 ms, prepare 29.1 ms, TTFT 7785.5 ms, decode 115.54 tok/s; 24057 fresh / 0 reused tokens.
- image8k-cold: median prompt 2837.0 ms, prepare 145.1 ms, TTFT 3022.5 ms, decode 100.59 tok/s; 8087 fresh / 0 reused tokens.
- doc8k-warm: median prompt 134.3 ms, prepare 0.7 ms, TTFT 158.9 ms, decode 105.01 tok/s; 5 fresh / 8047 reused tokens.
- doc24k-warm: median prompt 216.0 ms, prepare 3.1 ms, TTFT 239.1 ms, decode 120.89 tok/s; 5 fresh / 24052 reused tokens.
- image8k-warm: median prompt 122.3 ms, prepare 6.1 ms, TTFT 164.7 ms, decode 105.57 tok/s; 5 fresh / 8082 reused tokens.
- doc8k-followup-continuation: median prompt 346.2 ms, prepare 1.1 ms, TTFT 364.8 ms, decode 85.03 tok/s; 36 fresh / 8179 reused tokens.

The cached repeats need only **5 fresh tokens**; their small tok/s display
is not the long-prompt bulk throughput. The short follow-up uses the preceding
sampled reasoning in its history and therefore is not a controlled input
across different chunk geometries. Compare its latency within each profile.

All **57 requests / 7296 generated token IDs** reach the 128-token cap, report
zero expert file reads, reproduce exactly within the same profile over three
fresh engines, and match their own cold/warm outputs. Pure text outputs also
match exactly when toggling vision with 4096 prefill. Across old 3072 versus
new 4096 effective chunks, sampled output sequences differ despite equal seed;
no same-output decode speedup claim is made. Decode above describes each
profile's actual sampled request. This capped reasoning workload is not a
completed-answer quality evaluation or universal stability proof. Image cache
reuse passes; visual disk-session restore and near-capacity image contexts
were not tested here. Main language-model weights and the Q4XL deployment
remain untouched. No FP4 precision or MTP-depth change was introduced.

## Reproduce

```powershell
.\.venv-native\Scripts\python.exe tools/native_visual_performance.py --config config/native/rtx5090-262k-mtp2.json --output logs/vision-perf-repeat --rounds 3 --new 128
.\.venv-native\Scripts\python.exe tools/native_visual_performance_summary.py logs/vision-perf-repeat/results.json --out logs/vision-perf-repeat/audited-summary.json
```

Use a fresh output directory and release other model processes. Fixtures are
generated from docs/DETAILS.md and a synthetic 1024x1024 red PNG; the frozen
`fixtures.json` records the original document/image SHA256 and full messages.
Input token hashes are checked across controlled cases. Archived text is
normalized UTF8/LF; the source-config SHA256 refers to deployed local bytes.
The post-run harness cleanup uses Vision.shutdown() to remove its temporary
image embeddings; the timed run used close(), with the matching scratch files
cleaned after all engines exited. This cleanup occurs outside measured
request intervals. No core rebuild was required.
