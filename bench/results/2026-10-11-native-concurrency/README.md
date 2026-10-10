# Native safetensors concurrency acceptance, 2026-10-11

Hardware: NVIDIA RTX 5090 32 GB, AMD Ryzen 9950X3D, 96 GB RAM, Windows,
CUDA 13.0, MSVC 14.44, SM120 release build. Original
`Qwen3.8-flash-next-nvfp4` safetensors weights, native BF16 projections and head.
These runs exercise two text slots; they do not establish support for other
GPUs, HIP/SYCL, multiple GPUs, or more than two native slots.

Implementation and operating limits: [native concurrency](../../../docs/NATIVE_CONCURRENCY.md).
Each `results.json` includes the effective engine config, prompts or token
outputs, protocol/HTTP results and startup information. Raw protocol runs also
record the executable SHA256. Logs contain runtime memory and residency counters.
The PNG is a synthetic red square used to check vision exclusion.

Subsequent inspection found a missing adaptive tick in actual batch decode:
the measured executable accumulates routing heat, but periodic exchange is
scheduled only by solo generation. The recorded measurements remain observations
of that executable; configured adaptation and cumulative promotions do not prove
two-row batch promotions. A correction was compiled as `strata-concurrency3.exe`
(`adaptive-fix-build.log`). The correction passed new target-only GPU acceptance
with 98 real two-row adaptive rounds and 9367 promotions in those rounds;
see [the correction](../../../docs/NATIVE_CONCURRENCY.md#adaptive-exchange-correction).

## Functional results

| Directory | Check | Result |
|---|---|---|
| `smoke-8k` | Initial CUDA build, 8192 context, two overlapping text slots | Passed; 128 tokens per sampling pair |
| `elastic-262k` | 262144 context, external vision encoder loaded, elastic KV | Passed; 22777-token slot growth and later reuse beside a short request |
| `capacity-262k` | Full KV allocation, 261000 / 261007 token prompts | Passed; eight output tokens each matched solo; first slot completed before second admission finished |
| `capacity-overlap-262k` | Exploratory 64-token capacity fixture | Overlap assertion failed: slot 0 finished during slot 1 prefill; token parity passed before that assertion. Not a successful overlap acceptance run |
| `capacity-decode-262k` | 261000 / 261007 token prompts, 256 generated tokens each | Passed; exact solo parity and overlapping decode at full KV allocation |
| `http-262k` | Two-slot OpenAI SSE, three-client queue, disconnect/recovery, image exclusion, Anthropic | Passed; text matched serial reference; image returned red |
| `http-serial-262k` | Same executable with batching disabled, same-day HTTP comparison | Passed |
| `adaptive3-acceptance` | No MTP loaded, vision loaded, 128-token sampling pairs, real batch promotions | Passed; exact solo parity, cancellation/resume, followups, limits and EOS |
| `http3-no-mtp-parallel` | Corrected batch adaptation, target-only HTTP, three rounds per client count | Passed; vision exclusion, disconnect/recovery and Anthropic |
| `http3-no-mtp-serial` | Same target-only executable/config with parallel disabled | Passed; 24 shared completed responses also match the parallel run |
| `optimized5-acceptance` | Independent session KV and device slot copies, no MTP, vision loaded | Passed; two overlapping 261000/261007-token slots, 256 outputs each, 124135-token live-slot continuation and replacement/readmission |
| `http5-long-baseline` | Same `concurrency5` executable, both new flags off, 124135-token/short mixed HTTP | Passed; three rounds per client count, disconnect/recovery, image exclusion and Anthropic |
| `http5-long-optimized` | Same executable and fixture with both new flags on | Passed; all 27 shared completed responses matched flags-off run |
| `http5-short-baseline` | Same executable, both flags off, all short HTTP prompts | Passed; three rounds per client count and all HTTP/residency checks |
| `http5-short-optimized` | Same short fixture with both flags on | Passed; all 27 shared completed responses matched flags-off run |

Greedy, seeded sampling with different per-request temperature/top-k/top-p,
greedy and stochastic repetition/frequency/presence penalties, cancellation and
resume, follow-up cache reuse and one/two-token limits matched solo token IDs.
The capacity run additionally checks early EOS. These are finite regression
tests, not an unquantized-model quality or long-context retrieval benchmark.

No post-residency expert or MTP source reads were reported. DONE expert-file
counters were zero. These are logical source counters, not physical SSD
telemetry; ngram SSD reads are allowed. At full KV mapping, the expert cache
gave back 3166 of 3294 GPU slots in the first capacity run, and 3132 of 3260 in
the final overlapping run, leaving its 128-slot floor. Expert payloads
remained in RAM. Full-length concurrency therefore has a substantial GPU expert
cache cost even when allocation succeeds.

The final run's output intervals were 6.488–24.715 s and 16.532–27.110 s from
batch admission, with 512 total generated tokens. Its full mapping was 262144
cells / 9.54 GiB. Startup/cache differences across runs include WDDM allocation
variation. This repeated-text capacity fixture is not a retrieval or peak-rate
benchmark. Its executable SHA256 is
`538661b7d80e69ed870c791eda41e06720658e32ed6b07abff27dc0d26278dcb`.

`python-tests.log` records 33 passing native admission-gate, inherited parallel
server and run-config tests. `build.log` records the final CUDA build. No HIP or
SYCL build was performed on this machine.
Config-generator smoke checks confirmed that default generation omits `parallel`,
`--parallel 2` emits it, and parallel plus serial-only minimum drafts is rejected.
The native engine also returned code 2 for `--batch-mtp` before weight loading.

## Target-only follow-up and state-copy validation

`no-mtp3-http-summary.json` compares the corrected executable with one/two slots,
MTP entirely unloaded, otherwise the same 262K INT8/dedicated-4096/vision profile.
The three-round medians for 1/2/3 clients were 64.67/58.10/55.19 aggregate tok/s
serial and 61.43/73.27/62.24 with two slots. Two-client last first-content wait
was 1.640 versus 0.420 seconds. This warm-cache screen supports a measured 26.1%
two-client throughput gain, with a 5.0% one-client regression. Startup was excluded
and varied substantially; GPU clocks were not controlled.

`optimized5-acceptance` additionally checks independent KV and direct device
copies with real expert adaptation: 175 actual two-row rounds, 15279 promotions,
zero post-residency expert/MTP source bytes. At full target-only capacity the
three KV groups mapped 9504 MiB (9.28 GiB). With one 131072-cell slot and the
main/other slot at 8192 cells, they mapped 1872 MiB. Two full slots still require
their full KV. The historical 9.54 GiB above also included draft KV.

Live-slot reuse and replacement/readmission matched solo output; the last long
readmission recomputed its prompt, so it is not evidence of a RAM-cache hit.
Repeated-text capacity fixtures do not establish retrieval quality.
`device-copy-bytes-test.log` records 4219 passing CUDA snapshot/mapping checks;
`python-tests-optimized.log` and `python-native-gate-optimized.log` record 37
distinct passing server/config/gate checks. `device-copy-build.log` is the final
SM120 build, SHA256 `257208bff0788021c843205f2f8f222309e9ad1ae4aac536c4a8126ac5359b68`.
`independent-kv-build.log` is the intermediate independent-KV build, not the
accepted final executable. HIP/SYCL were not built or validated.

`optimized5-long-http-summary.json` compares both flags off/on in the exact same
binary, with 124135 formatted long tokens plus short questions, 96 greedy output
tokens, three rounds per client count. Long prompts reuse 124128 tokens after
reference warmups. Aggregate rates for 1/2/3 clients were 42.47/44.33/42.21 tok/s
off versus 47.11/51.97/49.84 on, gains of 10.9%/17.2%/18.1%. Last first-content
latency was 0.587/1.260/4.776 s off versus 0.367/1.273/3.865 s on.
Median slot admission increased 56 to 107 ms including independent mapping/cache
adjustments; restoration decreased 41.9 to 22.05 ms. This is an end-to-end warm
conversation screen, not a cold long-prompt prefill benchmark. Sequential runs,
uncontrolled clocks and adapting expert placement limit the comparison.

`optimized5-short-http-summary.json` repeats the same off/on comparison with
all-short prompts. Aggregate rates for 1/2/3 clients were 65.26/75.00/66.02 tok/s
off and 66.18/78.43/69.75 on (+1.4%/+4.6%/+5.6%). Last first-content latency
was 0.134/0.429/2.679 s off and 0.124/0.376/2.574 s on. Median slot admission
was 63.4 versus 22.8 ms; restoration was 35.1 versus 27.9 ms. All 27 shared
completed responses matched; disconnect/recovery, vision exclusion, Anthropic
and zero post-residency expert/MTP source counters passed. The small differences
are not guaranteed gains. Omit `--long-tokens` in the commands below to reproduce
the short fixture. The separate parallel desktop profile now opts into both
paths; engine defaults and the original serial desktop profile stay unchanged.

Reproduce the final state and mixed-length screens with:

```powershell
tools/build_safetensors_engine.ps1 -Jobs 8 -OutputName strata-concurrency5
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-concurrency5.exe --output bench/results/local-independent-acceptance --context 262144 --vision --limit 128 --adaptive --independent-kv --device-slot-copy --growth-tokens 131000 --capacity-tokens 261000
.venv-native/Scripts/python.exe tools/native_concurrency_http.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-concurrency5.exe --output bench/results/local-mixed-off --parallel 2 --rounds 3 --long-tokens 131000
.venv-native/Scripts/python.exe tools/native_concurrency_http.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-concurrency5.exe --output bench/results/local-mixed-on --parallel 2 --rounds 3 --long-tokens 131000 --independent-kv --device-slot-copy
.venv-native/Scripts/python.exe tools/native_concurrency_report.py --baseline bench/results/local-mixed-off --candidate bench/results/local-mixed-on --output bench/results/local-mixed-summary.json
```

The harness explicitly sets both new flags from its command-line switches,
so omitting a switch disables that path even if the desktop config enables it.
The requested fixture size is approximate: token merges produce 124123 body
tokens (124135 with chat formatting) for `--long-tokens 131000`.

## HTTP latency and throughput screen

262144 context, INT8 KV, dedicated 4096-row prefill, vision encoder loaded,
1536 MiB reserve, solo native MTP (`--spec 3 --spec-min-p 0.5`), target-only batch
decode, adaptive expert exchange enabled. Three measured rounds followed serial
reference warmups. Each client requested 96 greedy target tokens. Median values:

| Clients | Serial aggregate tok/s | Two-slot aggregate tok/s | Serial last first-content latency | Two-slot last first-content latency |
|---|---:|---:|---:|---:|
| 1 | 79.19 | 61.25 | 0.131 s | 0.144 s |
| 2 | 71.34 | 48.05 | 1.438 s | 0.455 s |
| 3 | 76.94 | 71.65 | 2.519 s | 2.744 s |

Aggregate rate includes prompt/admission and HTTP time. Last first-content
latency is the slowest client's time until visible text, not an SSE heartbeat.
The two-client wait improved, while aggregate throughput and the one-client
rate fell. The three-client wait did not improve. Profiles were run sequentially,
cache placement adapted and GPU clocks were not recorded. This is a functional
and latency screen; it does not support a general throughput speedup claim.
Keep the serial launcher for one user's maximum speed.

## Reproduction

Run from the repository root with the local native Python environment. Paths in
the JSON refer to this PC. Output folders must not already exist. The tools use
private child engines and ephemeral HTTP ports, and stop their children on exit;
they do not start the desktop server or write the model.

```powershell
tools/build_safetensors_engine.ps1 -Jobs 8 -OutputName strata-concurrency2
.venv-native/Scripts/python.exe -m unittest serve.test_native_concurrency serve.test_parallel serve.test_runconfig

.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-fast.json --exe build-native-engine/strata-concurrency2.exe --output bench/results/local-elastic --context 262144 --vision --limit 64 --growth-tokens 22000
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-fast.json --exe build-native-engine/strata-concurrency2.exe --output bench/results/local-capacity --context 262144 --vision --limit 32 --capacity-tokens 261000

.venv-native/Scripts/python.exe tools/native_concurrency_http.py --config config/native/rtx5090-262k-fast.json --exe build-native-engine/strata-concurrency2.exe --output bench/results/local-http --parallel 2 --rounds 3
.venv-native/Scripts/python.exe tools/native_concurrency_http.py --config config/native/rtx5090-262k-fast.json --exe build-native-engine/strata-concurrency2.exe --output bench/results/local-http-serial --parallel 1 --rounds 3
```

Raw protocol tests disable adaptive exchange to isolate state parity. HTTP
tests retain the desktop adaptive exchange settings. All tests override the
reserve to 1536 MiB; raw protocol tests also set batch size, context and early
dedicated prefill explicitly. Sampling is supplied by each synthetic request;
the user's saved web sampling defaults are not modified.

The original capacity record used eight output tokens. A 64-token follow-up
still completed the first answer between the second prompt's chunks, as the
inherited scheduler budgets decode for half of each chunk's elapsed time. The
current harness asks for a long explanation, uses 256 output tokens and requires
overlapping output intervals, so full-length simultaneous decode is checked
rather than merely two successful long admissions. Both earlier records are
retained; their outputs must not be used as proof of simultaneous full-length
decode.
