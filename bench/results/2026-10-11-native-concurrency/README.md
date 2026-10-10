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
