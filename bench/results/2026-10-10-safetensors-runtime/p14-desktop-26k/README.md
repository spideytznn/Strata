# Desktop request: 26K in 12.89 seconds

RTX 5090 32 GB / Ryzen 9950X3D / 96 GB, Windows, 2026-10-10.
This is the user's real desktop request, not another controlled benchmark.
The server remains running; it was not restarted or given competing GPU work.
Only numeric runtime diagnostics are archived, without the user's prompt.

| Measurement | Observed |
|---|---:|
| Prompt tokens / reused | 26,328 / 0 |
| Engine prompt time | 12,890 ms |
| Prefill rate | 2,042.5 tok/s |
| Generated tokens | 11,905 |
| Decode time / rate | 170,202 ms / 69.9 tok/s |
| Accepted / proposed drafts | 5,864 / 12,080 |
| Configured / allocated prefill | 4,096 / 3,072 |
| VRAM reserve | 3,072 MiB |
| Expert source bytes | 0 |

## Allocation explains why the private result did not transfer

The startup reports 2,819 MiB needed for the requested 4096-row workspace,
plus a required 512 MiB margin, with only 3,315 MiB available. The feasibility
check therefore selects 3072. The displayed MiB figures imply a shortage of
about 16 MiB; they are rounded, so they do not establish the exact byte gap.
The old `prefill allocation: requested=3072` refers to the value passed to
`Prefill::init` after this fallback, not the original command-line setting.

All three current-visual runs in P13 actually allocated 4096. Their ~24K
median of 3109 tok/s is valid for those runs, but does not establish that a
3072 MiB reserve always preserves that capacity in a desktop startup. This
real request is also a different input and a much longer generation; its
decode result must be reported separately from P13's 128-token samples.

## Where the prompt time went

There are ten batched chunks totaling 26,323 tokens, followed by the short
assistant header. The two batched segments contain 22,324 and 3,999 tokens.
Their tails contain 820 and 927 tokens and take roughly 1.115 and 1.072 s.
These are observed costs, not proof of unnecessary computation over every
allocated row. The prefill code uses each chunk's actual token count.

Prompt segments are separated at cache checkpoints in `generate.cpp`; root,
turn, optional message and pinned checkpoints can all introduce boundaries.
The existing log does not label the boundary reason. Removing these splits
without a replacement would sacrifice reusable checkpoint states.

The archived chunk times, callbacks and explicit PLE waits are summed in
`observation.json`. Most ngram reads overlap computation; the recorded prompt
wait is 37 ms. Cumulative ngram I/O counters include earlier requests and
decode, so they cannot be assigned wholesale to this prompt's 12.89 s.
No expert source reads occurred.

## Staged follow-up, not a deployed speed fix

`strata-efficiency13.exe` builds successfully for CUDA SM120; its `--help`
check exits with status 0, and `git diff --check` passes. It changes only
diagnostics: original configured and selected prefill, shortage including
headroom, and numeric checkpoint boundary flags. Allocation decisions, cache
behavior, weights and computation are unchanged. HIP and SYCL were unavailable
and were not built. The binary has not run model inference alongside the
user's active service.

`config/native/rtx5090-262k-vision-headroom-test.json` changes only the binary
to that diagnostic build and the reserve from 3072 to 3584 MiB. This is an
unmeasured candidate, not the desktop launcher profile. The extra 512 MiB is
intended to protect workspace capacity, but leaves fewer hot experts on the
GPU and could affect decode. It needs a fresh visual startup and paired
prefill/decode/cache tests before selection. No throughput gain is claimed.

The desktop remains on efficiency12 with its measured configuration. A
robust 4096 allocation and the small-tail cost remain open optimization work.
