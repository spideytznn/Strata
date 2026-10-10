# Native workspace and speculation tuning

RTX 5090 32 GB / Ryzen 9950X3D / 96 GB, Windows, CUDA 13, 2026-10-10.
The accepted desktop build was pushed at `f72d339d` before this work.
Tests use private stdin engines with the existing GPU visual encoder loaded
first. No HTTP server starts. Original safetensors and the Q4XL deployment
remain read-only inputs. Expert and MTP source reads are sealed after loading.
Source counters describe requested payload reads, not physical SSD telemetry.

## Changes under test

- Keep the prefill-first dedicated workspace and INT8 KV. Compare actual
  4096, 6144 and 8192 capacities, with a smaller explicit reserve after the
  workspace has already been allocated. A larger workspace leaves fewer hot
  GPU experts; capacity alone does not establish the best decode setting.
- Enable the existing opt-in `STRATA_MTP_BATCH_BF16=1` KV batch path. It uses
  original BF16 weights and the existing FP32 activation decomposition, without
  Q8 projection conversion. The main model's weights and sampling stay intact.
- `STRATA_NATIVE_WARM_WEIGHTS=1` optionally reads original GPU weight buffers
  into the existing disposable 512 MiB startup guard before READY. It runs no
  forward pass and changes neither weights nor session state. Timings measure
  whole profiles; the startup read alone has not been isolated as a speedup.
- Native `--mtp-draft-vocab FILE` explicitly selects unchanged BF16 head rows.
  The full main head still verifies proposals. Missing, malformed, duplicate
  and out-of-range IDs fail validation before GPU gathering. Native defaults
  retain the full head and never discover a sidecar vocabulary automatically.
  `data/native-draft-vocab-cjk.bin` contains 125169 IDs, retaining the older
  base plus all decoded CJK rows, UTF-8 fragments, byte fallback and added
  tokens from the original HF tokenizer. Its manifest records inputs and hashes.
- `--spec-min-drafts N` optionally retains the first N guesses before applying
  `--spec-min-p` to deeper drafts. Default zero preserves the existing policy.
  It is limited to serial MTP, obeys the configured maximum, and retains the
  ordinary one-token commit path. `STRATA_ONE_TOKEN_COMMIT=0` stays set.
- The prefill-first cache residency recheck also preserves the late MTP head
  allocation. Initial pricing already counted it, but the WDDM shrink loop
  previously checked only the operator reserve. A subset head could consume
  about 611 MiB of that remainder after cache writing. The corrected build is
  `strata-efficiency17.exe`; preceding screens are retained as discovery data.

These features are opt-in. Bare CLI and GGUF defaults are unchanged. No new
CUDA kernel or further model-weight quantization is introduced. CUDA SM120
Release and the vocabulary validation test pass; HIP and SYCL toolchains are
unavailable on this host and were not built. See [build evidence](build/validation.json).

## Discovery screens

These are single fresh-engine screens, not the final repeated comparison.
All use official thinking sampling, seed 9950 and fixed inputs. Token rates
exclude startup. First requests can include graph capture and paging work;
the speculation audit ranks warm requests separately.

- [Capacity screen](capacity-screen/audited-summary.json): 21 requests, 128
  tokens each, actual capacities 4096/6144/8192, 1024 MiB reserve. Own warm
  outputs and repeats are exact; sampled outputs can differ across geometry.
- [BF16 draft batch screen](bf16-batch-screen/audited-summary.json): 14 requests,
  128 tokens each, 4096/8192. All outputs, including the continuation, exactly
  match the corresponding capacity screen with draft batching off. The single
  24K prefill samples are 3325/4336 tok/s respectively.
- [Full-head MTP screen](mtp-screen/audited-summary.json): 120 requests, 256
  tokens each, two/four/six drafts and gates 0/0.2/0.5/0.7. Every output matches
  the two-draft reference. Blind six-draft warm aggregate is 73.16 tok/s versus
  93.59 for two drafts; six drafts with 0.7 recover to 95.49, a small difference.
- [Explicit subset screen](subset-mtp-screen/audited-summary.json): 90 requests,
  256 tokens each, gates 0/0.5/0.7. All outputs also match the prior full-head
  reference. Four drafts with 0.7 rank first at 99.58 warm aggregate tok/s.
  This build predates the late-head residency-budget correction; it does not
  establish performance with the final headroom policy.
- [Draft-floor and CPU/GPU screen](floor-balance-screen/audited-summary.json):
  108 requests, 256 tokens each, English/Chinese/code, first two guesses kept,
  two/four/six drafts, gates 0/0.7, cold-expert GPU fractions 0/0.25/0.5.
  All outputs match the full-head reference. Four drafts with 0.7 measure
  99.89/99.84 warm aggregate tok/s at fractions 0/0.25; 0.5 is slower at 87.52.
  This also predates the late-head budget correction. Hot experts always run
  on the GPU; these fractions apply only to cache misses.

The 318 speculation-screen requests retain all 81408 generated IDs, complete
configurations, binary hashes, timings and telemetry. The reference and
candidate are the same NVFP4 model; no BF16 model oracle is claimed.

## Prompt-geometry quality check

[234 teacher-forced positions per geometry](prefill-quality/validation.json)
compare original NVFP4 weights at 4096 and 8192, using a fixed document followed
by English, Chinese and code continuations. Inputs span 24602–24689 tokens.
All 248320 logits per position are finite, and expert/MTP source reads are zero.

At 8192, top-1 agrees at 232/234 positions and mean KL is 0.00136143. Mean
target NLL changes from 1.17950081 to 1.17627558. English changes by -0.0107892,
Chinese by +0.00341327, and code by -0.00358248. Both top-1 differences occur
in English; Chinese and code top-1 match throughout. These are small measured
geometry differences, not byte-identical main logits or proof of general
accuracy improvement. The reference is the accepted 4096 geometry, not an
unquantized oracle. Raw float dumps remain local; every position's hash and
comparison metrics are archived. Diagnostic timing is not throughput evidence.

[6144 quality check](prefill-quality-6144/validation.json) repeats all 234
positions. Its fresh 4096 reference logits match the previous reference
bit for bit at every position. At 6144, top-1 agrees at 232/234, mean KL is
0.00107806, and mean target NLL changes by -0.00331168. English NLL changes
by -0.0144777, Chinese by +0.00475588, and code by -0.00214799. English and
Chinese each have one top-1 difference; all code positions match. The same
finite-test and reference limitations apply.

## Long-answer screen

[First long-answer screen](long-operator/audited-summary.json): official thinking
sampling, seed 9950, 2048 tokens per request, English/Chinese/code cold and warm,
one fresh engine per profile. All 24 complete outputs (49152 IDs) match the
accepted desktop profile exactly, including own warm reuse. All experts remain
locked and expert/MTP source reads remain zero. The corrected build leaves
1139/1233/1231 MiB free in the three candidates, including the late subset head.

| Profile | Warm aggregate decode tok/s | Against desktop |
|---|---:|---:|
| Accepted 4096, reserve 3072, two drafts, gate 0 | 91.93 | reference |
| 4096, reserve 1024, four drafts, floor 2, gate 0.7 | 100.82 | +9.7% |
| 8192, reserve 1024, four drafts, floor 2, gate 0.7 | 94.71 | +3.0% |
| 8192, same four-draft policy, explicit subset | 96.62 | +5.1% |

Candidates enable original BF16 draft batching and startup weight reads. These
are profile comparisons, not isolated attribution to any one change. A single
engine is a screen; repeated visual/prefill acceptance and independent session
state/snapshot checks are separate gates.

[Two-draft follow-up](long-mtp2/audited-summary.json) uses the same official
sampling, input hashes and 2048-token cap. All 24 outputs (49152 IDs) also
match the accepted desktop reference exactly. With gate 0.5, warm aggregate
decode is 98.01 at 4096, 94.92 at 6144 and 93.89 at 8192. At 6144, increasing
adaptation from every four windows / 96 swaps to every two / 192 swaps raises
the screen to 98.49. These remain single-engine profile screens.

[Combined 6144 follow-up](long-combined/audited-summary.json) uses four drafts,
floor two, gate 0.7 and adaptation every two windows / 192 swaps. Its six
2048-token outputs (12288 IDs) also match the accepted desktop reference
exactly. Warm aggregate is 101.81 tok/s (+10.7%). Together the three long-answer
screens retain 110592 generated IDs. The small extra decode gain over the 4096
candidate does not establish that 6144 is better for visual or short requests.

## Operator controls

`tools/native_config.py` accepts zero through seven MTP drafts, optional
`--spec-min-p` and `--spec-min-drafts`, and a 6144-row workspace. These controls
are independent of target sampling `min_p`; changing the draft gate does not
change the official target sampling preset. Its default generated configuration
is byte-identical to the generator at `f72d339d`. Explicit six-draft / 0.7 gate /
floor two / 6144 arguments and invalid probability/floor combinations pass the
[generator check](build/config-generator-validation.json).

## Repeated visual and prefill comparison

[Three fresh engines per profile](final-visual-pair/audited-summary.json), with
profile order rotated, retain all 63 requests / 8064 generated tokens. Official
thinking sampling, seed 9950, image encoder loaded first, no state hashing or
logit dumps. All caps, own warm reuse, repeated outputs, physical expert locking
and zero expert/MTP source-read checks pass. The two 4096 profiles have exactly
equal complete outputs for all 21 corresponding requests. The 6144 sampled
continuations can differ, so its short decode rates are not an equal-output
comparison. These are 128-token capped performance requests, separate from the
2048-token stability screens and session-state acceptance.

| Cold request | Accepted 4096 prefill tok/s | Candidate 4096 | Candidate 6144 |
|---|---:|---:|---:|
| About 8K text, first request | 2688 | 2558 | 2148 |
| About 24K text | 2968 | 3331 | 4173 |
| About 8K including 1024 image tokens | 2741 | 3089 | 2772 |

Entries are three-run medians. The 4096 candidate gives +12.2% on 24K and
+12.7% on images. Its first 8K text prefill is 4.8% slower; first-request total
wall median also changes from 4.57 to 4.82 s. All samples, including the slow
cold decode, remain in the archive. The 24K total wall median falls from 9.37
to 8.35 s, and image total wall from 4.51 to 3.94 s. Image preparation is 143 ms
versus 149 ms; image TTFT falls from 3.13 to 2.79 s. No universal speedup or
isolated startup-warmup benefit is claimed.

The 4096 candidate actually allocates 4096 rows in all three starts, with
4786/4786/4803 expert slots and 1228/1228/1229 MiB final free VRAM. The accepted
profile has 3927/3948/4022 slots and 3506/3452/3286 MiB free. The 6144 candidate
improves long prefill, but gives less consistent image/short-request benefit
and has measured geometry rounding differences. The 4096 full-head candidate
is the general desktop choice after the full-session acceptance below.

## Full-session acceptance

[Runtime acceptance](runtime-final/validation.json) passes 80 completed requests
(5464 generated IDs) plus four deliberately cancelled requests. Zero, two,
four and six drafts use the final binary, actual 4096 workspace, automatic
expert-cache sizing, the configured GPU encoder, gate 0.7 and floor two. The
zero-draft reference retains MTP weights with `--mtp-max-t 1`; its state and
outputs are the reference, not its diagnostic throughput.

Every completed output and main-model state fingerprint matches zero drafts.
Fingerprint counts are checked explicitly. Greedy English/Chinese/code/math,
natural EOS, caps 1/2/3/7/9, seeded sampling, official thinking sampling up to
512 tokens, A/B/A, own disk continuation/state and cancel/resume all pass.
The visible cancelled prefix equals the uninterrupted prefix, and its resumed
continuation matches the next uninterrupted tokens. Expert and MTP source
reads remain zero, with the whole expert arena physically CUDA-pinned. State
hashing and snapshot work make this a correctness run, not throughput evidence.

## Reproduce on the measurement host

The complete local arguments, environment, tokenizer/template hashes and binary
hashes are retained in each result. Paths refer to the independent checkout and
the original read-only model/vision assets. Build the same CUDA source with:

```powershell
.\tools\build_safetensors_engine.ps1 -Jobs 2 -OutputName strata-efficiency17 -Targets @('strata','native_draft_vocab_test')
```

For the repeated comparison, use `tools/native_visual_performance.py` with
`--config` pointing to `configs/desktop-before.json`, `--profile-configs` naming
that file, `configs/candidate-full4k.json` and
`configs/candidate-full6144-mtp4-adapt2.json`, plus `--rounds 3 --new 128` and a
fresh `--output` directory. These scripts use private stdin
engines and refuse a second running Strata engine. They load the GPU encoder
before automatic cache sizing. Do not run another GPU benchmark or compile
during timing.

The independent auditors are `tools/native_visual_screen_summary.py` (use
`--min-rounds 3`) and `tools/native_operator_summary.py`. The latter accepts
`--reference long-operator/results.json` to locate the archived external
reference for `long-mtp2` and `long-combined`; its SHA-256 must match the recorded
original. Use a fresh `--out` under `logs/` to retain the measured artifacts.

[262K capacity acceptance](capacity-final/validation.json) passes eight requests,
actually mapping 262144 KV cells for 262000 input tokens. The exact code
`ZEBRA-262000` appears in the cold answer. Cold, live-prefix warm and disk-restored
outputs/main states agree; both warm and restored requests reuse 261993 tokens.
All 19 complete resident-expert content/table/alias audits have zero errors,
including byte-for-byte read-back of every resident expert against locked RAM.
Restored draft KV read-back also passes. Expert/MTP source reads remain zero.

The initial full-context RAM snapshot with all prompt checkpoints is 4607 MiB,
exceeding the configured 4096 MiB conversation parking budget; automatic parking
is skipped. Live prefix reuse still works. SAVE retains one checkpoint in a
4237575124-byte file, and restoring it after short A/B/A requests passes. The
four-slot budget does not promise four simultaneous full-262K resident sessions.
This is a repeated-document capacity/cache check, not throughput or retrieval
quality evidence. Large session payloads are excluded from Git.

The desktop launcher now selects `config/native/rtx5090-262k-fast.json` with
this 4096/full-head/four-draft configuration. Its measured args/environment
match `candidate-full4k.json`; only the product log filename differs.
`START-NATIVE-262K-STABLE.bat` retains efficiency14 and the accepted MTP2 config.
No HTTP service was started. The original checkpoint, main deployment and
existing stable executable were preserved.
