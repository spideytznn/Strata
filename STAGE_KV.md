# Stage-owned QSA state

`--stage-kv --stage-weights --layer-split 24` allocates QSA KV/indexer state
only for the layers that each GPU executes. It requires native serving with an
explicit multi-GPU split and at least one QSA layer per stage. Without the new
flag, the original allocation layout remains in use.

The global QSA indices are unchanged: for a 24/24 split, CUDA0 owns indices
0–5 and CUDA1 owns 6–11. Unowned entries contain no device allocations. Each
device retains its own RoPE table. GDN state, verifier scratch, PLE/ngram,
quantization, MTP settings and expert adaptation policy are unchanged.

Prefill, MTP and verifier scratch obtain context/format/RoPE metadata from the
first **owned** QSA state. Sequence reset, checkpoint save/restore and diagnostic
state readers skip unowned QSA entries. Whole-model token execution helpers
reject partial state; staged prefill/verification validate ownership before use.

## Validation

Build with `-DSTRATA_STAGE_KV_TESTS=ON` and run `stage_kv_test` for host-only
layout checks. `stage_kv_test --gpu 0` (or 1) also checks device carving, guard
bytes, zeroing, page-table identity and per-device RoPE ownership. GPU tests
should run only when the device has enough free memory.

For model parity, `STRATA_STAGE_KV_RESERVE_FULL=1` keeps the original allocation
budget while using the partial state layout. Compare flag off vs flag on with
this diagnostic using the same expert cache placement, prompts and sampling.
Remove the diagnostic for memory/performance measurements. It is never needed
in the normal configuration.

Before deployment, validate full-model generation, multi-chunk prefill,
continuation/checkpoint restoration and vision, then measure warmed long decode
under identical settings. Allocation savings alone do not establish faster decode.

This source is based on the pre-expert-parallel layer-split branch. It does not
restore the subsequently reverted expert-parallel helper implementation.
