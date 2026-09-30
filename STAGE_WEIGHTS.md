# Dual 2080 Ti stage-owned weights

This design describes the stage-weight and resident-exchange changes tested on
the modified Linux Strata 0.1.27 installation. The published branch retains the
repository's main history. See [the current configuration and full-model
results](docs/RTX2080TI_DUAL.md) for the deployed IQ3_S setup.

## Stage-owned weights

`--stage-weights` is opt-in and requires native `--serve`, `--spec >= 2`, and an
explicit multi-GPU layer split. For this machine, use `--layer-split 24`.
`auto` is deliberately rejected: the existing auto planner chooses the split
after loading full weights. It has not been changed in this patch.

Only `blk.N.*` tensors outside a stage are omitted. Global tensors are retained.
Canonical arena sizing and loading use the same range; omitted tensors retain
their metadata with `stage_resident=false`. NativeDense respects that ownership
without changing GGUF blocks, precision or shapes. Prefill/verifier accesses to
another stage's tensors fail explicitly.

For the initial IQ3_XXS pack, estimated aggregate savings were 3235.36 MiB,
about 3.16 GiB. This is an allocation calculation, not a runtime measurement of
the final IQ3_S configuration. Stage-owned weights alone retain full session
state; [the additional stage-KV option](STAGE_KV.md) partitions QSA allocations.

## Dynamic resident experts

With `--resident-cpu-experts`, an adaptive swap now exchanges the incoming RAM
expert with the outgoing GPU expert of the same layer:

1. Save the GPU victim to a reusable host scratch buffer.
2. Upload the incoming expert into that GPU slot and synchronize the copy stream.
3. Put the victim into the incoming expert's vacated RAM slot and update the RAM
   ownership map. Scratch is bounded by the largest single expert blob.
4. Publish GPU residency before the next verify window or prompt.

This transaction is called between completed verify windows, when the CPU
expert workers and GPU readers have finished. MTP runs with its own expert
weights; the adaptive thread is joined before another target window. Errors
abort the session instead of allowing partially exchanged weights to execute.
The host mapping is unchanged on a failed transfer; device rollback is not
promised, so a failed transfer must never resume inference.

Dynamic resident mode disables expert-slot borrowing for prefill. Reusing those
GPU slots would otherwise destroy experts with no RAM mirror. It retains the
same swap ranking/frequency; D2H copies add work, whose cost needs benchmarking.

Logs report cumulative RAM reads, mmap lookups and exchanges. A mmap lookup is
not proof of physical disk I/O. The original file mapping stays available for
diagnostic/reload paths. No quantization, ngram hash, ngram decoder, tokenizer,
chat template, router arithmetic or sampling policy is changed.

## Validation

Build with CUDA 12.8 and architecture 75. Run CPU checks with:

```
sh tests/run_stage_cpu_tests.sh /path/to/native/model/pack
```

The host-only CUDA adapter is included solely by these CPU tests. It never
appears in the production build's include paths. These tests validate ownership,
production canonical loading, exact bytes, bounds and failed transfers. They do
not validate real CUDA execution, full-model logits, output quality or speed.

Before production activation, compare the candidate with full-weight loading at
the SAME explicit split and prefill size. Check first-load allocations, short
and long prompts, repeated conversation continuations, prefix restore, MTP,
vision, and many adaptive swaps. Isolate stage loading first with adaptation
disabled, then compare resident dynamic exchange under fixed seeds and sampling.
Only after parity/stability checks should prefill sizes be ranked by actual
prefill/decode performance. The final measured configuration uses 4096 and the
separately validated stage-KV and resident-DMA options. NVLink P2P changes are
not part of this implementation.

Rollback consists of restoring the prior executable/configuration backup.
Publishing this source branch does not start or switch a model service.
