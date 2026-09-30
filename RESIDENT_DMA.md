# Pinned resident experts

`--pin-resident-experts` requires `--resident-cpu-experts` and opts into the
existing portable, mapped CUDA host allocation for the compact RAM complement.
Only experts absent from all GPU caches occupy this arena. The RAM capacity and
expert formats are unchanged. Allocation retains the available-memory/headroom
check and reports failure if the requested pinned allocation cannot be made.

Prefill can DMA these bytes directly, bypassing the pageable-to-pinned staging
copy. Dynamic swaps retain the existing transaction: save the GPU victim,
upload the incoming RAM expert, synchronize both transfers, copy the victim into
the vacated RAM slot, then publish new ownership. The caller still excludes
source readers during the transaction and publishes GPU residency before the
next verification window. No model arithmetic or ngram processing changes.

The option also exposes mapped aliases to the existing decode PCIe dispatcher.
Use `--pcie-frac 0` to preserve CPU execution for decode cache misses when testing
the prefill-only change. The tested configuration uses this setting and
`STRATA_PREFILL_RING=96`, with 4096 prefill and 262144 maximum context.

Pinned memory cannot be paged out. Preserve operating-system headroom; this is
not a way to increase physical RAM capacity. Monitor memory after requests and
conversation checkpoints, not only immediately after loading. Keep
`--no-prefill-borrow` with a dynamic RAM complement because GPU-only experts
have no second RAM copy.

Build the focused integration test with `-DSTRATA_RESIDENT_EXPERT_TESTS=ON`,
then run `resident_expert_test`. It checks full expert bytes, mapped aliases,
ownership, invalid exchanges, and cleanup on every visible GPU, with both
pageable and pinned RAM. Host allocations made on GPU 0 are also read on the
other GPU. The test uses a temporary synthetic pack, not model weights.
