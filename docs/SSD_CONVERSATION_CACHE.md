# SSD conversation parking (local extension)

This preserves the main conversation when an auxiliary request (for example a
memory-extraction request) uses another prompt prefix. It keeps one inference
session on the GPU and parks inactive conversation state on SSD. It does not add
concurrent inference slots, load another model, summarize history, or truncate it.

## Installed settings

The local desktop launcher uses `strata-ud-iq4_xs.json` with:

```
--conversation-cache-mib 0
--conversation-disk-dir G:\Strata\Strata\conversation-cache
--conversation-disk-mib 32768
--conversation-disk-slots 64
```

The 32 GiB limit includes committed payloads and their indexes; outgoing capture
is admitted against remaining space while the incoming image is protected from
eviction. Files with the oldest activity are removed first. The record cap bounds
token/image metadata in RAM; the byte limit normally decides how many long
conversations fit. Disk admission also keeps 2 GiB of free disk space.

The normal 262144 context, prefill 8192, INT8 KV, MTP, external FP8 ngram, CPU
vision, single concurrency and 33 GiB expert RAM request are preserved. The new
executable lives separately at `engine/fusion-v0402/strata.exe`; the old
executable is retained. Close the existing server and use the same desktop
launcher to activate the new build. Installation does not start/restart it.

The remote dual-2080-Ti launcher uses the same 32 GiB / 64-record SSD policy
at `/home/test/Strata-fusion-20261002/conversation-cache`. It keeps IQ3_S,
262144 context, prefill 4096, INT8 KV, MTP 4, layers 24+24, strict resident
expert RAM and GPU vision. The draft K/V belongs to the last stage. Both
installations retain the automatic cache's per-engine-run lifetime.

## What is saved

Saved state includes main-layer and draft-layer K/V and scales, pooled indexer
rows (including the moving spare row), GDN recurrent/convolution state, PLE
history and previous token reconstruction, indexer tails/dead keys/block
positions, token IDs, image identity hashes, steering mode, and turn checkpoints
with their LRU stamps. Layer-split images are serialized per stage.

Matching uses exact token/image prefixes, including a saved turn checkpoint
when the client re-renders the last answer. There is no dependency on a
client-supplied session ID. A normal continuation that rewinds only the previous
answer does not create a disk snapshot each turn. A rewrite before the prior
prompt boundary, a different steering mode, or restoring another branch does.
Cancelled/incomplete live state is not parked as a completed conversation.

Only switching/rewinding branches performs SSD operations. The decode loop does
not read or write SSD context snapshots. This does not change the model's existing
expert-weight SSD traffic. Switch latency still includes disk I/O, checksums and
GPU transfers; it is not zero, and full-model throughput has not been measured.

## Memory and file lifetime

K/V capture and restore use one aligned 8 MiB staging allocation per open payload.
Windows uses unbuffered sector-aligned file I/O, so multi-GiB K/V does not first
become a host-RAM snapshot or fill the Windows file cache. The running GDN state
still needs a temporary CPU checkpoint (about 118 MiB for this architecture).
Parking borrows existing CPU checkpoints rather than cloning them. Before loading
an incoming checkpoint chain, the outgoing chain is freed; the incoming chain
replaces the checkpoints already needed by the live session. Metadata, checkpoints
and transfer workspaces therefore still need RAM; this is not a zero-RAM feature.

A directory lease prevents two engines using the same folder. Payloads and indexes
are checksummed and published under their final names only after successful
capture/flush. Exact known temporary/cache filenames from an earlier process are
cleaned only under that lease. Other user files are preserved. The files contain
conversation token IDs and model state, and are temporary acceleration data:
normal shutdown removes them; the next engine start removes crash leftovers.
They are deliberately **not reused across engine restarts or model changes**.

Allocation, disk-full, invalid metadata and pre-restore checksum failures become
cache misses (ordinary prefill), not incorrect reuse. All states are validated
before GPU writes. A CUDA transfer failure during restore remains fatal: inference
must not continue from partially restored state. CPU state admission preserves
128 MiB of measured physical-RAM headroom and caps serialized checkpoint state
at 1 GiB. These limits cover the installed six checkpoints, not arbitrary huge
checkpoint-count configurations.
Batch concurrency (`--batch` / `parallel`) and a layer split onto the same GPU
are rejected with SSD parking enabled; use the installed single-concurrency setup.
The automatic SSD tier currently also requires fixed K/V pools; `--kv-grow`
and `STRATA_KV_GROW` are rejected while it is enabled. The local hot-RAM tier
requires synchronous exchanges without `--adapt-async` or `STRATA_EXCHANGE_ROTATE`.

## Upstream integration (2026-10-07)

The engine is now based on upstream v0.1.40; the Python server includes the
v0.1.40.1 hotfixes. The automatic SSD cache keeps its existing directory, quota,
state format and lifetime. Upstream's persistent `/slots/0?action=save|restore`
API is also retained, but is not enabled in the installed configuration. It is a
separate explicitly invoked feature: enabling it does not make the automatic
SSD cache persist across engine restarts. A successful explicit restore resets
the active prompt boundary used by automatic parking.

This integration passed 4,139 standalone GPU snapshot checks (including local
SSD A/B/A and upstream session-file streaming), the 60 local SSD CPU checks,
the upstream memory/cache/validation tests, and the local file-to-RAM expert
promotion tests. These are fixtures, not a long-context full-model benchmark.

## Observation and validation

The Monitor's Conversation cache card identifies SSD storage and shows bytes,
parked/restored counts and evictions. `/metrics` additionally exposes
`conversation_disk_cache`. The engine log uses `conversation SSD cache:` lines
for enabled/parked/verified/restored/skipped events.

On this Windows/CUDA machine, standalone GPU A/B/A fixtures check byte-exact
restoration of all running state, main/draft K/V, draft ring and earlier turn
checkpoints for FP16, INT8, Q4_0 and identity-layout K8V4. These use small fixtures,
not a second full-model service. CPU tests cover multi-chunk nonaligned extents,
image/steering isolation, corruption/truncation, quota, pinned-incoming eviction,
failed publication, directory isolation and orphan cleanup. Server tests cover
SSD metrics separately from RAM metrics. On 2026-10-07 the Linux/CUDA 12.8 build on two RTX 2080 Ti cards also passed
4,259 GPU snapshot checks, including 120 additional checks on two distinct
devices with carved layer ranges, the last stage's draft ring, borrowed
checkpoint chains and direct SSD-backed K/V. All 12 selected C++ test targets
passed there, including 60 SSD CPU checks, file corruption/limits and injected
CUDA-transfer failures. The Linux server suite also passed 453 tests
(8 environment/platform skips). This validates small fixtures, not a full-model
long-context switch or decode-throughput measurement.

A synthetic 512 MiB payload on G: took 478.2 ms to write, 149.6 ms to verify/load
metadata, and 150.1 ms for a second sequential read. That isolated CPU process's
peak working set was 12.4 MiB. This is a storage/staging check, not a 144K-token
full-model switch benchmark and not a decode-speed guarantee.

The local MSVC installation's localized include prefix was previously recorded
incorrectly by Ninja. The build helper probes the compiler's actual output bytes
to repair dependency tracking; all affected conversation sources were rebuilt.

## Upstream integration (2026-10-08)

Official v0.1.40.2 is merged. Shared-prefix sibling requests skip parking, while
SSD restores still preserve the outgoing conversation. Current tests and deployment
are documented in [LOCAL_FUSION_V0402.md](LOCAL_FUSION_V0402.md).
