# 主分支优化说明 / Main-branch optimizations (2026-10-02)

## 中文

本实现基于原作者 v0.1.35，提交 `d9ab8435f654c368c586340d490915f6addf56a3`。
实测设备为 Windows、RTX 5090 32 GiB 显存与 48 GB 系统内存。主分支保留两项内存规划改动，
模型、计算内核、FP8 PLE 读取器、HTTP 服务和会话快照格式沿用上游。

### 独立预填充缓冲

单卡专家缓存填满后，按剩余显存估算独立缓冲的批量。保留配置的显存余量，批量至少等于上游
借用方案才启用；不会主动缩小专家缓存来腾空间。自动缓存配置可能没有足够余量，此时沿用上游
借用路径。多卡与远端缓存路径保持上游逻辑。`STRATA_PREFILL_OWN_AUTO=0` 可关闭本地选择；
显式 `--no-prefill-borrow` 仍保持上游含义。

### 会话与专家 RAM 预算

启用 RAM 常驻专家和会话缓存时，专家 headroom 至少为会话预算 + 最低空闲 RAM + 256 MiB。
已有更大的 headroom 会保留；关闭会话缓存不改变上游预算。该预算不保证其他程序随后分配内存时
仍有同样余量。

本机配置为 `--conversation-cache-mib 2048 --conversation-cache-slots 2
--conversation-cache-min-free-mib 2560`，对应至少 4.75 GiB 的专家 headroom。
会话快照计费、前缀检查点、恢复、增长、隔离和淘汰全部来自上游。2 个槽是上限，不保证任意长度的
2 个会话都能放入 2 GiB。快照仅在 RAM 中临时保存，退出后丢失；未实现 SSD 会话。
服务接收多个请求后串行执行，同一时刻只有 1 条推理。

### 验证与测量

- 内存规划 11 项边界检查与上游会话缓存 Python 测试 30 项通过；CUDA 13 / MSVC Release，sm_120 编译通过。
- 上游服务的排队、断连取消与请求交接 4 项测试通过；该组使用 mock 引擎，没有做真实模型高并发压测。
- IQ3_S、INT8 KV、FP8 PLE 的真实模型 A→B→A 检查通过；启用/关闭会话缓存的输出与恢复主模型状态哈希一致。
  一次 984-token 检查点恢复约 26 ms。
- 相同 2 GiB 会话缓存、4.75 GiB headroom、自动专家缓存、262144 上下文上限下，暖文件的 17944-token
  完整 prefill：官方约 2925、融合约 2919 tok/s。切回原提示复用 17937 tokens。
- 首轮 512-token decode：官方 121.2、融合 112.6 tok/s；反向三轮融合 107.3/146.0/136.7，
  官方 95.7/94.2/128.2 tok/s。波动与顺序影响明显，不能声称稳定解码提速或退步。
- 独立缓冲单独对照使用同一二进制，专家缓存字节预算 5400 MiB，实际 7056 槽、13.39 GiB。
  批量均为 8192；暖长提示借用缓冲耗时 7762.6 ms（2311.6 tok/s），独立缓冲耗时
  6183.9 ms（2901.7 tok/s）。独立缓冲 3805 MiB，显存余量 1536 MiB。算术和 `OK` 答案匹配。
  单轮顺序测试有文件缓存影响，正式配置保持 `expert-cache auto`。

以上速度对照直接调用引擎，不启动独立视觉编码器，不能预测完整桌面服务速度。完整服务另已通过
GPU 视觉编码器启动与 HTTP A→B→A 恢复检查。这里的改动属于内存规划，不代表普遍提速。
精确测量 JSON 保留在本机部署记录中。

---

## English

This implementation is based on upstream v0.1.35, commit
`d9ab8435f654c368c586340d490915f6addf56a3`.
It keeps two small memory planning changes for the local Windows RTX 5090
(32 GiB VRAM, 48 GiB system RAM). It does not change the model, kernels,
FP8 PLE reader, HTTP server or conversation snapshot format.

### Independent prefill buffers

On a single GPU, after filling the expert cache, the planner prices independent
prefill buffers against the remaining free VRAM. It preserves the configured
VRAM reserve and chooses an independent buffer only if its batch is at least as
large as upstream's borrowed buffer. It does not shrink the expert cache to make
room. With the normal auto-sized cache there may be no suitable free space; the
upstream borrowing path then remains active. Multi-GPU and remote-cache paths
remain upstream's. `STRATA_PREFILL_OWN_AUTO=0` disables this local choice for A/B
comparison. An explicit `--no-prefill-borrow` keeps its upstream meaning.

### RAM budget with upstream conversation caching

When RAM-resident experts and conversation caching are enabled, expert headroom
is at least the conversation budget plus the physical RAM floor plus 256 MiB.
A larger explicit headroom is respected. Disabling conversation caching keeps
upstream expert headroom unchanged. This is an allocation budget, not a
guarantee against other applications allocating RAM later.

The local launch configuration uses upstream's `--conversation-cache-mib 2048`,
`--conversation-cache-slots 2`, and `--conversation-cache-min-free-mib 2560`.
These imply at least 4.75 GiB of expert headroom. Snapshot accounting, prefix
checkpoints, restoration, growth, isolation, and eviction all remain upstream's.
Two slots are an upper bound, not a guarantee that two arbitrarily large
conversations fit within 2 GiB. Snapshots are temporary RAM state, lost when the
engine exits. No SSD session cache is implemented.
Overlapping service requests queue behind one active inference request.

### Validation

The local planner's 11 boundary checks and 30 upstream conversation-cache Python
tests pass. Four upstream service tests cover queue attribution, disconnect cancellation,
and request handover using mock engines; they are not a real-model concurrency stress test.
The engine builds with CUDA 13 / MSVC Release for sm_120.
The upstream real-model A/B/A parity harness passes on IQ3_S, int8 KV and FP8 PLE:
output and the restored main-model state hashes match with caching off/on;
one 984-token checkpoint restored in approximately 26 ms.

A sequential direct-engine comparison with the same 2 GiB cache, 4.75 GiB
headroom, auto expert cache and 262144-token limit read a fresh 17944-token prompt
at approximately 2925 tok/s (official) and 2919 tok/s (local), with warm files.
Returning to the same prompt reused 17937 tokens. One 512-token decode measured
121.2 tok/s (official) and 112.6 tok/s (local); one pair cannot establish a stable
speed difference. The normal auto-sized cache used borrowing in both arms.
Three further decode runs in reverse order measured 107.3/146.0/136.7 tok/s
(local) and 95.7/94.2/128.2 tok/s (official). The order-dependent spread reinforces
that these short measurements do not establish a steady decode speed change.
The Python engine harness does not launch the separate vision helper, so these
measurements are not a prediction of the complete desktop service's speed.
The local changes are memory planning choices, not a claim of general speedup.

An isolated same-binary comparison capped the expert-cache byte budget at
5400 MiB (7056 sized expert slots, 13.39 GiB actual VRAM allocation). Both arms
used 8192-token prefill batches. Borrowing read the warm 17944-token prompt in
7762.6 ms (2311.6 tok/s); independent 3805 MiB buffers read it in 6183.9 ms
(2901.7 tok/s), retaining a 1536 MiB VRAM reserve. Arithmetic and the expected
`OK` answers matched. This was one sequential pair with warm-file confounding,
so it supports retaining the option rather than predicting a general speedup.
The default expert cache remains `auto`; it is not capped to obtain this result.

The exact benchmark results are kept with the local deployment report.
