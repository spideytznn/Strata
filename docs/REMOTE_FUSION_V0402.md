# 双 RTX 2080 Ti 融合 v0.1.40.2（2026-10-08）

官方版本：`v0.1.40.2`，提交 `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`。
从现有 `strata-2080tix2` 合并，保留双卡分支历史，不强推覆盖。

## 中文

保留严格 RAM 专家补集、跨设备锁页交换缓冲、调用生产函数的双卡交换测试、
独立 prefill 缓冲，以及基于上游快照的 SSD 会话缓存。上游已有的专家所属 GPU
路由、全局热度排序和按层权重裁剪继续采用上游实现。

合并会话入口时保留上游共享前缀的 `pin_sibling` 判断，同时把 SSD 恢复来源纳入
切换条件，避免共享前缀的不同提问触发无意义的会话保存。文件专家代码同时保留
上游锁页节奏和内存预算修复，以及本地热专家测试；双卡配置不启用单卡 RAM 热升级。

正式入口仍是桌面 Strata IQ3_S，二进制升级到 `engine/fusion-v0402`。
保留 IQ3_S、262144 上下文、4096 prefill、INT8 KV、MTP4、24+24 层、GPU 视觉、
`fit_max_tokens: true`、4 GiB RAM 余量，以及用户原有的免 key 和通配访问设置。
SSD 缓存预算仍为 32 GiB、64 份记录；RAM 整会话缓存为 0。
本次没有复制模型、改 KV 格式或降低上下文。

Linux / CUDA 12.8 / 双 RTX 2080 Ti 上，12 组 C++ 测试通过，包括 4259 项 GPU 快照检查、
4198 项 RAM 缓存检查、60 项 SSD 检查、187 项会话文件检查；生产函数在 pageable/pinned
两种模式下共完成 512 次双卡交换，权重字节一致且没有文件回退。
服务端运行 563 项测试，通过，8 项因平台或环境跳过。

投影与推测采样两组内核测试通过，注意力 FP16/INT8/Q4 数值校验通过。
完整 IQ3_S 加载测试启动后，按用户要求提前停止，以尽快完成部署。
因此没有完成真实对话与会话切换验证，也没有测量本版本端到端性能。

旧引擎和升级前配置保留在远端，可用于回退。正式服务保留停止状态，用户从桌面启动。
启动脚本运行预编译引擎，不在每次启动时编译。

## English

Merged official `v0.1.40.2` into the existing `strata-2080tix2` history without rewriting it.
Retained strict RAM expert residency, portable pinned exchange buffers, regression tests calling the production
dual-GPU swap helper, dedicated prefill buffers and SSD conversation parking based on upstream snapshots.
Existing upstream implementations handle owning-GPU routing, global expert ranking and layer weight trimming.

The merge preserves upstream `pin_sibling` handling while including SSD restoration in source selection, so
questions sharing a pinned prefix do not park a conversation unnecessarily. Upstream pin pacing and memory
budget changes coexist with local hot RAM tests; single-GPU hot RAM promotion remains disabled on this server.

The desktop entry uses `engine/fusion-v0402`, keeping IQ3_S, a 262144-token context limit, 4096 prefill, INT8 KV,
MTP4, a 24+24 split, GPU vision, `fit_max_tokens`, 4 GiB RAM headroom and the user's existing authentication/access
settings. SSD parking stays at 32 GiB / 64 records, with whole-conversation RAM parking disabled.
Model files and KV precision remain unchanged.

On Linux / CUDA 12.8 / dual RTX 2080 Ti, all 12 C++ groups passed: 4259 GPU snapshot checks,
4198 RAM cache checks, 60 SSD checks and 187 session file checks. The production helper completed
512 byte-exact dual-GPU swaps across pageable and pinned modes without file fallback.
The server suite ran 563 tests successfully, with 8 platform/environment skips.

Projection and speculative probability tests passed, as did FP16/INT8/Q4 attention parity checks.
The full IQ3_S smoke test was stopped during loading at the user's request to expedite deployment.
Real conversation switching and end-to-end performance were not measured for this build.

The previous binaries and configuration are retained for rollback. Production is left stopped for desktop startup.
The launcher runs prebuilt binaries; it does not compile on each start.
