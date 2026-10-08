# 本机融合 v0.1.40.2 / Local integration v0.1.40.2

## 中文

2026-10-08：在现有 `main` 上融合官方 v0.1.40.2（`e8ca9af`），保留本地
独立 prefill 缓冲规划、专家与会话 RAM 预算协调、热专家进入既有 RAM 槽、
SSD 自动会话缓存及 Monitor 指标。FP8 ngram 使用上游实现。

2026-10-08 清理：移除旧 Atomic Q5 模型省略第 2 个 PLE 分片的特例，
GGUF 分片发现、元数据校验与打包工具恢复为上游 v0.1.40.2 实现。
启动脚本的旧环境变量清理行一并移除。官方独立 FP8 ngram 加载路径不受影响。
本次仅更新源码与脚本，已安装的引擎二进制未重新编译；该删除在下次构建后进入引擎。

上游新增的 MTP 验证窗口投影优化、固定共享前缀 `strata_prefix`、服务端启动
与视觉请求超时、会话和 API 修复一并保留。处理共享前缀与本地 SSD 缓存的
融合冲突时，同一 pinned 前缀的兄弟查询沿用上游的免停车路径；需要从 SSD
恢复另一条会话时，仍先保存当前会话。上游与本地专家源测试均保留。

生产配置保持 UD-IQ4_XS、FP8 ngram、INT8 KV、262144 上下文、8192 prefill、
MTP 4、33 GiB 专家 RAM 请求、CPU 视觉及 8880 端口；RAM 实际分配仍受可用
物理内存约束。免 API key 与任意 Host / Origin 设置保留。SSD 缓存继续为
32 GiB / 64 条记录，仅在本次引擎运行内复用。并发槽及其他新增 opt-in 优化
没有默认启用。

部署目录为 `engine/fusion-v0402`，桌面入口继续使用 `Start-Strata-IQ4XS.bat`。
旧引擎保留在 `engine/fusion-v0401`，配置、启动器及被替换的源码另有部署备份。
验证服务使用临时端口 18880，完成后卸载模型并退出；正式服务由桌面入口启动。

验证结果：

- 服务端 563 项测试完成，6 项因测试条件跳过，其余通过。
- 缓存、快照、SSD 文件、内存规划、专家源等 10 组 CTest 通过。
- 新的交错投影逐位对照与推测解码概率测试共 2 组 CTest 通过。
- 注意力 GPU 对照程序通过：INT8、FP16、Q4_0，含短上下文与稀疏选择边界。
- RTX 5090 上实际加载 IQ4_XS、FP8 ngram、MTP 和 CPU 视觉编码器，确认
  262144 上下文与免 key / 任意 Host、Origin 请求。
- 实际 API 连续完成 A、A 的兄弟问题、B、恢复 A：首次提示 4691 token；兄弟
  问题不停车到 SSD；B 触发停车；返回 A 从 SSD 恢复并复用 4669 token，答案正确。

这次是功能与兼容性验证，未进行旧版/新版交替速度对照，也未填满 262K 上下文。
文档中的历史速度数据不能作为本融合版的性能提升结论。CPU 视觉编码器通过加载
检查，未在这次验证中新增图像理解质量评测。

## English

On 2026-10-08, official v0.1.40.2 (`e8ca9af`) was merged into the existing `main`.
The local independent prefill planner, coordinated RAM budgets, hot expert
promotion into existing RAM slots, automatic SSD conversation cache and Monitor
metrics are retained. FP8 ngram uses the upstream implementation.

Cleanup on 2026-10-08 removes the old Atomic Q5 exception for an omitted second
PLE shard. GGUF shard discovery, metadata validation and packing now match
upstream v0.1.40.2. Obsolete environment-variable cleanup is removed from the
launchers. Official standalone FP8 ngram loading is unchanged. This cleanup
updates source and scripts only; installed engine binaries have not been rebuilt.
The engine change takes effect after the next build.

Upstream MTP projection optimization, pinned shared prefixes (`strata_prefix`),
engine/vision timeouts and session/API fixes are included. Sibling queries from
the same pinned prefix keep upstream's no-parking behavior; restoring a different
SSD conversation still parks the outgoing state first. Both upstream and local
expert-source test suites are kept.

The installed IQ4_XS configuration retains FP8 ngram, INT8 KV, a 262144 context
limit, 8192 prefill, MTP 4, a 33 GiB requested expert RAM budget, CPU vision and
port 8880. Actual RAM allocation is clamped to available physical memory. No-key
access and wildcard Host/Origin settings remain. SSD snapshots are limited to
32 GiB and 64 records within the current engine run. Batch concurrency and new
opt-in optimizations are not enabled by default.

The desktop launcher uses `engine/fusion-v0402`. The previous engine remains in
`engine/fusion-v0401`; replaced source files, configuration and launcher are backed
up. A temporary validation server used port 18880 and was unloaded and stopped
after the checks; the normal service is started from the desktop launcher.

Validation: 563 server tests completed (6 skipped); 10 cache/memory/expert-source
CTest groups and 2 projection/speculation groups passed. GPU prompt-attention
checks passed for INT8, FP16 and Q4_0, including short and sparse-boundary cases.
A real RTX 5090 run loaded IQ4_XS, FP8 ngram, MTP and the CPU vision encoder, then
completed A / sibling A / B / restored A API requests. The initial prompt had
4691 tokens; the sibling did not park to SSD, B did, and restored A reused 4669
tokens with the correct answer. Wildcard Host/Origin and no-key access passed.

This validates functionality, not an old/new speed comparison or full 262K
context quality. Historical throughput figures are not measurements of this
integration. Vision startup was checked; no new image-quality evaluation was run.
