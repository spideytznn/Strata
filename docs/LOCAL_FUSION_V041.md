# 本机融合 v0.1.41 / Local integration v0.1.41

## 中文

2026-10-09：在现有 main 合并官方 v0.1.41（fb58e0dbc8399662c0e47c76578c6e878b14f6cf）。
官方独立目录 G:/Strata/Strata-Official 已在该提交，无需重复下载或编译。

保留本地独立 prefill 缓冲选择、专家与会话 RAM 预算协调、可选的热专家 RAM 升级、
SSD 会话缓存和 Monitor 指标。完整分片检查沿用上游；此前删除的 Atomic 缺片特例
本次随重新构建进入部署二进制。FP8 ngram 仍是上游实现，没有加入 NVFP4 扩展。

新融合引擎位于 engine/fusion-v041/strata.exe，CUDA 13、sm_120、Release 构建，
STRATA_Q6K_EXPERTS 和 STRATA_MMQ_KQUANTS 均开启，IQ4_XS、Q5、Q6 复用同一引擎。
各模型的路径、上下文、KV、MTP、RAM 请求、视觉模式和访问设置按现有配置保留。
视觉程序复用已有可用版本。桌面入口不需要在启动时编译。

验证：CUDA 构建完成；四组 CTest（本地内存规划、专家源、会话快照、SSD 会话文件）通过。
服务端检查结果记录在本机 work/upgrade-v041/server-tests.log。
本次不启动额外的大模型，不中断正在运行的官方 Q4 服务，未进行版本间速度对照，
也未填满 262144 上下文。因此不声明性能提升；当前运行实例在自行重启后使用对应更新。

## English

On 2026-10-09, official v0.1.41 (fb58e0dbc8399662c0e47c76578c6e878b14f6cf) was merged
into the existing main branch. G:/Strata/Strata-Official already has this commit,
so its source and engine need no duplicate download or rebuild.

The local independent prefill planner, coordinated RAM budgets, optional hot
expert promotion, SSD conversation cache and Monitor metrics are retained.
Shard validation follows upstream. The earlier Atomic missing-shard exception
removal is now included in the rebuilt executable. FP8 ngram remains upstream's
implementation; this update does not add NVFP4 support.

The integrated engine is engine/fusion-v041/strata.exe, built in Release with
CUDA 13 for sm_120, STRATA_Q6K_EXPERTS=ON and STRATA_MMQ_KQUANTS=ON. IQ4_XS, Q5
and Q6 share this executable. Existing model paths, context, KV, MTP, requested
RAM budgets, vision modes and access settings are preserved. Existing vision
binaries are reused. Desktop startup performs no compilation.

Validation: the CUDA build and four CTest groups (local memory planning, expert
source, conversation snapshots and SSD conversation files) passed. Server test
results are recorded locally in work/upgrade-v041/server-tests.log. No extra
large model was loaded, and the running official Q4 service was not interrupted.
No comparative speed benchmark or fully populated 262144-token test was run.
An existing process uses its replacement executable after the user restarts it.
