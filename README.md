# Strata · 双卡推理优化 / Dual-GPU inference optimizations

[中文](#中文) · [English](#english)

## 中文

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.40.2**，
本分支为两张扩容至 **22 GiB 的 RTX 2080 Ti**、32 GB 系统内存的 Linux 主机保留并验证双卡改造。
单卡 RTX 5090 的 Windows 版本位于 [`main`](https://github.com/spideytznn/Strata/tree/main)。

### 保留了什么

- **双卡 RAM 专家交换。** 从专家所属 GPU 取回权重，在相关传输完成后提交交换；
  锁页交换缓冲可跨设备使用。回归测试直接调用生产交换函数，检查权重字节和设备归属。
- **严格专家驻留与显存分配。** 采用上游按层裁剪权重、全局专家热度排序和独立 prefill 缓冲；
  保留严格 RAM 补集检查及旧启动参数兼容，避免唯一专家副本被预填充缓冲覆盖。
- **自动 SSD 会话缓存。** 在上游完整状态快照基础上，分块保存两张 GPU 的会话状态，
  支持 A→B→A 切换。当前部署上限 32 GiB、64 条记录，RAM 整会话缓存关闭。
  缓存在本次引擎运行中复用；跨重启持久化使用上游显式会话文件 API。
- **会话监控。** 网页监控显示 SSD 缓存的保存、恢复与淘汰记录。

前缀检查点、计算内核、基础会话状态、服务接口与视觉编码器来自上游。
上游已吸收的双卡交换和全局排序采用上游实现；本分支保留必要的集成、兼容和回归检查。
单卡热专家 RAM 升级代码保留在源码中，当前双卡配置不启用该功能。

### 当前部署与验证

IQ3_S、INT8 KV、262144 上下文上限、4096 prefill、MTP4、24+24 层与 GPU 视觉。
启动保留 4 GiB 系统内存余量；普通 11 GiB 2080 Ti 不能直接照抄容量配置。
服务接收多个客户端请求并排队，同一时刻执行一条推理。
上下文上限是配置值，不表示已经填满 262K 做性能测试。

当前融合与测试见 [v0.1.40.2 记录](docs/REMOTE_FUSION_V0402.md)。
[此前双卡测量](docs/RTX2080TI_DUAL.md)中的速度属于旧版本，不能作为本次升级的提速幅度。
模型文件、机器专用配置和二进制不随源码上传。

### 构建

```sh
git clone --branch strata-2080tix2 https://github.com/spideytznn/Strata.git
cd Strata
./setup.sh --build
```

自定义 C++ 改动需要源码构建。GPU 架构为 SM75，启用
`STRATA_BUILD_CONVERSATION_TESTS=ON` 与 `STRATA_BUILD_FUSION_TESTS=ON` 可构建对应回归测试。
构建和安装参数见 [安装说明](docs/AI_SETUP.md)，接口见 [DETAILS](docs/DETAILS.md)，
SSD 缓存限制见 [SSD_CONVERSATION_CACHE](docs/SSD_CONVERSATION_CACHE.md)。

感谢原作者 [Niko1221/Strata](https://github.com/Niko1221/Strata)、Qwen、量化作者及 llama.cpp / ggml。
源码采用 [MIT 许可](LICENSE)，模型与依赖遵循各自许可。

## English

This branch integrates [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.40.2**
for a Linux server with two modified **22 GiB RTX 2080 Ti** cards and 32 GB RAM.
The Windows RTX 5090 integration is on [`main`](https://github.com/spideytznn/Strata/tree/main).

- **Dual-GPU RAM expert exchanges:** copy weights from the owning GPU, finish the relevant transfers before
  committing ownership, and use portable pinned exchange buffers. Regression tests call the production helper.
- **Strict expert residency:** retain strict RAM complement checks and compatible launcher arguments while using
  upstream layer weight trimming, global expert ranking and dedicated prefill buffers.
- **Automatic SSD conversation caching:** stream upstream full-state snapshots for both GPUs to SSD for A→B→A
  switches. The deployment allows 32 GiB and 64 records, with whole-conversation RAM parking disabled.
  This cache lasts for the current engine run; upstream explicit session files provide persistence across restarts.
- **Monitoring:** expose SSD parking, restoration and eviction activity in the web monitor.

Prefix checkpoints, kernels, base conversation state, APIs and vision support come from upstream.
Dual-GPU exchange and global ranking changes already adopted upstream use the upstream implementation, with
local integration and regression checks retained. Single-GPU hot RAM promotion is not enabled on this server.

The installed configuration keeps IQ3_S, INT8 KV, a 262144-token context limit, 4096 prefill, MTP4, a 24+24 layer
split, GPU vision and a 4 GiB RAM headroom. Standard 11 GiB cards need different capacity settings.
Requests are queued and one inference runs at a time. The configured context limit is not a full-262K benchmark.

See the [v0.1.40.2 validation record](docs/REMOTE_FUSION_V0402.md).
[Earlier measurements](docs/RTX2080TI_DUAL.md) describe older versions, not a measured speedup from this upgrade.
Model files, machine-specific configuration and binaries are excluded from the source repository.

Use the clone/build commands above to include the custom C++ changes. SM75 is the target architecture;
`STRATA_BUILD_CONVERSATION_TESTS=ON` and `STRATA_BUILD_FUSION_TESTS=ON` enable regression targets.
See [setup](docs/AI_SETUP.md), [API details](docs/DETAILS.md), and [SSD cache scope](docs/SSD_CONVERSATION_CACHE.md).

Thanks to upstream Strata, Qwen, quantization authors and llama.cpp / ggml. Code is [MIT licensed](LICENSE);
models and dependencies retain their own licenses.
