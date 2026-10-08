<!-- Local integration: upstream v0.1.41 plus memory planning, hot RAM and SSD sessions. -->
# Strata · 本地推理优化 / Local inference optimizations

[中文](#zh-cn) · [English](#english)

<a id="zh-cn"></a>

## 中文

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.41** 的优化分支，让
**Qwen3.8-Flash-Next** 在个人电脑上更合理地使用显存与系统内存。
我们重点保留经过本机验证的内存规划改动，并沿用上游的计算内核、FP8 ngram、服务接口与会话缓存。

### 分支与设备

| 分支 | 定位 | 实测设备 |
| --- | --- | --- |
| [`main`](https://github.com/spideytznn/Strata/tree/main) | 单卡内存规划、热专家 RAM 升级与 SSD 会话缓存 | Windows，RTX 5090 32 GiB 显存，96 GB 系统内存 |
| [`strata-2080tix2`](https://github.com/spideytznn/Strata/tree/strata-2080tix2) | 双卡权重分配、RAM 专家常驻与动态交换（基于 v0.1.35） | Linux，两张扩容至 22 GiB 的 RTX 2080 Ti，32 GB 系统内存 |

双卡测试使用扩容卡，普通 11 GiB RTX 2080 Ti 的容量与性能需要另行验证。
两个分支的优化和测试范围各自独立。

### 我们优化了什么

**主分支保留四项改动：**

1. **独立 prefill 缓冲选择。** 填满专家缓存后，根据剩余显存估算独立预填充缓冲。
   保留配置的显存余量，且批量不小于上游借用缓冲方案时才启用。无需移出专家来腾空间；
   显存不足时自动沿用上游借用路径。正式配置仍使用 `--expert-cache auto`。
2. **专家与会话的 RAM 预算配合。** 启用 RAM 常驻专家和多会话缓存时，专家加载至少为
   会话缓存预算、最低空闲 RAM 和额外 256 MiB 留出空间，避免两者分别按同一份空闲内存做预算。
   已有更大的专家 headroom 会被保留；会话缓存关闭时，维持上游预算。

3. **热专家进入已有 RAM 槽。** 将 SSD 上频繁使用的专家替换进同层较冷的 RAM 槽，
   保持 RAM 容量与 CUDA 映射边界，并在 GPU 使用完成后再交换。
4. **自动 SSD 会话缓存。** 基于上游快照，将完整会话状态分块保存到 SSD，支持 A→B→A
   会话切换；当前配置为 32 GiB、最多 64 条记录，仅在本次引擎运行内复用。

本机关闭 RAM 整会话缓存，保留上游前缀检查点、会话匹配与服务接口。FP8 ngram
使用上游实现；已移除旧 Atomic 模型缺少 PLE 分片的特殊兼容，恢复上游完整分片检查。
当前部署与验证范围见 [v0.1.41 融合说明](docs/LOCAL_FUSION_V041.md)。

**双卡分支的改动：** dense 权重按所属 GPU 的层加载；在 RAM 常驻两张 GPU 的专家补集；
动态交换在确认两张卡上传完成后提交专家归属。该分支配合上游的专用 prefill 缓冲与按层状态分配。
详见 [双卡实现与测试记录](https://github.com/spideytznn/Strata/blob/strata-2080tix2/docs/RTX2080TI_DUAL.md)。

### 实测结果

以下性能数据基于 v0.1.35，不代表 v0.1.41 的速度。

主分支：2026-10-02，RTX 5090 / 48 GB RAM，IQ3_S、FP8 ngram、INT8 KV，配置上下文 262144。
以下是顺序对照，文件缓存与运行状态会影响结果。

| 测试 | 对照 | 优化版 | 含义 |
| --- | ---: | ---: | --- |
| 17944-token 完整 prefill，自动专家缓存、暖文件 | 上游 2925 tok/s | 2919 tok/s | 默认配置预填充速度接近 |
| 17944-token 完整 prefill，较小专家缓存、同一二进制 | 借用缓冲 2312 tok/s | 独立缓冲 2902 tok/s | 独立缓冲在有显存余量时值得保留 |
| 切回相同的 17944-token 提示 | — | 复用 17937 tokens，只重读 7 tokens | 上游多会话恢复有效 |

独立缓冲对照使用 `--expert-cache 5400`，引擎实报 7056 槽、13.39 GiB 实际分配；
两组 prefill 批量均为 8192，独立缓冲约 3805 MiB，显存余量 1536 MiB。
这是一次顺序测试，不代表默认配置或所有提示都有同等收益。
解码复测波动较大，**尚未证明主分支有稳定的解码提速**。

双卡分支在上述扩容双卡上，单轮同提示测试中，13K prefill 从旧改造版的 **1040.5** 提升至
**1207.4 tok/s**，1024-token decode 从 **40.3** 提升至 **51.5 tok/s**。
没有清空 OS 文件缓存；详细条件与限制见双卡文档。

主分支已通过 11 项内存规划边界检查、30 项上游会话缓存测试，以及真实模型 A→B→A 状态一致性检查。
完整服务包含 GPU 视觉编码器的启动与 HTTP 会话恢复也已验证。
更多参数、测量和范围见 [主分支优化说明](docs/LOCAL_VARIANT.md)。

### 多会话与并发

当前本机配置接收多个客户端请求并排队，**同一时刻执行 1 条推理**。
上游另提供显式开启的并发槽；本地 SSD 热专家升级暂限单请求路径，不能与 `--batch` 混用。
RAM 多会话缓存加速不同历史之间的切换，不增加同时推理数量。
当前没有应用层队列长度上限或推理限流；已验证排队、断连取消与请求交接，尚未做高并发容量评测。
本机 SSD 自动会话缓存已实现，切换时分块保存完整状态；缓存随本次引擎运行结束而清理。
上游 v0.1.40 新增独立的磁盘会话保存/恢复 API，可跨兼容引擎重启复用，需要主动调用。
详见 [SSD 自动缓存](docs/SSD_CONVERSATION_CACHE.md) 与 [上游会话文件](docs/DETAILS.md#using-it)。

### 安装与配置

```sh
git clone https://github.com/spideytznn/Strata.git
cd Strata
```

使用自定义改动需要从源码编译：Windows 运行 `START-HERE.bat --build`，Linux 运行
`./setup.sh --build`。已有安装调整构建时可加 `--setup`。安装器准备依赖与模型，并按设备构建引擎；
仅使用上游预编译引擎不会包含本仓库的 C++ 改动。
主分支自定义改动的实测平台为 Windows / CUDA；Linux 双卡使用对应分支。

本机当前配置使用 UD-IQ4_XS、FP8 ngram、INT8 KV、MTP4、CPU 视觉与 262144 上下文上限；
上下文上限是配置值，不代表已填满 262K 实测。硬件、后台内存占用与模型大小都会影响可运行配置。
保留自己的模型路径；如选择上游 RAM 整会话缓存，可在 `args` 中按需加入（本机使用上述 SSD 配置）：

```text
--conversation-cache-mib 2048
--conversation-cache-slots 2
--conversation-cache-min-free-mib 2560
```

`STRATA_PREFILL_OWN_AUTO=0` 可关闭本地独立缓冲选择，用于对照。
安装步骤见 [AI_SETUP](docs/AI_SETUP.md)，模型选择见 [MODELS](docs/MODELS.md)，
服务接口与完整参数见 [DETAILS](docs/DETAILS.md)。模型、机器专用配置与编译产物不随源码上传。

### 致谢与许可

原作者 [Niko1221/Strata](https://github.com/Niko1221/Strata) 提供基础引擎、计算内核、服务层、
前缀检查点与 RAM 多会话缓存。FP8 ngram 使用上游读取与打包实现，本仓库没有把这些功能标为自研。
模型来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，量化版本来自
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)；
底层还使用 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)。
源码遵循 [MIT License](LICENSE)，模型和依赖遵循各自许可。

---

<a id="english"></a>

## English

An optimized fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.41** for running
**Qwen3.8-Flash-Next** on personal computers with a practical balance of VRAM and system RAM.
We retain measured memory-planning changes while using upstream compute kernels, FP8 ngram support,
service APIs, and conversation caching.

### Branches and hardware

| Branch | Focus | Tested hardware |
| --- | --- | --- |
| [`main`](https://github.com/spideytznn/Strata/tree/main) | Single-GPU memory planning, hot RAM experts and SSD sessions | Windows, RTX 5090 with 32 GiB VRAM, 96 GB system RAM |
| [`strata-2080tix2`](https://github.com/spideytznn/Strata/tree/strata-2080tix2) | Dual-GPU weight placement, resident RAM experts, and dynamic exchange (based on v0.1.35) | Linux, two RTX 2080 Ti cards modified to 22 GiB each, 32 GB system RAM |

The dual-GPU measurements use modified cards. Capacity and performance on ordinary 11 GiB RTX 2080 Ti
cards require separate validation. Each branch has its own implementation and test scope.

### What we optimized

**Main retains four changes:**

1. **Independent prefill buffer selection.** After filling the expert cache, the planner prices independent
   buffers against remaining free VRAM. It preserves the configured VRAM reserve and selects them only if
   the batch is at least as large as upstream's borrowed-buffer option. It does not evict experts to make
   room. If free VRAM is insufficient, upstream borrowing remains active. Production keeps `--expert-cache auto`.
2. **Coordinated expert and conversation RAM budgets.** With resident RAM experts and conversation caching
   enabled, expert loading reserves at least the conversation budget plus the minimum free-RAM floor plus
   256 MiB, so both allocations do not budget against the same free memory independently. Larger existing
   expert headroom is respected. Disabling conversation caching preserves upstream headroom.

3. **Hot experts move into existing RAM slots.** Frequently used file-tier experts replace colder experts
   in the same layer after GPU use completes, preserving RAM capacity and CUDA mapping boundaries.
4. **Automatic SSD conversation caching.** Upstream snapshots are streamed to SSD for A→B→A conversation
   switching. The installed limit is 32 GiB and 64 records, reusable only within the current engine run.

The local deployment disables full RAM conversation snapshots and retains upstream prefix checkpoints,
matching and service APIs. FP8 ngram uses the upstream implementation. The old Atomic exception
for an omitted PLE shard has been removed, restoring upstream checks for complete shard sets.
See the [v0.1.41 integration notes](docs/LOCAL_FUSION_V041.md) for deployment and validation scope.

**The dual-GPU branch** loads dense weights for each GPU's assigned layers, keeps the combined expert
complement resident in RAM, and commits dynamic-exchange ownership only after both GPUs confirm uploads.
It uses upstream dedicated prefill buffers and layer-owned state allocation. See the
[dual-GPU implementation and measurements](https://github.com/spideytznn/Strata/blob/strata-2080tix2/docs/RTX2080TI_DUAL.md).

### Measurements

The performance figures below were measured on v0.1.35 and do not describe v0.1.41 performance.

Main: 2026-10-02, RTX 5090 / 48 GB RAM, IQ3_S, FP8 ngram, INT8 KV, configured context limit 262144.
These are sequential comparisons affected by file caching and runtime conditions.

| Test | Reference | Optimized | Interpretation |
| --- | ---: | ---: | --- |
| Full 17944-token prefill, auto expert cache, warm files | Upstream 2925 tok/s | 2919 tok/s | Similar default prefill speed |
| Full 17944-token prefill, smaller expert cache, same binary | Borrowed buffers 2312 tok/s | Independent buffers 2902 tok/s | The independent option is useful when free VRAM permits |
| Return to the same 17944-token prompt | — | 17937 tokens reused, only 7 reread | Upstream conversation restoration works |

The independent-buffer comparison used `--expert-cache 5400`; the engine reported 7056 slots and
13.39 GiB of actual allocation. Both arms used 8192-token batches; independent buffers used about 3805 MiB,
preserving a 1536 MiB VRAM reserve. This single sequential pair does not establish the same improvement
for default settings or all prompts. Repeated decode measurements varied substantially;
**a stable main-branch decode speedup has not been established**.

On the modified dual-GPU machine, one matched-prompt run improved 13K prefill from **1040.5** in the old
custom version to **1207.4 tok/s**, and 1024-token decode from **40.3** to **51.5 tok/s**.
OS file caches were not cleared. Conditions and limitations are documented on that branch.

Main passes 11 planner boundary checks, 30 upstream conversation-cache tests, and real-model A→B→A state
parity checks. Full-service startup with the GPU vision encoder and HTTP conversation restoration also pass.
See [main-branch implementation and validation](docs/LOCAL_VARIANT.md) for detailed parameters and scope.

### Conversations and concurrency

Multiple clients can submit requests and wait in the queue; **one inference request runs at a time**.
RAM conversation caching speeds up switching histories without adding parallel inference slots.
There is currently no application-level queue-length cap or inference rate limit. Queueing, disconnect
cancellation, and request handover are checked; high-concurrency capacity has not been measured.
**Automatic SSD session caching is implemented.** Its files last for the current engine run.
Upstream's explicit session save/restore API is retained as a separate opt-in feature.

### Install and configure

```sh
git clone https://github.com/spideytznn/Strata.git
cd Strata
```

Build from source to use the custom changes: run `START-HERE.bat --build` on Windows or `./setup.sh --build`
on Linux. Add `--setup` when changing an existing installation's build/settings. Setup prepares dependencies
and the model and builds for your GPU. An upstream prebuilt engine does not contain this fork's C++ changes.
Main's custom changes are tested on Windows / CUDA; use the separate branch for the tested Linux dual-GPU setup.

The current local deployment uses UD-IQ4_XS, FP8 ngram, INT8 KV, MTP4, CPU vision, and a 262144 context limit.
The context limit is configured, not a full-262K input measurement. Hardware, background RAM use, and model
size determine what fits. Keep your own model paths and optionally add these engine `args` to the generated config:

```text
--conversation-cache-mib 2048
--conversation-cache-slots 2
--conversation-cache-min-free-mib 2560
```

`STRATA_PREFILL_OWN_AUTO=0` disables the local independent-buffer choice for comparison.
See [AI_SETUP](docs/AI_SETUP.md) for installation, [MODELS](docs/MODELS.md) for model choices, and
[DETAILS](docs/DETAILS.md) for service APIs and all options. Models, machine-specific configs, and build
artifacts are not included in the source repository.

### Credits and license

[Niko1221/Strata](https://github.com/Niko1221/Strata) provides the base engine, compute kernels, service layer,
prefix checkpoints, and RAM conversation cache. FP8 ngram uses upstream readers and packing tools;
these are credited to upstream rather than presented as our additions.
The model is from [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), with quantized versions from
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
Strata also uses [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp).
Source is under the [MIT License](LICENSE); models and dependencies retain their own licenses.
