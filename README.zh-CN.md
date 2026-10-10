# strata-safetensors

**在单张 GPU 和系统内存上，直接运行原始 NVIDIA NVFP4 safetensors 格式的 Qwen3.8-Flash-Next。**

[English](README.md) · [实现与验收](docs/SAFETENSORS_NATIVE.md) · [测速记录](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) · [上游 Strata](https://github.com/Niko1221/Strata)

本项目把 Strata 已有的前向计算、专家调度与会话缓存接到原始 Hugging Face
模型目录。保留专家的 NVFP4 编码和缩放参数，将专家放入锁定内存，热专家缓存在显存，
ngram 表按需从 SSD 读取。主模型不需要先转换成 GGUF，也不生成必须持久保存的专家包。

当前实际验证的平台是 **Windows + RTX 5090 32 GB + Ryzen 9950X3D + 96 GB RAM**。
主分支为 `main`，引擎程序和命令仍叫 `strata`。

## 当前能力

| 能力 | 实现范围 |
|---|---|
| 原始权重 | 直接读取 NVIDIA NVFP4 safetensors、BF16 dense 权重、FP8 ngram 数据和原生 MTP 权重。专家仅做布局适配，保留编码与缩放参数。 |
| 专家常驻 | 全部主模型专家保留锁定 RAM 副本，包括已进入 GPU 缓存的专家。加载结束后封闭专家与已加载 MTP 权重的源文件读取。 |
| CPU/GPU 协同 | 热专家在 GPU 计算；冷专家可用 CPU AVX-512 计算，或传到 GPU。自适应交换更新热专家集合。 |
| Prefill | 专用工作区、W4A8 专家批量计算、BF16 dense 投影及激活余项补偿；优化后的桌面配置先分配工作区，再确定专家缓存大小。 |
| 上下文 | 支持 FP16 / INT8 KV；桌面使用 INT8 和按需增长。已验证 262,000-token 输入及 262,144-token 总窗口的容量与状态复用。 |
| MTP | 直接使用模型原生草稿权重，由主模型验证；可调整草稿长度和置信度门槛，也可完整卸载 MTP。 |
| 多会话缓存 | 前缀复用、会话驻留与切回、A/B/A 切换，以及 KV 和线性注意力状态的磁盘保存/恢复。 |
| 视觉 | 通过已有外部 BF16 编码器接入图片；视觉侧车仍使用 mmproj / 词表 GGUF，主模型保持 safetensors。 |
| 服务 | 网页聊天、OpenAI / Anthropic 兼容 API、流式输出、工具调用。原生推理串行执行，请求排队。 |

这是针对该模型结构的后端，尚不是通用 safetensors 推理框架。

## 为什么速度提升了

- **减少搬运。** 整个专家内存区通过 CUDA 锁页并映射给 GPU，省去额外的主机 staging 复制。
  推理阶段按需读取 ngram 行；主动保存/恢复会话时另有会话文件读写。
- **使用 CPU 和 GPU。** 桌面配置约 75% 冷专家在 CPU 计算，25% 在 GPU 计算；
  比例只针对缓存未命中的专家。CPU 行循环展开、GPU 同专家多 token 权重复用减少重复工作。
- **保证 prefill 空间。** 先实际分配并写入工作区，再分配专家缓存，同时计算 MTP 等后续分配的预算。
  启动容量不足会报错，不再静默缩小工作区。
- **减少运行时等待。** 专家常驻后关闭启动阶段的缓冲权重句柄；Windows 上的配对测量显示明显收益，
  但具体缓存或驱动机制尚未确定。
- **复用会话。** 同时保存 KV 和线性注意力状态，续聊时处理新增后缀，减少完整历史重算。

真正的 **SM120 FP4 Tensor Core** 路径已经接入，可显式选择。
当前桌面默认 decode 使用 NVFP4 权重和 FP32 激活；专家 prefill 使用 W4A8，
dense 投影保留 BF16。此前 FP4 decode 未测出明确的端到端优势，FP4 prefill 会改变输出，
因此未把 FP4 路径设为默认。保留权重编码不代表所有算术逐位相同，也不代表等同于未量化 BF16 模型。

## 有条件的实测结果

以下均为 **2026-10-10，RTX 5090 32 GB / 9950X3D / 96 GB RAM，Windows，
原始 NVIDIA 权重**的历史对照。各行是不同阶段的实验，提升比例不能相乘。

| 实验 | 之前 → 优化后 | 条件 |
|---|---|---|
| 关闭启动权重句柄，prefill | 822 → 2465 tok/s | 同一程序，约 8K 输入，三组交替配对，128-token 完整输出一致。 |
| CPU/GPU 与内核组合，decode | 45.28 → 81.59 tok/s | 三组交替配对，两边都用两枚草稿，128-token 完整输出一致；候选范围 64.57–85.16。 |
| 后期 4096 行配置，文本 prefill | 2968 → 3331 tok/s | 约 24K 文本，视觉编码器已加载，三次启动中位数；首个约 8K 文本请求反而从 2688 降到 2558。 |
| 后期配置，图片请求 prefill | 2741 → 3089 tok/s | 约 8K 输入含 1024 个图片 token，三次启动中位数；总耗时从 4.51 降到 3.94 秒。 |

另一次英文、中文、代码长回答筛查测得 warm aggregate decode **91.93 → 100.82 tok/s**。
每套配置只启动一台引擎，不能当成多轮稳定吞吐估计；该候选使用四枚草稿、0.7 草稿门槛及两枚强制草稿。

**当前温度 0.7、两枚草稿、门槛 0.5 的桌面试用组合尚未重新测速。**
上述成绩只对应归档配置，不构成当前配置或所有任务的速度承诺。

完整命令、全部样本和验收范围见
[P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md)、
[P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md)。

## 构建与运行

新机器请使用 [英文 README 的源码构建步骤](README.md#build-and-run-on-windows)：
需要 Git、Python 3.12+、Visual Studio 2022 C++ Build Tools、CUDA 13、
CMake 3.24+ 和 Ninja。原始模型下载到仓库外，使用配置生成器填入自己的路径。

生成器保留参考 FP16 prefill 设置，不能直接复现全部桌面优化参数；采样参数需在生成配置的
`sampling` 字段中明确设置。视觉还需要另行配置编码器资源。
专家内存区本身占 **63.282 GiB**，其他权重、会话、运行缓冲和操作系统还需要额外内存。
96 GB 是已验证配置；尚未验证 64 GB 下的完整原生常驻方案。

现有桌面部署使用：

```powershell
.\START-NATIVE-262K.bat
```

该脚本读取 [桌面配置](config/native/rtx5090-262k-fast.json)。
配置和本地构建辅助脚本含部署机器的绝对路径，其他机器必须适配。
服务在前台运行，Ctrl+C 停止。网页地址为 **http://127.0.0.1:8880**，
OpenAI 客户端地址为 **http://127.0.0.1:8880/v1**。
可用 `/health`、`/v1/models` 检查就绪状态和模型 ID。
对外监听前需配置 `--api-key`。

| 参数 | 当前桌面试用配置 |
|---|---|
| 总上下文 / 专用 prefill | 262144 token / 4096 行 |
| KV / 视觉 | INT8 按需增长 / 外部 GPU 编码器，每张图最多 1024 token |
| MTP | `--mtp native --spec 3 --spec-min-p 0.5`，最多两枚草稿，无强制最小数量 |
| 采样 | temperature **0.7**、top_p 0.95、top_k 20、min_p 0、presence_penalty 0、repetition_penalty 1 |
| 模板 | Froggeric v22.5，默认开启思考；请求未指定强度时默认 medium |
| 服务 | 127.0.0.1:8880，HTTP Host 头不设白名单 |
| 输出预算 | `fit_max_tokens: true`，把请求输出上限缩至剩余窗口，输入不截断 |

温度 0.7 是用户选择的试用设置。[Qwen 官方思考模式](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)推荐温度 **1.0**，
其模板默认思考强度 **xhigh**。客户端显式参数及网页持久化设置可以覆盖服务默认值。
输出预算适配不改变 ZCode 等客户端自己的压缩策略，客户端上下文也应设为 262144。

完整关闭 MTP 可选 [no-mtp 配置](config/native/rtx5090-262k-no-mtp.json)；
`START-NATIVE-262K-STABLE.bat` 保留较早的 efficiency14 配置。

`START-NATIVE-262K-PARALLEL.bat` 选择可选的**两路文本并发**，仍为 262K 上下文、
4096 行专用 prefill。图片请求独占引擎；并发配置完全不加载 MTP，所有请求直接由主模型解码。
本机并发配置还启用了按会话长度独立伸缩的 KV 和显存内槽位迁移。
内存开销、验收及首字/吞吐测量见[原生并发说明](docs/NATIVE_CONCURRENCY.md)。原桌面入口不变。

## 验收与边界

- 已记录权重布局往返、真实专家 FP64 对照、完整 logits 比较，以及采样、EOS、
  输出上限、取消与恢复检查。
- efficiency17 最终验收覆盖 80 个完成请求和 4 次取消，跨零/两/四/六枚草稿，
  检查对应输出和主状态、切换会话、磁盘恢复。这些是有限回归测试，不是全面质量基准。
- 262K 文档测试证明容量及测试范围内的缓存正确性，不证明通用长上下文推理、
  检索质量或 262K 吞吐。
- 原生路径可通过 `"parallel": 2` 同时解码两路文本请求，prompt 读取仍共用一个入口。
  暂不支持多 GPU、并发槽位 MTP、流水线并行及外部草稿模型；图片请求需要独占生成。
- 本地验证了 CUDA SM120；Linux、HIP、SYCL 和其他硬件尚未验证原生后端。
- 主模型无需 GGUF；视觉仍需 GGUF 侧车。尚未实现原生视觉塔或通用原生一键安装器。

## 文档、来源与许可证

[原生实现详情](docs/SAFETENSORS_NATIVE.md) ·
[历史 NVFP4 / GGUF 文档](README.legacy-nvfp4.md) ·
[原始 Strata README](README.upstream.md)

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) 和
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)，
起点为 NVFP4 fidelity 提交 `d167eb89301a02e0d6299f49d35494ba5096bc10`，
保留原作者署名与许可证。原个人 GGUF 主分支保存于 `archive/gguf-main-20261011`。

截至 2026-10-11，[上游项目提案](https://github.com/Niko1221/Strata/issues/1858)
与 [独立 safetensors 读取接口 PR](https://github.com/Niko1221/Strata/pull/1859)
均为开放状态，完整原生推理尚未合并进上游。

引擎源码使用 [MIT 许可证](LICENSE)；模型与视觉资源遵循各自许可证，仓库不分发权重。
