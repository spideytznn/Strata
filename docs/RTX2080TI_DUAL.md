# 双 RTX 2080 Ti / strata-2080tix2

当前 v0.1.40.1 部署与 SSD 缓存验证见 [REMOTE_FUSION_V0401.md](REMOTE_FUSION_V0401.md)。下文保留此前版本的改造和实测记录，性能数据及末尾二进制 hash 不代表当前版本。

本分支融合官方 **v0.1.39**（`6f32ec070f23ced9f50e704d854d775da52591ab`）。目标机器是两张扩容至 **22 GiB** 的 RTX 2080 Ti、32 GB 系统 RAM，Linux / CUDA 12.8；普通 11 GiB 卡不能直接照抄容量配置。

## 保留与采用的实现

- 保留跨 GPU 的 RAM 专家补集和动态交换：从专家所属 GPU 保存被淘汰数据，等待所有相关卡的上传事件后再提交 RAM/GPU 归属；交换缓冲使用 `cudaHostAllocPortable`。
- 修复 RAM 预算只按 CUDA0 层排序的问题：在分割显存 profile 前保留全局排序，使两张卡缺失的热门专家都能进入 RAM。
- 保留 `--resident-experts-strict`；RAM 驻留模式的 layer split 强制独立 prefill 缓冲，防止借用唯一的专家副本。
- 分层权重加载改用上游 `--trim-stage-weights`，保留 CUDA0 的 router 以供预取。旧 `--stage-weights` 仅作为兼容别名，不再维护第二套 loader。
- 解码内核、服务层和会话处理采用 v0.1.39。并行会话需要单独配置与验证；本次单请求提速没有开启 `parallel > 1`，没有实现 SSD 会话缓存。

当前服务器已切回 IQ3_S，保留 `--max-context 262144`、INT8 KV、24/24 层划分、4096 prefill、MTP4 和原有 GPU 视觉配置。专家使用完整 RAM 补集与严格驻留检查，启动预留 4 GiB 系统余量。Q3XL 模型及专用运行文件已删除。v0.1.39 的 IQ3_S 尚未重新实测性能。

构建沿用上游固定的 llama.cpp revision。配置 `STRATA_BUILD_FUSION_TESTS=ON`，构建目标 `strata fusion_resident_swap_test file_expert_source_test`；GPU 架构为 75。旧的自定义 weight-stage loader 测试已随重复实现移除。

## English

This branch merges official **v0.1.39** (`6f32ec070f23ced9f50e704d854d775da52591ab`) for two modified **22 GiB** RTX 2080 Ti cards, 32 GB system RAM, Linux and CUDA 12.8. Capacity settings need adjustment for standard 11 GiB cards.

- Retains a shared RAM expert complement with swaps routed to each expert's owning GPU. RAM ownership changes only after all participating GPU uploads finish; exchange buffers are portable across devices.
- Preserves the global expert ranking before splitting the VRAM profiles, so a RAM budget covers hot missing experts from both GPUs.
- Retains strict residency and dedicated prefill buffers with resident layer splits.
- Uses upstream stage weight trimming, including CUDA0 router retention for prefetch. `--stage-weights` is a compatibility alias for `--trim-stage-weights`; the duplicate loader was removed.
- Uses upstream decode kernels, server and conversation handling. Concurrent slots are not enabled by this single-request configuration. SSD session storage is not implemented.

The server is configured for IQ3_S again, retaining 262144 context, INT8 KV, a 24/24 layer split, 4096 prefill, MTP4 and the original GPU vision setting. It uses a complete RAM expert complement with strict residency and 4 GiB startup headroom. Q3XL weights and dedicated runtime files have been removed. IQ3_S performance on v0.1.39 has not been remeasured.

Build for CUDA architecture 75 with `STRATA_BUILD_FUSION_TESTS=ON`; targets are `strata fusion_resident_swap_test file_expert_source_test`. The duplicate weight-loader test was retired with that implementation.

## 历史记录：v0.1.35 / Historical v0.1.35 results

下面记录仅属于 2026-10-02 的 v0.1.35、IQ3_S 配置，不代表 v0.1.39 的性能。
The following measurements describe the October 2 v0.1.35 IQ3_S configuration, not v0.1.39.


本分支采用上游 **0.1.35**（`d9ab8435f654c368c586340d490915f6addf56a3`）加实测双卡改动。设备为两张扩容至 **22 GiB** 的 RTX 2080 Ti、双 Xeon E5-2682 v4、32 GB DDR4，CUDA 12.8，Linux。普通 11 GiB 卡需要另行规划容量。

## 本次融合

- `--stage-weights` 按层所属 GPU 加载 dense 权重，全局张量保留；当前支持显式 layer split 的 native serve、spec ≥ 2。
- `--resident-experts-strict` 在 RAM 保存两张 GPU 的专家补集；内存不足时直接报错，避免部分驻留或退回文件 mmap 导致吞吐波动。沿用上游默认锁页及 RAM headroom。
- 动态交换先在所属 GPU 保存被淘汰专家，再上传新专家。两张卡各自的上传完成事件均确认后，才提交 RAM 槽归属和 GPU 驻留表。交换缓冲使用跨 GPU 可见的锁页内存。
- 专用 prefill 缓冲、96 槽环以及 KV/GDN 状态按所属层分配使用上游实现。旧版 `--stage-kv`、`--resident-cpu-experts`、`--pin-resident-experts`、`STRATA_PREFILL_RING` 等配置不再适用。

采用实测的上游计算内核和服务层；未迁回旧 Windows 专用存储链路。原 Windows 改造仍在仓库 `main`，旧双卡代码保留在 `cee3f85`。本次没有修改模型专家、ngram、模板或量化数据。

双卡按层执行；prefill 可以跨批次流水重叠，单序列 decode 仍需依次经过两张卡。本次不是专家并行或并发吞吐改造。

## 构建

先用上游 setup 准备 Python 环境、模型、专家缓存、tokenizer、MTP 和视觉组件。该分支的自定义引擎应从源码构建，勿用上游预编译引擎覆盖。固定 llama.cpp 版本与上游 setup 相同：`3cf03257f219afbe7334045ff7c6a06ac68c627d`。

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_FUSION_TESTS=ON
cmake --build build --target strata fusion_weight_stage_test fusion_resident_swap_test file_expert_source_test --parallel 8
```

保留自己已有的模型/pack 路径；将生成的配置 `exe` 指向新编译的 strata。setup 的 `--build` 也可构建生产引擎；上述目标用于运行新增检查。

## 实测配置

配置顶层 GPU 为 `[0, 1]`，`layer_split` 为 `"24"`，模型名称保持 `qwen3.8-flash-next-iq3_s`。模型 IQ3_S，原 IQ4_NL ngram、固定 Jinja template、视觉组件；引擎参数如下，文件路径另按自己的安装填写：

```text
--native-dense-gguf <IQ3_S第一分片>
--expert-cache auto
--prefill 4096
--spec 4
--spec-min-p 0.5
--max-context 262144
--kv int8
--mmap-experts
--vision
--vram-reserve-mib 700
--stage-weights
--resident-experts-strict
--no-prefill-borrow
--adapt-every 4
--pcie-frac 0
```

保留正确的 `--pack`、`--native`、`--ple-gguf`、`--expert-profile`、`--mtp`、tokenizer 和 vision 配置。显式 dense 第一分片防止把 ngram 分片重复作为 dense 权重扫描。

启动 server 前设置：

```sh
export STRATA_VERIFY_DEVICE_PLAN=0
export STRATA_SPLIT_RING=96
export STRATA_RESIDENT_HEADROOM_GIB=4
unset STRATA_RESIDENT_PIN
```

`STRATA_VERIFY_DEVICE_PLAN=0` 固定默认 host planner：每层计划根据最新 host residency 构造，在异步替换尚未完成时，旧专家由 RAM 计算，新槽暂不参与 GPU 计算。可选 device planner 未覆盖这套交换验证。`--pcie-frac 0` 让 decode 缓存未命中的专家由 CPU 计算，不关闭 prefill DMA 或动态交换。RAM 专家没有完整镜像，因此使用 `--no-prefill-borrow`。

本机正式服务端口8880，按用户要求无 API key 验证；这些服务设置没有改成框架默认值。启动器拒绝重复启动已有模型，不会主动停止进程。

## 2026-10-02 实测

三版按相同提示、顺序、seed417、temperature0、reasoning none 测一轮，262144 上下文、4096 prefill、MTP4、INT8 KV。prefill 请求无前缀复用，decode 请求输出1024 tokens。

| 指标 | 旧改造版0.1.27 | 纯上游0.1.35 | 融合版 |
| --- | ---: | ---: | ---: |
| 5,047 tokens prefill | 482.0 tok/s | 23.9 tok/s | 794.2 tok/s |
| 13,374 tokens prefill | 1040.5 tok/s | 216.3 tok/s | 1207.4 tok/s |
| 1024 tokens decode | 40.3 tok/s | 38.0 tok/s | 51.5 tok/s |
| 加载至 READY | 211.1 s | 344.6 s | 276.9 s |

没有清理 OS page cache，冷暖加载不同；纯上游首次请求有严重专家文件读取，所以这些数字不能代表所有提示或固定性能倍数。相对旧版，本轮13K prefill提高约16%，长输出decode提高约28%。

追加融合版独立长输入：46,032 tokens，完整 prefill **1446.3 tok/s**（31.827s），3 checkpoints。RAM 专家补集22.28 GiB，稳态最低可用RAM3.16 GiB，系统无新增swap-out；三版对照中融合最低可用RAM2.89 GiB。启动阶段有系统swap-out，运行阶段有少量swap-in，不能称完全无swap。运行期专家无文件blob回退。

## 验证范围

- 实际生产 loader 的47个权重分割点与完整 loader 比较字节、metadata及arena边界。
- 两张GPU共512次普通/锁页RAM专家交换，GPU新专家及RAM旧专家字节准确，归属互补，无文件回退。
- `file_expert_source_test`；会话续接、重复前缀复用、红色图像识别、工具调用结构。

新增测试入口由 `STRATA_BUILD_FUSION_TESTS=ON` 启用。工具没有真正外部执行，参数语义未全面验证；功能检查不等同于广泛质量或并发评测。262K已配置分配，最长输入实测46K，没有填满262K。

正式 Linux 引擎 SHA256：`d021c816511763c3968fde709469cc1dc1bfebf29e21e1a3a5f6e969adbc0c02`。模型、机器配置、账户凭据及编译产物不上传。旧框架和旧启动脚本保留供回退，转正不会重启正在运行的服务。
