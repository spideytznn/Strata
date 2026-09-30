# 双 RTX 2080 Ti / strata-2080tix2

此分支基于本仓库的 `main`，加入已在 Linux 实机部署的双卡优化。测试设备是两张扩容至 **22 GiB 显存**的 RTX 2080 Ti、双 Xeon E5-2682 v4 和 32 GB DDR4。以下配置与性能记录对应这台设备，普通 11 GB 2080 Ti 需要重新规划容量。

## 改造内容

- `--stage-weights`：每张卡加载自己执行层的 dense 权重，共用的全局张量仍保留，减少层权重副本。
- `--stage-kv`：每张卡只分配自己负责的 QSA KV/indexer 状态。24/24 分层、262144 上下文时，每卡节省约 1777.52 MiB。全局 QSA 索引保持一致，GDN 状态仍沿用原布局。
- `--resident-cpu-experts` 配合动态交换：GPU 专家和 RAM 专家组成互补集合。换入专家上传后，GPU 中的旧专家回到空出的 RAM 槽，继续保留随输出积累热点专家的收益。
- `--pin-resident-experts`：锁页保存 RAM 中的专家，prefill 可以直接 DMA 到 GPU，省去普通 RAM 到临时锁页缓冲区的 CPU 拷贝。
- `--resident-ram-headroom-mib`：允许显式设置加载 RAM 专家前必须保留的可用内存。

模型专家字节、量化格式和 ngram 算法沿用原实现。双卡仍按层执行：长输入的 prefill 可跨批次流水重叠，decode 的同一序列需要依次经过两张卡，因此 GPU 利用率不代表两卡容量没有得到使用。此分支没有加入后来已回退的专家并行引擎。

## 构建与配置

使用源码构建，CUDA 架构为 `75`。已测试的编译环境是 CUDA 12.8。通过 setup 准备环境时可使用：

```sh
python setup.py --build --no-start --gpus 0,1 --layer-split 24 --port 8880 --vision
```

选择 IQ3_S，并提供兼容的模型、ngram、MTP 与 vision 文件。构建和生成机器配置后，按下面的设置修改生成的 `strata-iq3_s.json`。这些是配置要点，不是包含模型路径的完整配置文件；已有参数应修改其值，避免重复添加。

```json
{
  "gpu": [0, 1],
  "layer_split": "24",
  "port": 8880,
  "env": {"STRATA_PREFILL_RING": "96"}
}
```

引擎 `args` 中使用：

```text
--expert-cache auto
--prefill 4096
--spec 4
--spec-min-p 0.5
--max-context 262144
--kv int8
--mmap-experts
--resident-cpu-experts
--pin-resident-experts
--resident-ram-headroom-mib 4096
--adapt-every 4
--stage-weights
--stage-kv
--no-prefill-borrow
--vram-reserve-mib 700
--pcie-frac 0
--vision
```

保留配置中正确的 `--pack`、`--native`、`--ple-gguf`、`--expert-profile` 与 `--mtp` 路径，以及 tokenizer/vision 配置。测试使用 IQ4_NL ngram；本仓库已有的 Q8_0 ngram 支持仍保留。

`--pcie-frac 0` 保持 decode 缓存未命中的专家由 CPU 计算，用于隔离 prefill 直传优化；它不会关闭 prefill 的 DMA，也不会关闭动态专家交换。动态互补集合使用 `--no-prefill-borrow`，因为 GPU 专家没有第二份 RAM 镜像。锁页内存不能被换出，不会增加实际物理内存容量。

此分支没有上传模型、机器绝对路径、SSH 凭据或编译产物，也没有把这套硬件配置设为所有设备的默认值。

## 实测与验证范围

2026-09-30，IQ3_S，262144 最大上下文、4096 prefill、MTP 4，温度 0。测速输入为重复的中文监控日志段落，各次请求没有复用 prompt 缓存。

| 输入长度 | 原 4096 配置 | RAM 直传配置 |
| --- | ---: | ---: |
| 5046 tokens | 492.5 tok/s | 734.8 tok/s |
| 约 13K tokens | 646.2 tok/s | 1014.6–1043.6 tok/s |
| 约 46K tokens | 702.7 tok/s（开启计时诊断） | 1192.1 tok/s；计时诊断轮为 1209.3 tok/s |

三段各生成 2048 tokens 的综合 decode：原配置 **46.11 tok/s**，直传配置 **46.36 tok/s**，未观察到明显退步。资源每 3 秒采样：可用 RAM 最低约 **1.97 GiB**，每卡剩余显存最低约 **1114 / 1335 MiB**；加载阶段系统 swap 写出约 **927 MiB**，推理阶段没有新增 swap 写出，但有约 246 MiB 系统 swap 读入。采样可能漏掉瞬时峰值。

已完成 13 项接口测试（含图片）、8 轮连续对话/缓存续接、三段长输出、额外无缓存长输入复测，以及两卡上共 512 次完整专家字节校验的 RAM/GPU 交换。早先的 stage-KV 验证还覆盖了所有合法 QSA 分区、两卡状态零化/RoPE 检查，以及相同专家缓存布局下的启动 logits 逐位对照。

262K 是已配置并分配的容量，本次最长输入约 46K，没有灌满 262K。这些是功能、数据完整性与性能回归结果，不是全面的质量或并发评测，吞吐会随实际输入内容变化。

## 源码与测试入口

运行核心对应已部署源码提交 `a60b4dd79fc8b4c2700a6d5ada5fe4620e2f4698` 的内容；发布到本分支时保留本仓库历史，因此提交编号会不同。已部署 Linux 引擎的 SHA-256 为 `2265ef4c47f873878662df55960878ffc5c77bc2427e6b8a2344d5cbf253658e`。

- [权重分层与动态交换设计](../STAGE_WEIGHTS.md)
- [QSA 状态分层与验证](../STAGE_KV.md)
- [锁页专家与 DMA](../RESIDENT_DMA.md)
- `tests/run_stage_cpu_tests.sh`：CPU 层归属、加载与交换检查。
- `-DSTRATA_STAGE_KV_TESTS=ON`：构建 `stage_kv_test`。
- `-DSTRATA_RESIDENT_EXPERT_TESTS=ON`：构建 `resident_expert_test`，检查两卡上的普通/锁页 RAM 交换与跨设备映射。

默认关闭新增优化选项。回退已有设备时，恢复对应引擎与配置的备份即可；源码分支的发布本身不会重启模型服务。
