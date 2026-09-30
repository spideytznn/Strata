# 本机定制版 / Local variant

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.27，上游固定提交 `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`。保留上游历史、版权声明与 MIT 许可证。

本版本为 48 GB RAM + 32 GB VRAM 的单卡设备调整了权重存储和 prefill，并支持独立 Q8_0 ngram 表。没有重新训练模型，也没有改变 ngram 哈希规则。

## 主要改动

- `--vram-only-experts K`：将专家 profile 的前 K 项只存入显存，在 RAM arena 中紧凑排除其副本。相关显存槽不能被逐出或借给 prefill。
- 当常驻专家使可借用缓存过小时，prefill 规划器也考虑缓存之外的空闲显存，使用独立缓冲区扩大批次，保留设定的显存余量。
- PLE/ngram 根据 GGUF 类型使用 IQ4_NL 的 90 字节行或 Q8_0 的 170 字节行；覆盖异步读取、缓存、跨页与批量路径。
- 独立、仅含 `per_layer_token_embd.weight` 的 ngram GGUF 不加入主模型 dense 分片列表，避免不同模型发布包的分片元数据冲突。
- 保留 `STRATA_MTP_REPO` 下载源覆盖和默认关闭的 `STRATA_VERIFY_LOGITS` 诊断开关。

## 构建与配置

请使用此仓库的源码构建。`setup.py` 默认可能使用上游预编译引擎，它不含本机补丁；使用 `--build` 指定源码构建。若只准备环境而不启动模型，另加 `--no-start`。本机编译使用 Windows、MSVC 2022、CUDA 13、sm_120。

VRAM-only 必须搭配 `--expert-profile`。不兼容 `--mmap-experts`、`--no-pool`、按层缓存分配，以及关闭所需 token graph 的诊断组合。当前也明确拒绝与 `--layer-split`、`--expert-cache-remote` 混用；这不影响 K=0 时使用上游路径。

原部署使用 7000 个 VRAM-only 专家、裁到 7300 项的热度 profile、`--expert-cache auto`、`--prefill auto` 和 1536 MiB 显存预留。这是特定模型和设备下的配置记录，**不是任意 profile 或其他设备都适用的默认值**。完整排名可能让自动缓存占满更多显存，改变 prefill 的规划；应按实际资源和模型调整。

Q8 ngram 通过 `--ple-gguf` 指定兼容 GGUF；不需要把整张表加载到内存，也不需要为了使用现成 Q8 表下载 BF16 全模型。主模型权重、热度 profile、ngram、MTP 权重、机器配置和构建产物不随此源码仓库上传。上游自带的小型 draft_vocab.bin 等运行数据保留。

本机另行使用 froggeric v22.5 模板，位于模型 pack 的 `tokenizer/chat_template.jinja`。该外部模板没有被复制进此源码仓库，仓库默认模板仍来自上游；如需使用自定义模板，应在准备 pack 后单独配置。

## 验证范围

v0.1.27 合并版完成 CUDA Release 编译及 CPU 检查：

- 90/170 字节读取器自检、4,099 行合成 Q8 表参考比对、错误类型/形状/截断拒绝测试。
- Q8 解码穷举 20,316,160 个比较值，与 pinned GGML 参考实现逐位一致。
- 实际 Q8 表抽查 14,784 行，8 种读取组合全部与独立文件偏移读取及 GGML 解码一致。
- 纯 CPU Python 测试 23 项通过，1 项因 Windows 符号链接条件未满足而跳过；模板渲染和图片标记兼容检查通过。
- 专家分层 GPU 测试程序编译通过，但本轮未执行。

**v0.1.27 合并后没有启动完整模型，也没有执行 GPU 运行测试。** 新版吞吐、峰值内存和输出质量尚待实机验证。旧版 M2/M3 的数值与性能记录见 [VRAM_TIER.md](VRAM_TIER.md) 和 [PREFILL_M3.md](PREFILL_M3.md)，它们不是此次上游合并的性能或正确性证据。

[多对话缓存文档](MULTI_CONVERSATION_CACHE_DESIGN.md)仅为设计，尚未实现共享缓存池或并行推理。

## 后续维护

保留上游 remote，用三方合并更新，重点检查专家常驻保护、prefill 缓冲规划、PLE 行宽、分片列表与新增 GPU 路径的交叉影响。不要用上游发行包覆盖本版本引擎后继续沿用新增参数。
