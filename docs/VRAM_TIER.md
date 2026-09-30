# VRAM-only expert tier（本机 M2 验证）

2026-09-29，Strata 0.1.20，本机 Windows / RTX 5090 32 GB / 48 GB RAM。

`--vram-only-experts K` 将专家热度文件前 K 个专家只保留在显存中；RAM arena 直接按其余专家构建。GPU 常驻槽不能被动态替换或借给 prefill。默认 K=0 保持原有布局。

## 本机配置

IQ3_S 使用 `--vram-only-experts 7500`、`--expert-cache auto`、`--vram-reserve-mib 1536`。上下文为 262144，KV 为 int8，视觉功能开启。模型仍使用原 IQ3_S GGUF 和 SSD 上的 ngram 表。

专家 arena 从 46.84 GiB 降至 **32.62 GiB**，减少 **14.22 GiB**。一次正式配置启动中，约 30 GiB 由 CUDA 分片注册，余下约 2566 MiB 由 VirtualLock 成功锁定；GPU 专家缓存实际为 7849 个槽、14.88 GiB，全部加载后 CUDA 报告剩余显存 4473 MiB。自动定容会根据当时资源变化，这些槽数不是固定承诺。

## M1 修复

- 修正稀疏 arena 的集合反转：RAM 保留 skip 集的补集。
- 稀疏加载发生短读或 I/O 错误时拒绝成功返回。
- 全部专家被排除的层不再返回虚假的 CUDA 层地址。
- 启动日志及网页 INFO 的 arena 大小改为实际保留字节数。
- 新增 `strata-tier-test`，用小型合成权重验证 RAM 所有权、文件字节、CUDA 映射地址、重复名单、空层、全排除和截断文件。
- 新增默认关闭的 `STRATA_VERIFY_LOGITS` 环境变量，导出首个原生 verify 窗口的 logits，弥补现有 `--dump-logits` 不覆盖该路径的问题。格式为 int32 vocab、int32 rows、int64 position，随后是 FP32 数值。

## 验证结果

固定前 10000 个专家作为缓存排名、32K 上下文、视觉关闭、PCIe 分流 0.55、动态替换每 4 轮一次，顺序执行相同四组请求：短数学题、24576-token 代码资料、续接、新会话代码题。

| 引擎 | 24576-token 提示词处理时间 | 速度 |
| --- | ---: | ---: |
| 原发行版 | 156.03 秒 | 157.5 token/s |
| 本地补丁，K=0 | 143.82 秒 | 170.9 token/s |
| 本地补丁，K=7500 | 110.80 秒 | 221.8 token/s |

三组的四次请求均正常退出，生成 token 全部相同。此表是一次顺序测试，不是多轮统计平均；操作系统缓存与负载可能影响速度。长资料请求只生成很少的结束 token，因此它用于衡量 prefill，不用于衡量持续生成速度。

默认 PCIe 分流时，增加可 DMA 的 RAM 层会改变部分专家的 CPU/GPU 计算分配，因此 logits 并非逐位一致。另做 PCIe 分流为 0、关闭动态替换的控制实验：K=0 与 K=7500 首个 6-token 窗口的 **1489920 个 logits 全部逐位相同**，128 个生成 token 也一致。这验证了受控条件下搬运和布局的正确性，不代表任意输入均有逐位一致保证。

另已验证：

- 小型 canonical experts.bin 与多 GGUF 分片合成测试全部通过。
- 5 个不兼容 CLI 参数组合在加载前拒绝。
- 注入两次缓存分配失败，最终缓存缩小到 6986 个槽；要求全部 24576 个专家只驻显存时明确失败，未继续运行缺失专家的模型。
- 262K + 视觉候选配置启动成功。HTTP 文本请求正确回答 391，红色图片识别正确；24604-token 长请求正常回答 OK，提示词处理约 51.21 秒。
- 命令行单次生成路径以 K=7500、每轮动态替换运行，生成 96 个 token，完成 1745 次专家替换并正常退出；也验证了该路径的借槽与归还。

**262K 是已成功初始化的配置；本轮最大实测输入约 24.6K token，并未完整压测 262K 输入。** 正式配置的缓存数量和 prefill chunk 与上述固定缓存基准不同，51.21 秒不能与基准表直接作为同条件性能比值。

## 复现与限制

构建主程序：`cmd /c D:\Strata\build-strata-m0.bat`。另构建目标 `strata-tier-test`，运行时传入一个不存在的新目录作为合成文件目录。M2 的脚本、原始日志、输出 token、logits 和内存采样位于 `D:\Strata\work\m2`。

功能需要专家 profile、GPU 专家计算及驻留表；不支持 `--no-pool`、`--mmap-experts`、`--expert-cache-per-layer`、`--no-capture`、`--no-token-graph` 和 layer/half dump 组合。固定 GPU 专家会减少可动态替换及可借给 prefill 的槽位，不应把 K 盲目设到缓存总槽数。

现有 `--expert-cache N` 对 native pack 使用 N × 最大专家尺寸作为字节预算，实际槽数可能大于 N。应以启动日志为准。

重新运行 SETUP 更改配置、或手工覆盖源码升级后，应检查 `--vram-only-experts 7500` 仍然存在并重新构建补丁。发布版 exe 不包含本地功能。

## 已部署与回滚

现役 `D:\Strata\engine\strata.exe` 已替换为实际通过上述验证的二进制，SHA-256：

`290fb945270abb310d7e619e7563c6f81968ddce91b737a1f1483dfa7e80d353`

`engine\BUILD.json` 标记为本地构建，并记录与安装器一致的源码指纹。`strata-iq3_s.json` 已加入 K=7500，并将显存预留设为 1536 MiB。测试后台服务已停止，仍使用原桌面快捷方式启动。

原引擎保存在 `D:\Strata\engine\strata.exe.orig`；原配置及 BUILD.json 保存在 `D:\Strata\work\m2\deployment-backup`。需要回滚时，先关闭 Strata 服务，再在 PowerShell 执行：

```powershell
Copy-Item -LiteralPath D:\Strata\engine\strata.exe.orig -Destination D:\Strata\engine\strata.exe -Force
Copy-Item -LiteralPath D:\Strata\work\m2\deployment-backup\BUILD.json -Destination D:\Strata\engine\BUILD.json -Force
Copy-Item -LiteralPath D:\Strata\work\m2\deployment-backup\strata-iq3_s.json -Destination D:\Strata\strata-iq3_s.json -Force
```

再次启动即可恢复原发行版和原配置。M1 原始改动快照在 `work\m2\m1`，上游对应文件快照在 `work\m2\original`；源代码改动另导出为统一 diff 供后续重放与审查。
