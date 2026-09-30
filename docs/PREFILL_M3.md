# Strata M3：prefill 优化结果

已部署新版引擎和折中配置，原桌面 Start-Strata-IQ3S.bat 继续适用。

此前 prefill 自动规划只考虑可临时借用的专家缓存。7849 个缓存槽中有 7500 个是权重的唯一副本，不能借出，导致批次只有 256，缓存之外的空闲显存没有用于扩大批次。

现在规划器比较借用缓存与独立缓冲区两种方案；显存独占层启用时，如果独立缓冲区能容纳更大批次且保留指定的后续分配余量，就选择独立缓冲区。使用实际 bytes_needed 估算，不覆盖常驻专家。显存不足时仍保留原借用方案，未修改计算内核、量化权重和 KV 精度。服务端与单次 CLI 两条路径均已处理。

## 实测对比

同一份 24604-token 输入，262144 上下文容量、int8 KV、视觉开启、相同采样设置。两次都是零前缀复用的全量 prefill，中间插入独立请求；第二次包含文件/行缓存变热的影响。速度只计算 prefill，不包含模型加载和输出生成。每配置两次测量，非大样本统计，不能据此承诺任意输入都达到同一速度。系统内存受其他桌面应用影响。

| 配置 | 第一次 tok/s | 第二次 tok/s | 全测试过程最低可用 RAM（GiB） |
|---|---:|---:|---:|
| M2 原配置，256 批次 | 478.6 | 481.6 | 4.07 |
| M3，7500 专家，2048 批次 | 1223.7 | 1252.4 | 3.00 |
| M3，7500 专家，自动 6144 | 1665.8 | 1742.9 | 3.29 |
| M3，6500 专家，自动 8192 | 1710.7 | 1794.5 | 1.43 |
| M3，7000 专家，自动 8192（采用） | 1721.8 | 1829.3 | 2.83 |

采用 7000 方案：专家 RAM arena 为 33.56 GiB，比 M2 的 32.62 GiB 增加约 0.94 GiB；7300 个缓存槽占 13.84 GiB 显存，其中 300 槽可动态替换。独立 prefill 缓冲估算 3805 MiB，自动选择 8192 批次；全部加载后日志显示剩余约 1568 MiB 显存。保留 --vram-reserve-mib 1536，该余量在 prefill 规划时保留，后续 verifier 等分配还会使用一部分。

6500 方案在整个测试过程最低仅剩 1.43 GiB RAM，速度收益有限，未采用。当前 7000 方案测试最低可用 RAM 2.83 GiB，不以“刚加载完成”的单一时点作为判断依据。

## 验证

- 长输入回复 OK；随后数学续聊正确返回 391，并复用 24597 token。
- 60049-token 首尾编号回忆正确，prefill 32.840 秒，约 1828.5 tok/s。
- 60080-token 续聊再次正确返回编号，复用 60058 token，只重读 22 token。
- 图片识别正确返回红色；代码生成完成 256 token（达到设置上限），约 103.8 tok/s，仅作为解码运行检查，不作与旧版的解码性能比较。
- 单次 CLI：4096-token 输入、64-token 解码、动态专家替换正常；检查了独立缓冲区在 CLI 中按实际输入缩小批次的路径。
- Release 构建成功。没有声称不同批次下所有 logits 逐位相同；本次是显存分配与批次规划变化，未改数值内核。
- 保留 262144 上下文容量，最大实际输入验证到约 60K，尚未完整压测 262K 输入。

## 当前配置与回滚

配置 D:\Strata\strata-iq3_s.json：--vram-only-experts 7000、--expert-cache auto、--prefill auto、--expert-profile D:\Strata\data\expert-profile-prefill7300.bin。新 profile 是原始排名的前 7300 项，原始完整排名文件保留。

M2 引擎、配置、BUILD.json、PDB 的完整回滚备份位于 D:\Strata\work\m3\deployment-backup。更早的未改造引擎 D:\Strata\engine\strata.exe.orig 保留。

关闭 Strata 服务后，回滚到 M2：

```powershell
Copy-Item -LiteralPath D:\Strata\work\m3\deployment-backup\strata.exe -Destination D:\Strata\engine\strata.exe -Force
Copy-Item -LiteralPath D:\Strata\work\m3\deployment-backup\strata.pdb -Destination D:\Strata\engine\strata.pdb -Force
Copy-Item -LiteralPath D:\Strata\work\m3\deployment-backup\BUILD.json -Destination D:\Strata\engine\BUILD.json -Force
Copy-Item -LiteralPath D:\Strata\work\m3\deployment-backup\strata-iq3_s.json -Destination D:\Strata\strata-iq3_s.json -Force
```

## 多套对话缓存的结论

可以实现，不需要复制模型权重；当前版本尚未实现。本次只处理 prefill 性能。

目前的 prompt-cache 是同一条历史分支中的检查点，每个检查点只保存 GDN 递归状态、PLE 历史和索引尾部，约 118 MB；KV 和已完成的索引数据仍共用一份。增加 --prompt-cache 数量不能独立保留多条对话分支。

完整多对话快照需要保存有效长度内的各层 KV、索引、递归/PLE 状态、token 与图像标识，以及恢复 MTP 所需的状态；恢复到固定地址，维持 CUDA graph 指针有效。可以共用权重、RoPE 表和工作缓冲区。应按有效长度存储，而不是每套都保存最大 262K 容量。

按当前 12 层 QSA、每层 2 个 KV head、head_dim 256、INT8 KV 与每 4 token 一个索引池化项估算，主模型每 token 的 KV 加索引约 14208 字节，再加一份约 112 MiB 递归状态：32K 约 0.55 GiB，128K 约 1.85 GiB，256K 约 3.58 GiB。以上为主要数据估算，不含额外历史检查点、MTP、元数据和对齐，完整实现会略大。

本机更适合“显存保留活动会话、SSD 保存少量不活动快照、内存只做受限暂存”的方案，按 LRU 和磁盘容量淘汰。这样 ZCode 的后台记忆提取可以使用另一套状态，结束后恢复主对话；实际切换延迟需实现后测量。不能直接把不匹配的旧 KV 当作当前请求复用。

没有修改 ZCode 设置，后台记忆请求引起的会话缓存切换仍可能发生，本次只显著缩短重读耗时。
