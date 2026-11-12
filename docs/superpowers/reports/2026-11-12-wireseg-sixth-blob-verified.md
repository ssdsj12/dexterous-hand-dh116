# WireSeg 原始 RGB 分片续传与六文件核验（2026-11-12）

## 本次结果

沿用已实现的 pinned catalogue、分块 resume 和 verified-subset 路径，没有重新实现下载器。Windows 完成 `data/train-00000-of-00045.parquet` 的剩余分块并验证完整 LFS SHA，随后经 SCP 传入 Ubuntu 独立 incoming 目录；Ubuntu 再次验证尺寸、SHA 和 Parquet 内容后使用 no-clobber 移入数据目录。此次不是新原生采集，也未训练新模型。

- Dataset: `DeformX/WireSeg-36K`
- 固定 revision: `1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`
- catalogue SHA256: `bc9afab256cb247b60d45116b9351810c99d6e0818fd26e3690d1cf9dbe86e93`
- 本分片完整尺寸: **583626806 bytes**
- 本分片 LFS/实测 SHA256: `17ddf17fe837d17027780dda78a916b4874403c3be0e320cb95d06977d207018`
- Ubuntu 私有原始目录: `/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/data/train-00000-of-00045.parquet`
- 接收回执 SHA256: `35c8b39ee7c9073b5de106d6821f2e68a519d88afdbde2e33b9d7b74ad5fa1d4`
- 接收回执: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-train00000-relay-LpMr1m/receipt.json`
- 核验完成 UTC: `2026-11-12T11:02:55.971741+00:00`

此时完整数据为 **6/148 blobs：3 个 RGB Parquet、3 个深度分片**。完整下载尚未完成；保留已有 ranges/resume。它不阻断无相机 Teacher。

## 数据身份与训练边界

读出 800 条真实 RGB/Mask 记录，全部属于 `data_center` source family。虽然文件路径以 `train-` 开头，本项目已冻结的 source-family split 将 `data_center` 放在 **test**，故这 800 条不得加入固定16样本训练或调参。既有 `flying_wire` train 与 `wire_on_plane` validation 不变。

使用现有 `depth_member` 规则按 source family/cable group/frame stem 精确匹配，当前已验证 `depth-00019`、`depth-00024`、`depth-00062` 中找到的配对是 **0/800**。没有将其他深度帧作为标签，没有从文件序号推断对应关系，没有为缺失深度补零。该分片现在可提供 held-out RGB/Mask，不能声称增加了 800 条深度监督。

| 已有深度分片 | bytes | 完整 SHA256 | 本新增 RGB 精确配对 |
|---|---:|---|---:|
| 00019 | 22541803 | `a3b12e0269c626e2c6983cc861f46eaa3c3ae53b543338ba4f8eba9d12c352c5` | 0 |
| 00024 | 22972279 | `9e7c7280f6aa0e75b2f0799a6d3e950b08217b40aa5724958882bc93b2478eba` | 0 |
| 00062 | 913337418 | `b730d6c70377585055522a52cb1c246ac88592a36d65ad470f4a365f8a54bccb` | 0 |

读取使用数据工具目录既有 `pyarrow 19.0.1`，未替换 DH116 Python、PyTorch 或 Isaac 依赖。所有 Parquet、深度档案、模型和完整实验日志保留私有路径，不上传 GitHub。

## 验证与后续

本工作包运行现有 relay/subset CPU 回归：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_wireseg_blob_relay.py tests/test_dh116_wireseg_pretraining.py
```

实际结果记录于发布时 registry/readiness 的 `flex.f5-wireseg-sixth-blob`。原生验证次数 **0**，新增训练更新数 **0**，Stereo/状态估计资格 **未取得**。固定16样本诊断的完整结果及失败记录见 [诊断报告](2026-11-12-wireseg-fixed16-overfit-diagnostic.md)：深度仍不及常数基线，不能将下载完成或训练 loss 下降解释为泛化通过。

后续仍按完整 SHA 和 exact-depth identity 增量接入；继续下载不需要等待 148/148 才使用已验证子集。测试 family 必须保持隔离，新增深度须精确配对后才能进入对应 split 的监督。
