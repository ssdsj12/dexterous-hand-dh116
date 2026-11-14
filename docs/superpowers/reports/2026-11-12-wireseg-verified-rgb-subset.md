# WireSeg 原始 RGB 子集、恢复和持出验证

使用 Hugging Face `DeformX/WireSeg-36K` revision `1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`。先以 HTTP 206 读取锁定 Parquet footer 检查场景族，再经 Windows 四路断点续传。Windows 组装后与 Ubuntu 入库前分别按官方 LFS 校验。

| 来源 | 字节 | LFS SHA256 | 场景族 |
|---|---:|---|---|
| `data/train-00008-of-00045.parquet` | 623775831 | `9ad438f849cf5718a11abac5e40b4d77c17d559b76c24e285458aa0479493af6` | flying_wire |
| `data/train-00028-of-00045.parquet` | 928769640 | `bf983007bc3c087a0f9b48edba2ddd04f0398c3cf595e98c90e517ef8eee3ca3` | wire_on_plane |

训练文件传输 156.2 秒，验证文件 119.5 秒，均为实测分块网络下载时间。原有深度 19、24 分片各 360 行保留。训练与验证整族分开，data_center 和 real 保留为独立测试域。

`pretrain_dh116_wireseg.py` 新增 `subset`、`evaluate`、`--subset-manifest`、`--validation-manifest`、`--resume` 及显式 `seg-only`／`seg-depth`。Manifest 固定 revision、整包 SHA 与有序样本身份。训练开始前检查所有 manifest 行和场景族。恢复核对数据与训练配置，保留 AdamW、RNG、有效行和绝对更新步。所有 Checkpoint 均写入独立私有输出目录。

| Ubuntu CPU 实验 | 结果 |
|---|---|
| seg-only 128 → 160 | 从真实 128 步断点续到 160 步，384 条冻结候选，已解码 188 张，28 张中央裁剪无 Mask，160 张用于更新，深度监督 0 步。 |
| 持出验证 | wire_on_plane 128 张，93 张裁剪内有 Mask，可见 IoU 0.127053，平均总损失 3.252691。 |
| 精确配对核验 | 两个原始深度分片 720 个唯一成员，与已校验训练 RGB 同名同族匹配 360 个。 |
| 独立 seg-depth 16 | 19 张已解码，16 个有效更新均有精确深度，累计 32506 个有效监督像素。持出深度精度仍待配对验证数据。 |

| 证据 | SHA256 |
|---|---|
| 训练子集 | `e9ec39f2211ebd364b677b836fb42eb3a63e2a7e70077f06accd74b8e4fb5932` |
| 验证子集 | `5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f` |
| 恢复后 Checkpoint | `963b2443926dee7e3ed28523560232b359c315516ce958f4e0df39c918fa4011` |
| 持出验证结果 | `9e9626a4615ff556af0c4d7a9a856bfdcf0b656a26e64991021f36831333c38b` |
| 配对监督 Checkpoint | `055ca13e11d3cc55a778d6fcde368685a3ee0c24aa80a4becf25a92c520832bd` |

运行命令使用当前 Ubuntu 解释器 `python -m scripts.pretrain_dh116_wireseg subset|pretrain|evaluate`，每次使用独立 `--output`，固定训练裁剪 192 和 seed 116。CPU 回归 `python -m pytest -q -p no:cacheprovider tests/test_dh116_wireseg_pretraining.py tests/test_dh116_wireseg_blob_relay.py` 共 17 项。原始样本、模型和实验日志保留在 Ubuntu 私有证据目录 `wireseg-original-rgb-eQP8ZB`。本次为小规模单目诊断，正式泛化、双目和策略资格仍待原生验证。
