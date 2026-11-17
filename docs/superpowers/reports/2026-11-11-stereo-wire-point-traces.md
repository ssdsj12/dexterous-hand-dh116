# WP1-A 双目软分割、匹配与采样归因

以 `98ba1fb872636459c13212ae97425fbb8b7c56c5` 为基线。Front 和 Roll20 各重放 4 帧双环境的已有原生 RGB，使用同一 balanced200 分割 Checkpoint。处理在服务器 CPU 上进行，两组均退出 0。候选仍为 `blocked`，未启动 Isaac、采集或训练。

现有对应模块新增三类逐视图概率归档、同帧左右 ID/光轴 Z 联合评分、固定像素的双向代价追踪，以及按像素对齐的正确点增减归因。复放沿用现有分割、深度、轮廓匹配与 Ribbon96，保持这些 v1 Checkpoint 源码绑定文件的字节不变。右侧监督复用现有独立原生 ID 解析、有效性和头标签入口。合成 CPU 测试使用左右不同 ID 数值，并确认缺右 ID 或右光轴 Z 无效时保留 unknown。两份历史原始捕获均缺右原生 ID，因此右 IoU、真实对应准确率与原生右监督仍为 NA。没有把 unknown 概率、左右不一致或光轴 Z 缺测当作物理遮挡标签。

## 分割覆盖与最终 Stereo

各列为左图原生线缆上的分割或最终 Stereo 指标。unknown 拒绝率的分母是 raw visible 命中的 native wire 像素，Stereo 覆盖率的分母是同一帧全部有效 native wire reference。visible/runtime IoU 在同一已解析标签像素上计算。

| 配置 | 抓法 | visible IoU | runtime IoU | visible coverage | runtime coverage | unknown 拒绝率 | 软引导 Stereo 覆盖率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Front | 二指 | .585 | .111 | 83.55% | 11.84% | 85.83% | 6.30% |
| Front | 三指 | .699 | .113 | 85.03% | 11.50% | 86.47% | 5.59% |
| Roll20 | 二指 | .625 | .155 | 86.11% | 16.23% | 81.16% | 53.06% |
| Roll20 | 三指 | .740 | .158 | 87.10% | 16.16% | 81.44% | 52.82% |

软引导保留完整灰度窗口，分割 visible 概率最多增加 0.03 ZNCC 代价。右预测概率低不剪除灰度窗口。原色轮廓仍为默认。额外的 `--colour-anchor-fallback` 实验保留原色轮廓的已有深度，仅在同一左色彩轮廓内、软匹配通过 LR 且右投影的全部插值像素有颜色支持时填补未测深度。该路径只读取 RGB 推理输出，沿用原 RGB 可见路径，并在原生参考评分之前决定深度。

## 正确点下降的分解

沿用同像素反投影欧氏误差不超过 **5 mm**。Front 分母 2128/2165，Roll20 分母 2009/2054。下表为每帧与参考相交的已采样点，格式为“正确 / 已测”，四帧数量一致。

| 方法 | Front 二指 | Front 三指 | Roll20 二指 | Roll20 三指 |
|---|---:|---:|---:|---:|
| 原色轮廓 | 6 / 15 | 0 / 14 | 91 / 96 | 89 / 96 |
| 预测硬 Mask | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 |
| 全窗口无引导 | 4 / 75 | 0 / 38 | 63 / 100 | 62 / 100 |
| 全窗口软引导 | 9 / 100 | 5 / 63 | 67 / 100 | 65 / 100 |
| 原色锚定软补测 | 14 / 82 | 5 / 58 | 73 / 98 | 64 / 98 |

Roll20 二指的 91 个原正确点在软引导下保留 5 个，83 个仍有 ≤5 mm 深度但未被新采样选择，3 个深度被拒，新增 62 个正确点。三指的 89 个原正确点保留 9 个，76 个有正确深度但未被采样，3 个深度被拒，1 个匹配深度误差大于 5 mm，新增 56 个正确点。退化主要来自扩大候选与测深支持后采样位置变化，固定尾部仍存在匹配误差。原色锚定保持所有旧正确点的像素测深，但采样仍移走二指 82 个、三指 77 个原正确点。Roll20 软引导的正确完整率为 3.335%/3.165%，锚定补测为 3.634%/3.116%，原色为 4.530%/4.333%。Front 的改善局限于同一 cohort。

Roll20 二指/三指的采样光轴 Z P95，原色为 6.06/20.60 mm，软引导为 43.20/37.04 mm，原色锚定为 21.23/22.27 mm。逐帧归档含 MAE、RMSE、P95、P99、覆盖和唯一/重叠拒绝原因。

## 固定尾部与复验

原来 12 个位置 × 4 帧始终留在回归中。原色的整数 winner 42 px 有 40 条，44 px 有 8 条，48 条全部 >5 mm 且 LR 误差为 0。软引导仍全部 >5 mm，44 条 LR 误差为 0，4 条为 0.234375 px；其中 4 条 winner 为 44 px 的位置亚像素偏移 +0.375 px。无引导有 28 条 >5 mm 和 20 条缺测，硬 mask 的 48 条均未通过。锚定补测保留原色的 48 条尾部。LR 一致性本身无法排除左右同时选错的视差。

所有比较方法均输出固定位置的整数代价曲线、实际 winner、亚像素偏移、反向候选曲线、LR 误差与最终深度误差。未进入搜索或无可用右查询时记录空值。软匹配曲线直接截取实际搜索；原色和硬 mask 的审计按原整数轮廓规则重新计算，并与实际 forward disparity 核对。原色额外保存灰度窗口 ZNCC 证据，此 RGB 代价没有参与原色排名。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_perception.py
python scripts/audit_dh116_wire_correspondence.py --proof <completed_native_proof> --segmentation-checkpoint <balanced200_checkpoint> --segmentation-sha256 7c9c3a60945e17a8363144d6ceebbd9cb18d676240582d2ac1f0428dc9333920 --output <new_private_output> --colour-anchor-fallback
```

41 项 CPU 回归通过，0 失败，3.14 秒。Roll20 另传 `--tail-fixture configs/dh116_training/stereo_wire_correspondence_roll20_tail_v1.json`。复核了最终 8 份帧归档的 SHA、三个概率头和两视图字段、LR 对应有限视差及 3D 采样有效性。Front、Roll20 摘要 SHA256 分别为 `4e35643fe27775bef16bc9347faf5e4664d4d26883e42943c54982d7c4fed8cb`、`baf44fac615a4cb4ca8a08e76063942d5b0fbc09aadb14a588b3748489f127fd`。执行的对应模块 SHA256 为 `e091af9e1a99c57b29f97451baf305d2549793b6026e08294476d3166b437a2c`，复放脚本为 `a3860e107f876223135e0097b5d5f3ab9cd98841c32031a9f06385b444379e3e`。原始帧与日志保留在独立私有目录。

下一步以经批准的左右同步原生 ID 与光轴 Z 建立配对真值，再比较保留已测路径的采样策略和视差尾部。两份旧捕获和额外候选均不授予感知、原生泛化或正式训练资格。
