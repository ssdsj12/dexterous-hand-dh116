# 双目分割候选策略与诊断回放

Ubuntu 任务工作树以 `75d337e8c2c5c7a21d60e96992b41df2ef7cc185` 为基线。实际故障是 `segmentation_masks()` 将高 unknown 分数作为硬否决，即使 visible 头已给出高概率。unknown 的现有标签语义是渲染器实例 ID 能否解析。

## 策略与身份

- `dh116-wire-mask-policy-v1` 的默认候选策略 `visible_proposal` 只依据 RGB 推理的 visible 分数。
- `legacy_visible_and_unknown` 显式保留旧阈值及历史审计结果，仅用于诊断回放。
- `qualified_reliability_filter` 逐视图核对 visible 和 unknown 头。启用否决需要标签监督、持出评估、非零证据 SHA、明确的 renderer ID 语义及 veto 许可。单侧具备能力时，过滤仅作用于该侧。元数据校验仅证明合同字段有效，实际部署资格仍需要独立评估证据。
- 原生相机和独立 RGB 推理共用同一个诊断 wrapper。读取同一份已验 SHA 的 checkpoint 字节，以 weights-only 方式解析，并按原形状、dtype 和有限值要求校验。它只接受 RGB、标定和两侧像素有效性，策略和版本写入处理合同。
- 已有 v1 诊断权重可与显式策略配合复现。未声明策略的学习分割配置会明确被拒绝。源代码 SHA 绑定仍生效，旧部署包应在其匹配的代码快照使用。现有无学习分割配置继续走原有 RGB 匹配路径。

## CPU 回归

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_inference.py \
  tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py \
  tests/test_dh116_stereo_depth_auxiliary.py
```

初始 RED 为 6 项预期失败、31 项通过。额外的全零证据 SHA 用例在补齐校验前预期失败。最终 62 项通过，退出码 0，日志 SHA256 `75bd852f0ef1cde6379e4270ec63301a26b71a5ec07796079317b4315dba2a3f`。528 个保护文件和两个历史 Checkpoint 均通过 SHA 核对。

## 历史同批 CPU 回放

以同一已保存 `balanced200` 诊断权重和对应 RGB/标定，回放 Front 与 Roll20 的第 21 帧，每帧两个环境。A/B/C 使用同一个 `wire_silhouette` matcher，D 使用现有 full-patch 软引导 matcher 和未遮蔽的 RGB patch，D 与 A/B/C 的比较同时涉及 matcher 差异。候选生成和匹配完成后才读真值标签评分。该诊断使用既有图像，独立持出证据仍缺。

诊断 checkpoint SHA256 为 `7c9c3a60945e17a8363144d6ceebbd9cb18d676240582d2ac1f0428dc9333920`。Front 和 Roll20 的绑定原始捕获 SHA256 分别为 `2f0ee494d1339f7d68d661eb81af23d8578843b935ae21863802c9c1bc60bdec` 与 `733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`。完整诊断、脚本及日志保留于 Ubuntu 的 `/tmp/dh116-segmentation-mask-ablation-CC88Ws`，摘要 SHA256 为 `499c97536966cf16078d00afdeb8d84dcc1cc7907e22f6ff68b9dabfb8afe729`。

两份捕获均缺少独立右视图 Native ID，右侧真值 IoU、precision、recall 和标签覆盖均为 NA。左侧已知标签覆盖（env 0/1）为 Front 56.1%/51.2% 和 Roll20 53.0%/48.1%。以下误差在匹配有效且有原生 wire 参考的左像素上计算，为光轴 Z 绝对误差。耗时为两个环境合计的 Ubuntu CPU matcher 耗时，排除分割前向和评分。

| 场景/env | 方法 | 左 IoU / P / R | 候选左/右像素 | unknown 否决左/右像素 | 左匹配有效 / 参考上有效 | Z P95 / P99 mm | 两环境秒 |
|---|---|---|---|---|---|---|---|
| Front/0 | A 原色 | 0.895 / 0.895 / 1.000 | 2377 / 2363 | 0 / 0 | 20 / 20 | 80.80 / 263.16 | 0.95 |
| Front/1 | A 原色 | 1.000 / 1.000 / 1.000 | 2165 / 2174 | 0 / 0 | 25 / 25 | 33.89 / 37.26 | 0.95 |
| Front/0 | B visible | 0.585 / 0.661 / 0.836 | 137173 / 137168 | 0 / 0 | 244 / 22 | 53.37 / 261.80 | 46.98 |
| Front/1 | B visible | 0.699 / 0.797 / 0.850 | 152067 / 152137 | 0 / 0 | 270 / 9 | 349.15 / 350.57 | 46.98 |
| Front/0 | C legacy AND unknown | 0.111 / 0.646 / 0.118 | 705 / 699 | 136468 / 136469 | 3 / 3 | 32.90 / 32.90 | 0.25 |
| Front/1 | C legacy AND unknown | 0.113 / 0.883 / 0.115 | 285 / 283 | 151782 / 151854 | 2 / 2 | 63.68 / 64.30 | 0.25 |
| Front/0 | D visible 软引导 | 0.693 / 0.693 / 1.000 | 137555 / NA | 0 / 0 | 801 / 134 | 132.70 / 309.16 | 25.35 |
| Front/1 | D visible 软引导 | 0.822 / 0.822 / 1.000 | 152391 / NA | 0 / 0 | 748 / 121 | 79.24 / 123.41 | 25.35 |
| Roll20/0 | A 原色 | 0.889 / 0.889 / 1.000 | 2261 / 2268 | 0 / 0 | 654 / 654 | 14.25 / 21.81 | 0.87 |
| Roll20/1 | A 原色 | 1.000 / 1.000 / 1.000 | 2054 / 2069 | 0 / 0 | 702 / 702 | 14.47 / 21.80 | 0.87 |
| Roll20/0 | B visible | 0.625 / 0.695 / 0.861 | 146541 / 135978 | 0 / 0 | 1022 / 422 | 13.80 / 21.61 | 45.57 |
| Roll20/1 | B visible | 0.740 / 0.831 / 0.871 | 161613 / 150767 | 0 / 0 | 980 / 408 | 3.26 / 21.70 | 45.57 |
| Roll20/0 | C legacy AND unknown | 0.155 / 0.773 / 0.162 | 744 / 783 | 145797 / 135195 | 0 / 0 | NA | 0.25 |
| Roll20/1 | C legacy AND unknown | 0.158 / 0.865 / 0.162 | 388 / 406 | 161225 / 150361 | 4 / 4 | 2.12 / 2.17 | 0.25 |
| Roll20/0 | D visible 软引导 | 0.717 / 0.717 / 1.000 | 146853 / NA | 0 / 0 | 2821 / 1066 | 22.27 / 67.60 | 27.65 |
| Roll20/1 | D visible 软引导 | 0.849 / 0.849 / 1.000 | 161878 / NA | 0 / 0 | 2791 / 1085 | 21.81 / 48.08 | 27.65 |

移除 unknown 否决后，四个环境的左侧 recall 从 11.5%–16.2% 升至 83.6%–87.1%。直接把 visible 当作硬候选也产生了 13.7 万–16.2 万个左候选，匹配耗时上升，部分误差尾变差。该诊断结论支持单独审查候选阈值和 RGB 证据，生产配置继续沿用原色匹配。后续资格需要新增独立初态和两侧 Native ID，并在持出集评估候选范围、误差尾和吞吐。本工作包没有启动 Isaac、训练或采集。
