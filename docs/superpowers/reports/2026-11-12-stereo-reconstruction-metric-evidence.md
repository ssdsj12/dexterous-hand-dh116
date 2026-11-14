# 双目重建的密集、稀疏与空间指标

基线 `75d337e8c2c5c7a21d60e96992b41df2ef7cc185`。现有 `audit_wire_frame()` 增加离线 `wire_reconstruction_evidence_v1`。同像素 3D 欧氏误差 ≤5 mm 的判据保持一致。参考只进入审计评分，实际匹配、候选选取和保留槽位仍从传感输出计算。

- 密集 precision 的分母为有合格光轴 Z 参考的已接受 wire ROI 像素。密集 recall 的分母为该帧全部合格可见 wire 参考像素。总接受像素和 ROI 内缺参考的像素分别记录。
- 稀疏 precision 的分母为有合格 wire 参考的已选唯一像素。未评分像素单独计数。历史 `correct_3d_point_completeness_5mm` 仍保留，新增等值字段 `sampled_correct_pixel_contribution`，同时报告槽位上限与理论贡献上限。
- 空间支持使用左图有参考交集像素的行列外包范围，分别报告密集和稀疏范围。单轴参考跨度为零或无观测时该项为 NA。外包范围不会计量孔洞、完整 3D 空间或隐藏表面。

## 同 cohort 的实际离线重放

Front 与 Roll20 各四帧、每帧二指和三指各一个环境，使用已归档双目 RGB 与 SHA 绑定的同一 balanced200 Checkpoint。两次重放均退出 0。同一布局四帧的下列数值一致。密集 precision 和 recall 以及稀疏 precision 单列分母。

| 布局和抓法 | 参考像素 | 原色密集 precision | 原色密集 recall | 原色稀疏 precision | 稀疏贡献与槽位上限 |
|---|---:|---:|---:|---:|---:|
| Front 二指 | 2128 | 6/20 | 6/2128 | 6/15 | 0.282% / 6.015% |
| Front 三指 | 2165 | 2/25 | 2/2165 | 0/14 | 0% / 5.912% |
| Roll20 二指 | 2009 | 614/654 | 614/2009 | 91/96 | 4.530% / 6.371% |
| Roll20 三指 | 2054 | 661/702 | 661/2054 | 89/96 | 4.333% / 6.232% |

Roll20 原色稀疏已评分像素覆盖参考列跨度的 88.98%/91.73%，空间范围与密度须分别解读。启用原色锚定和槽位保留后，Roll20 密集 recall 为 740/2009、787/2054，稀疏正确点仍为 91/89。通过逐帧读取新旧归档并核验 SHA，实际视差、有效掩码、最终像素、保留池、保留槽位和几何张量均相等。六种候选方法的历史评分和固定尾部保持一致。

Front 与 Roll20 的新摘要 SHA256 分别为 `e50fda56974cbb312b4704fe1588600160aaaf3c40ed91cbb1bd53739429be5c` 和 `ce6f43fa0ca1ec034b93126fbb350df387480cd349fcbb7549462cb875d0ac4e`。配对读回日志 SHA256 为 `539c6bef3029bf78d49f2e86a0302a9efe0ccef2b3fa7af4ba19f6a78241c3f2`。原始重放存于 Ubuntu 私有目录 `/tmp/dh116-v2-metrics-replay-jTbiXN`。历史 capture 仍缺右侧原生 ID，因此相关真值结果保持 NA。两组相机布局共用两个物理初态，这些结果不授予部署、泛化或训练资格。未启动 Isaac，未进行新采集或训练。

## CPU 回归

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_sample_retention.py tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_wire_geometry.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_error_diagnostics.py
```

新增的两项测试先复现缺少分离指标的失败。最终 Ubuntu 56 项通过，退出码 0，日志 SHA256 `baa93ad96ea1ffd3cd0557bad297a527d2fdc47703886126a80dd1b33a3e22f7`。改变测试参考深度会改变离线精度，保留候选、已选槽位、几何和运行掩码逐项相等。下一步使用独立双侧同步原生参考扩展持出评价和深度辅助监督。
