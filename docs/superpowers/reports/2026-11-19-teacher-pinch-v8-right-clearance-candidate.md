# Pinch v8 右拇指主动与被动分解和新 Reset 候选

Antigravity 与独立复查均使用 11 个实际关节和 `base_link` 实测姿态。模型保持 Reset 被动联动残差，将主动关节变化映射到完整手树，再单独代入实测被动关节。SDK 目标是理想跟踪反事实。SDK 第 7 步 `finger13_link` 的主要贡献为实测主动关节变化 **−20.867 µm**，新增被动残差为 **+0.0017 µm**，全部实测关节的 FK 与原始凸包净空读回相差 **−0.034 µm**。同一步 SDK 目标代入产生 **−86.768 µm** 的反事实净空变化。

固定旧左手及原自由杆。右手初选上移 100 µm 后，虽可回放实测 5 步，SDK 目标反事实净空仅 2.069570 mm，留存为被拒证据。正式候选右腕上移 150 µm、世界 X −75 µm、绕世界 Y −0.001 rad，并调整右拇指初始屈曲 +0.001 rad。以现有实际凸包、原关节限制、双手独立扫掠和拇指与食指对向几何重新检验。

| 检查 | 候选结果 |
| --- | ---: |
| 右手连续几何路径桌面下界 | 2.238694 mm |
| v8 5 步实测关节运动回放最小值 | 2.223173 mm |
| v8 同 5 步 SDK 目标反事实最小值 | 2.157236 mm |
| 右手要求指腹终点最大间隙 | 0.034735 mm |

原 2.100 mm 保护线、杆、关节限制、动作语义及成功标准保持原样。CPU 候选无原生资格。新的 Reset-only 冻结指纹 `a8a447b8b68d9b8fb8569de2c21c41df87701031892ad3dd16f990e7ed6c0109`。需按新原始 Reset 读回与自然退出独立验收，再生成下一份 PPO 配方。

```bash
python -B -m scripts.prepare_dh116_pinch_v8_clearance_candidate \
  --journal "$V8_JOURNAL" --journal-sha256 f86e670f7912d59d30a7a68f5e57f3c1c8ab759e3ee8c9db0a0151299eafb88b \
  --pair "$PARENT_PAIR" --pair-sha256 eb60404a0f9a889edfd377e1ef59d989e0ca1498ec45e7e636810678a9cad063 \
  --output "$AUDIT" --candidate-output "$NEW_PAIR"
```

24 项针对性回归通过。另两项要求历史原生源与当前运行时代码字节完全一致的夹具失败。在基线 `dc6803f` 的同一脚本上独立复现，两项均报 `exact measured geometry implementation/source identity required`。保留失败日志。最终审计 SHA `69123c80b790fc84b0e949d8afcf7b8b4bcf00ed42f8227161a4b51dfe18a006`，新 pair SHA `1d8e048de67f337f3bf41ca1e15f5d3ea4d255139f5a71f0a11ce481737c7cba`。本包执行的原生步数与 PPO 更新均为 0。
