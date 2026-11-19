# Pinch v7 实测历史运动回放

使用 Pinch v6 原生 SDK2 Reset 和 SDK3–12 全部 10 条连续子步的逐 link 刚体位姿，将各 link 的世界平移及旋转增量分别作用到 v7 原生 SDK2 Reset。先核对同一场景中的 run、reset、episode 与物理步、左右手 link 路径和碰撞几何 SHA，再复算原地面净空。旧左手失败步 SDK12 的净空为 2.097674 mm，使用 SDK 位姿及原几何重算为 2.097665 mm。每侧使用 17 个 source 和 5 个 locked cooked hull。原保护线 2.100 mm。

| 侧别 | v7 回放最小净空 mm | 距保护线 µm | 限制 link |
| --- | ---: | ---: | --- |
| 左 | 2.147673 | 47.67 | finger22_link |
| 右 | 2.116113 | 16.11 | finger13_link |

反例中，单指腹下降 60 µm 可将初始余量 50 µm 降至 −10 µm。该回放只覆盖已记录的运动，v7 的接触反力、mimic 与实际跟踪仍待验证。v7 注册配方和冻结指纹保持原样，待限额内原生 PPO 诊断。Isaac 新物理步 0，PPO 更新 0。

```bash
python -B -m scripts.audit_dh116_pinch_recentered_tracking \
  --journal "$V6_JOURNAL" --journal-sha256 92fc82b7a52fb911dec02231eb6cd3ce3e2c1496ee7029df5a3dacfe654a2aac \
  --recentered-reset "$V7_RESET" --recentered-reset-sha256 c6e1d6b2cf08608f6047ea32c32d27770a82b25951535f5fdb3857afcc315369 \
  --output "$NEW_AUDIT"
```

25 项 CPU 回归通过。结果 SHA `6300d26081cdd85acd5fc39d5262b97aa5880068a0ecf02f4c3f4d5ea17c84ad`。
