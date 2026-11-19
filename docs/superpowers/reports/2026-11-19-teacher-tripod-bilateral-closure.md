# Tripod 双手连续预闭合参考检查

直接从通过的静态终点固定腕位回退 0.0025 rad，左 `source:finger22_link` 的桌面净空由 2.100010 mm 降到 1.964982 mm。修改参考路径：预张开 0.01 rad，同时将起点腕抬高 0.7 mm；闭合时同步降低。左终点另作世界 X −0.020 mm 和 Z +0.010 mm 微调，每侧仍重验真实凸包指腹终点。位置与材料轴沿同一根 41 节点线缆，借鉴了 OakInk 的物体局部表述。 [OakInk 数据说明](https://github.com/oakink/OakInk/blob/main/docs/datasets.md)

复用 Antigravity 指出的 `compile_actions` ，逐侧写入原 24D 的独立槽位后叠加。命令为 20 个 60 Hz 控制区间、160 个 480 Hz 参考步，闭合角速率 0.03 rad/s，腕竖向速率 0.0021 m/s。该脚本用原控制器参考重放检查投影，用全 source/cooked 凸包的逐 link 运动界和保守的折线杆近点检查连续区间。认证对理想目标跟踪成立。

| 侧别 | 连续桌面净空下界 mm | 连续杆非穿透间隙下界 mm | 终点最大指腹间隙 mm | 终点两对方向点积 |
| --- | ---: | ---: | ---: | --- |
| 左 | 2.102592 | 0.033081 | 0.498702 | −0.334125 / −0.201937 |
| 右 | 2.187334 | 0.000145 | 0.498801 | −0.295785 / −0.708533 |

双侧扫掠沿 Y 的证书分离下界为 231.847706 mm，两侧各 0 次参考投影截断。右手杆净空余量仅约 0.145 µm，还需检查实际跟踪误差，同手异指碰撞以及原生六指腹接触反力。现有旧 Tripod 原生入口只驱动左手，独立双侧诊断不能直接复用该配方签发资格。

```bash
python -B -m scripts.audit_dh116_tripod_bilateral_closure --endpoint "$PAIR_ENDPOINT" \
  --endpoint-sha256 16fcae86fcdf1a10f914d99d2cbc487dca1a44b359cf7584259cf7adb4a6f146 \
  --output "$NEW_REFERENCE_AUDIT"
```

报告源数据 SHA `0e1d183f2fcf8e9152bea1f671fc34fe8fe749a69dcffc2eb1a29f882204be7e`。31 项相关 CPU 回归通过。扩展套件含 8 个历史单侧 exchange 失败，原因是历史 `/home/xk/.local/share` 已解析到另一绝对路径，`read_locked` 按原有规则拒绝路径别名。保留失败日志和原产物。Isaac 实际步 0，PPO 更新 0。后续需在独立 Io/反力路径同时验证六指腹、杆速度与脱离支撑，再谈 Lift / Hold。
