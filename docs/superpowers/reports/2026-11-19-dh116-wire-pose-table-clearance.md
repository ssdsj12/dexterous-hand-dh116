# DH116 三抓法候选初态桌面净空筛查

基线 `822ff0a74d53f4b0cabff74a38a9dbf6706eb66d`。Antigravity 复核了三抓法需要独立物理来源，并区分 Power 的掌面或近端承载。它未能读取上一轮静态桌面报告。Codex 从 Ubuntu 独立读到原始配置、报告与同批候选，确认使用现有 FK、已锁定 hull 和有限桌面盒体 SAT。

`prepare_dh116_wire_pose_candidate.py --audit-table-clearance` 将实际输出的候选手腕和手指状态送入同一几何检查。每手保留 17 个原 USD 凸包、5 个已锁历史 cooked 指腹、逐链接结果和来源 SHA。允许输出失败候选以保留证据，退出码为 2。筛查使用已有 Pinch/Tripod 运行时 2.1 mm 净空作为统一预筛阈值。实际任务配置、手腕、关节、物理和线缆状态均未改写。默认 CLI 行为保留。

| 抓法 | 左手最低净空 | 右手最低净空 | 几何结果 |
|---|---:|---:|---|
| Pinch | 2.111 mm | 2.116 mm | 通过本次静态桌面预筛 |
| Tripod | 1.904 mm | −2.475 mm | 左手无桌面交叉，但低于 2.1 mm 预筛阈值。右手源几何 `finger32_link` 与 `finger33_link` 交叉，锁定 cooked 的 `finger33_link` 交叉 |
| Power | −4.065 mm | −2.475 mm | 两手源几何 `finger32_link` 与 `finger33_link` 交叉，两手锁定 cooked 的 `finger33_link` 交叉 |

净空是几何最低点到桌面的有符号垂直距离。负值结合 SAT 报告用于定位干涉。这些候选来自上一轮锁定输入，均未获得原生接触、PPO 或抓取资格。Tripod 下一步在维持食指、中指可达的前提下调整靠近位置，再验证双手各两组对向反力。Power 仍需专用包覆姿态和真实掌面或近端承载。

测试先复现 3 项缺少 CLI 选项的失败。最终 CPU 回归 96 项通过，包含三抓法的原样本复测、原候选入口和力闭合测试。日志 SHA-256 为 `aeb2310bce2db4051fbe2e9f65477a2f19e0cec18c618e0d03a4101f24866799`。运行命令、复测产物 SHA 和来源见 readiness。新增 Isaac 实例和 PPO 更新均为 0。
