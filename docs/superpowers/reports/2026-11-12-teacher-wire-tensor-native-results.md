# 原 GPU 任务 Wire 对象侧 PhysX Tensor 原生结果

Ubuntu 原始记录位于私有目录 `teacher-wire-gpu-tensor-20261112-wyaUbE`，源码任务提交 `d8e74fa3805ddf0a908456af2c5e29f443a2c7f9`。运行时间 06:35:47–06:36:33 UTC，外层自然退出 0。沿用第 1 次 PPO 更新的历史 Teacher Checkpoint，二指、三指各 1 环境，共 16 策略步。原 GPU reset 和全量报告设置沿用基线。无 PPO 更新、完整首回合或原生抓取成功。

## 原生样本复核

完整 128 个连续全局物理步，World 1–128、SDK 19–146，时间间隔在 1/480 秒容差内。探针记录的 256 条逐环境身份、时钟、策略步与终止字段均和原生物理账本逐值对齐。每个物理步验证了返回的两个 Wire sensor actor、逐环境 34 个手部 filter actor 和三个静态 filter collider 的实际顺序。全部 getter 通过 shape、局部 buffer ownership 及法向矩阵闭合校验，0 读取错误。每个 view 容量 512，手部法向与摩擦各最多占用 12 槽，两个端支架各最多 4 槽，桌面 0。上游容量未独立标定。

| GPU 读回项 | 原生证据 |
|---|---|
| Wire 对手部法向 | 256 个双侧均非零的 filter pair 时刻，方向余弦为 -1 至数值舍入误差。Wire＋手侧单 pair 最大残差 2.46×10⁻⁷ N。 |
| Wire 对手部摩擦 | 256 个双侧均非零的 filter pair 时刻，方向余弦约 -1。全部 8704 个 pair slot 双侧相加逐元素为零。 |
| 无 filter Wire net 减已枚举法向 | 256 条逐环境记录的残差 P95 为 4.15×10⁻⁷ N，最大 6.50×10⁻⁷ N。 |
| 无 filter Wire net 减法向与摩擦 | 残差 P95 为 5.8093 N，最大 6.3894 N。该 GPU 运行的 net getter 与法向合计闭合，摩擦须从独立 getter 读取。 |
| 负 Y 端支架 | 256 条法向与摩擦均非零，原生法向 Z 为 +0.5459 至 +1.7718 N。 |
| 正 Y 端支架 | 256 条法向与摩擦均非零，原生法向 Z 为 +0.5737 至 +4.4133 N。 |
| 桌面 | 256 条法向和摩擦均为 0，两个端支架仍持续承载。 |

组合场景常规 `Stage.Traverse` 枚举 8 个 CollisionAPI prim，当前枚举没有未归类碰撞 prim 或 PhysxForceAPI prim，92 个 joint/attachment 候选均未记录指向 Wire 的关系。常规遍历会遗漏 instance proxy，碰撞排除范围仍需独立核验。两个 Wire WrenchComposer 在 128 次 post-step 读回均为 inactive、零力／零力矩。瞬时 composer 可在写入后清零，仍需检查 pre-step 与实际 SDK 调用。未知外部 partner、可能抵消的法向、无 filter 摩擦、全程外力和附着覆盖均不能从本轮证据排除。

实际同一 GPU 任务的对象侧和手侧法向／摩擦方向相反，支持了对象作用侧在本次样本中的判读。正式力学资格仍要求独立 world/SI 与全场景来源标定、完整且无截断的同回合逐 480 Hz 读回，以及实际 Lift/Hold 期间的无外部支撑和外力证明。本轮只记录了约 0.267 秒的受支撑窗口。正式 force/support、Teacher、专家池和 Student 资格均为 0。

## 证据与回归

| 产物 | SHA256 |
|---|---|
| Wire Tensor 原始 gzip | `bdfb000f3a6986d06cffe5660e994787f99f12899df114ac5c5e3a0e7e6cabfb` |
| 合成场景与精确路径清单 | `3d3d58ec3459264d5b36e977d5f2b4ab9160c8157fcb9d828a42ca85aedfaaea` |
| 回调完成文件 | `40e7ffbfd08e09ff707927ca14872b935e7adef9e75c3a65dc6a6facedd631eb` |
| 原生物理账本 | `b9df6b687b532bbc9d978f850aaa5dd1570a507a4b9cd4ba9a3942a745956df8` |
| 外层日志 | `0c826a0009c7785a2188d027be1d4c615c257311943c62a7fdc6fd769e40d45d` |
| 冻结源码清单 | `ca73c74c051153760128eb7417bddb0a93b8b3ea332b487ef77c93d9fe08b51c` |
| 独立只读核验脚本 | `6fe087d35a726f1a5034489f2953092323b01a72b6253233117a3c608767da2b` |
| 独立只读核验结果 | `f96216609afd7cde5430c54a5aa88fb715072eb911e86daff77b2f272520dae5` |

Ubuntu 私有核验目录 `/tmp/dh116-wire-tensor-native-audit-xcv1WS` 保存逐项分析。CPU 回归 104 项通过、0 失败，日志 SHA256 `2980432241c3c601b2c9ec96cbcbac550839a821cc8f95c3cd1ee2bad1666115`。528 个保护文件和 2 个历史 Checkpoint 再核对通过。
