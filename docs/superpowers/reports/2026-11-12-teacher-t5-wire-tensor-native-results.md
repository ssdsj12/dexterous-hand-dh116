# 原 T5 物理任务的 R0 PPO Teacher Wire 读回

本次原生探针沿用现有 T5 刚性任务，加载实际 R0 PPO 的 update 1 冻结 Checkpoint，二指、三指各 1 个环境，共 16 策略步。SHA256 `71ea209f6710db9edc9457e5ed3bfde8cecc9daede4e0dd5eff0167e905e6638` 的 Checkpoint 实测含 1 次 PPO 更新、64 个环境 transitions、2 环境 × 32 的训练 rollout 与 13 个 Adam 状态项。探针未更新模型。

回读的 T5 合同语义 SHA `cc7d9f525f6ebeb04ff97a2ad0ba9afac02767f897ff3f14558c33c1ae56a0d6`，物理 SHA `7128e10e341316dd76132eec5ee1d5af6f769691eecbe281d99a26b974c00612`，原生 `support_deficit` 保留。派生任务文件 SHA `56cef9f2f811918a070fefd9dc766ca9626410ee02b4a118e49773fba43e4aba`。运行 06:44:26–06:45:09 UTC，外层自然退出 0。

| 实测项目 | 结果 |
|---|---|
| 原生时钟与路径 | 128 个连续全局 480 Hz 步，World 1–128、SDK 19–146。256 条环境物理记录的时间、env、episode、策略步和终止逐值对齐。逐步复核 34 个手部 filter actor 和三个静态 collider。0 读回错误。 |
| Wire 与手侧法向 | 双侧 324 个非零同名 pair slot，单侧非零 0，方向余弦约 -1。单 pair 两侧向量和最大 5.96×10⁻⁸ N。 |
| Wire 与手侧摩擦 | 双侧 325 个非零同名 pair slot，单侧非零 0，方向余弦约 -1。全部 8704 个 pair slot 两侧向量和逐元素为 0。 |
| 无过滤 Wire net 减已枚举法向 | 最大残差 2.46×10⁻⁷ N，P95 1.62×10⁻⁷ N。 |
| 无过滤 Wire net 减法向和摩擦 | 最大残差 1.56875 N，P95 1.24273 N。此运行中的 net getter 与法向合计闭合，摩擦来自单独 getter。 |
| 静态接触 | 桌面 256 条法向和摩擦皆零。两个端支架各 256 条法向和摩擦非零，原始法向 Z 分别为 +0.22194–0.86762 N、+0.55868–1.70987 N。 |
| 本地 buffer | 容量 512，手部法向／摩擦最多占用 15／14 槽，两端支架各最多 4 槽。当前证据无法排除上游截断。 |

常规组合场景枚举仍只有 8 个 CollisionAPI prim。该枚举未发现未归类碰撞 prim、PhysxForceAPI prim 或 92 个 joint/attachment 候选中指向 Wire 的关系。两个 Wire WrenchComposer 在 128 次 post-step 读回都为 inactive、零力和零力矩。完整实例代理、运行时 SDK 写入、上游容量、无过滤摩擦和外部来源仍待独立核验。此次为约 0.267 秒的受支撑窗口，0 个完整首回合、0 次原生成功。正式力学、专家准入和 Teacher 成功资格均保持关闭。

## SHA256 与回归

| 产物 | SHA256 |
|---|---|
| Wire Tensor gzip | `b24aaac985d5dc7045bf2df4b43b31ea24223f83429526fba2a8c0ac4e3925a2` |
| 合成场景及 filter 清单 | `3d3d58ec3459264d5b36e977d5f2b4ab9160c8157fcb9d828a42ca85aedfaaea` |
| 原生物理账本 | `a0a31b95b4dd6b68f7ea036ddb17b0da7a5b7b1466d8d55451236e88f1d1ad3e` |
| 回调完成文件 | `966a5c392b0deae35ee724425996596c6ec07b4dc56b409d25fc014eddaa3481` |
| 外层日志 | `95d17802090a876de329f109616397b4b68e3127fef355a1f6d8ccac1d404402` |
| 冻结源码清单 | `48a8f0a2d0287b852f00cc207fd9eee8bb76c7aae0829ecda5a0f25f4472bdc6` |
| 独立审计结果 | `87eff980ae943a15b8da3c59f776ece15202a882997f7fad6addcd87493e0486` |

Ubuntu 原始目录为 `teacher-t5-wire-gpu-probe-20261112-DsBgLZ`，私有独立审计目录为 `/tmp/dh116-wire-t5-native-audit-I5SnUL`。Ubuntu CPU 回归 104 项通过，0 失败，日志 SHA256 `7c603adc1122a72d96dfe013ba68a54e06c5cb01c880578bce2b8ed645988abc`。本报告只读复核已有原生证据。
