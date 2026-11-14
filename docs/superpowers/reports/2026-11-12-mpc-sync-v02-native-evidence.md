# MPC 371D/24D 同步采集 v02 原生验证

| 项目 | 实测结果 |
|---|---|
| 启停 | 2026-11-12 04:38:52–04:39:59 UTC，外层自然退出 0 |
| 冻结源码 | `8fad3724d9d9cdff962e0a42ec38cb319e2c5873` |
| 控制路径 | `solid_cylinder`，`pressure_geometry_constrained`，原始固定 224×96 RGB-D |
| 执行预算 | 二指和三指各 1 个环境，各 32 个 60 Hz 策略步，共 64 条 371D/24D 配对 |
| 物理证据 | 256 个全局 480 Hz 步，512 条环境记录，每条配对对应连续 8 个物理子步 |
| 相机证据 | 32 个双环境 RGB-D 帧，逐环境标定 SHA 与对应 pre-action 时钟已绑定 |
| 状态有效性 | 每条观测的 24 个主动关节 q/qd 字段通过逐值对照，其余 347 个字段保留原值及未资格化掩码 |
| 结果 | 0 个终止首回合，0 个原生成功，0 个 PPO 更新，0 个新专家轨迹 |

v02 重试与 [v01](2026-11-12-mpc-sync-v01-recorder-failure.md) 的 73 项预检查参数逐项一致，独立输出路径除外。通过原控制入口读取并执行同一历史参考数据，参考 SHA-256 为 `f4fa715c58ce0d767588a6f2bf5aed248d62d22833d4794a97343baf31b8aaf9`。训练入口、采集入口、Teacher 采集器、物理环境、任务和 MPC 480 Hz 记录器与基线逐字节一致。新增同步归档、原视觉执行记录与 512 个原生子步中的 issued 24D 动作逐值一致。原 constrained MPC 本身有 25 条 active、32 条相对参考有改变的动作记录，保留这些原执行动作作为配对标签。

## 右 finger12

每环境、每手各 256 条有效 480 Hz 记录，原始传感器档案保存了每手全部 11 个主动及被动关节的实测 q/qd。下表右 finger12 对应 canonical 24D 索引 7。两组距名义下界的最小余量均出现在 world step 1 / SDK step 19 / policy step 0。

| 抓法 | env_id | episode_uid | 实测 q 范围（rad） | 最小名义余量（rad） | 最大实际目标跟踪误差绝对值（rad） |
|---|---:|---|---:|---:|---:|
| 二指 | 0 | `555c2480-62fe-4c75-8205-93a48fbef7f8:0:0` | 0.030402111–0.031614888 | 0.022489488 | 0.000048131 |
| 三指 | 1 | `555c2480-62fe-4c75-8205-93a48fbef7f8:1:0` | 0.031918358–0.036347244 | 0.024005735 | 0.000256956 |

本次短采集的 512 条记录均未出现右 finger12 越限，可用作该两个初态的 MPC 对照证据。Teacher 随机动作和确定性 mean 尚缺相同物理初态的配对证据。旧 Teacher128 的五个终止窗口继续单独保留，根因待同初态实验验证。

## 原始证据与下一步

原始数据保存在 `/mnt/mydisk/xk/dh116/runs/wp3_mpc_privileged_sync32_v02`，外层证据和审计脚本在 `/tmp/dh116-mpc-sync32-v02-execution-20261112-VbJQwV`。

| 证据 | SHA-256 |
|---|---|
| 外层 `launcher.log` | `c178d61f4b3286183d847699dab52a247cd2d51e10bb39d21b8c3d3c34ae876a` |
| `mpc_privileged_pre_action_unqualified.pt` | `5f9a419049121a61caa451a03878c6d5be673f94b2fa98b28bf6bca56d0fdfed` |
| `all_executed_supervision.pt` | `131a4a192d2b0861c35907c86079c1b593f851a72964e5a44577f8178d0b1212` |
| `mpc_state_capture/states_480hz.jsonl.gz` | `ea69e89c1fb04ad2aac67d194283ba341aaee48ed9c2b883d4e23662a88cba22` |
| `mpc_state_capture/sensor_evidence_480hz.jsonl.gz` | `88a83461d1b3d6b814b6e58426da4b62d158a85113cb2baa7a37c52cd643fe59` |
| `raw_rgbd/manifest.json` | `f5c51abfeca64e990d0aa96b02ffbca640a7a1aeb40d0a5e5f9152937a3ede04` |
| `audit-independent.log` | `ae45e3f9be265ea0879d35ba7efd1b10e0559f9d5c85e4910f7804c762b49600` |

CPU 回归见 v01 修复报告，30 项通过。原生证据审计使用服务器现有解释器运行 `audit-independent.py`，核对外层退出、原始 SHA、同步时间、观测、动作、关节和保留门控，退出 0。数据链路验收仅覆盖这两个诊断初态。产物标记 `qualified_for_bc=false`，原始 native 报告也未授予新的成功资格。

完整 MPC 采集可按历史 [T5 reference](2026-11-08-stage5-native-validation.md) 冻结原路径：二指、三指各 4 环境，最多 1800 策略步，首回合独立终止。历史报告记录二指 4/4、三指 0/4。只读目录为 `/mnt/mydisk/xk/dh116/runs/stage5_physical_mapping_20261108T0840Z`。根目录的 `t5_teacher_reference_launch.json` SHA-256 `d232f5be63367aac58daa805f9af32ad025e9cda3a2d575faf9d3c9a0eed41c5`、`t5_teacher_reference/initial_native_state.pt` SHA-256 `7a8d7987b7c0f72d441244f746ad8dbf82f826f6e2e56066fe9523f9352ca67e`、`t5_teacher_reference/teacher_collection_report.json` SHA-256 `e4c08f5e71d29e38e9cb626cb27f1e31dc1c7344968e48b032c98ee6641391da`。

现有同步记录器上限为 180 策略步，且独立核验范围为 24 个 q/qd 状态字段。完整成功采集前需单独准备长回合流式保存、完整 371D 来源与掩码、实际下发动作、首回合原生终止及成功和失败分流。成功资格保留净抬升 30 mm、连续保持 5 s 和固定锚滑移不超过 0.5 mm。完整回合需独立预算与批准，原始路径和数据只读。
