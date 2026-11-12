# DH116 工作包执行计划

基线为开发分支 `codex/dh116-stereo-privileged-dagger` 的 `3c80166da75fb5496701c9f3ff673296c5db5615`。最终验收分别在刚性代理和柔性 Cosserat 任务上执行，每种抓法至少 100 个独立原生初态，Lift ≥30 mm，Hold ≥5 s，Slip ≤0.5 mm。评估须排除外部支撑、Teacher/MPC 接管和仿真特权泄漏。

1. 在 `dh116_work_package_registry.json` 登记依赖、CPU 配方、待集成代码和原生/外部阻断。`dh116_release_manifest.json` 只记录已实际推送并读回的工作包提交。修改包先提交，再推送并读回真实 SHA，最后独立提交发布清单；清单记录它所验证的工作包提交。
2. `scripts/run_dh116_work_packages.py --plan` 只读检查当前 HEAD、本地已核验的远端跟踪引用、工作区和可并行工作包。`--run --limit N --jobs J` 只运行有限个就绪 CPU 配方。测试可并行，提交与推送串行。已完成的开发改动以 `--prepared PACKAGE --file PATH ...` 接入，文件清单必须与工作区改动完全一致。
3. 测试前后核对保护源文件和历史 Checkpoint。报告记录实际命令、通过/失败数、退出码、私有日志 SHA 和后继依赖。测试失败保留日志与工作文件，只发布已审查的失败报告与状态。中断后在 Ubuntu 用 `--resume` 恢复本地提交。Windows 推送遇到远端分歧时停止发布并保留证据。
4. 并行优先推进 WP1 原色采样槽保留、WP2 训练模式及来源门槛、WP3 Teacher 离线合同与 WP4 PhysX 来源回归、WP5 Solver 来源、WP6 Actor-only 部署验证。新算法由独立工作区实际编写、测试并交给调度器，注册表中的已有 CPU 配方负责验证当前源码。
5. 原生同步采集、长时间训练、依赖安装和物理修改须独立批准。先准备单实例 PhysX 自然退出，后续接触资格依据逐 contact 的 A/B、作用侧、单位、切法向和外部支撑证据。MPC 371D/24D 同步重采是 Actor-only BC 的前置条件。真实 Solver 仓库、许可证和独立 Toolkit 12.8/SM89 构建是柔性任务验证的前置条件。正式 Teacher、BC/DAgger/Student PPO、最终原生验收逐级按证据晋级。

## DeformX 柔性线缆三种抓法闭环

自 2026-11-12 起，默认 `--plan` 和 `--run` 选择 `flex.f10-actor-only` 及其实际依赖。F0 固定源码，F1 构建，F2 验证 standalone GPU，F3 验证单刚体双向耦合。F4 依据 DH116 原始碰撞几何逐级验证单指、单手和双手接触同一根杆。F5 视觉数据和 F6 资产可独立推进。

F7 生成时钟一致的柔性状态和双目数据，F8 分别训练 Pinch、Tripod、Power 特权 Teacher，F9 验证三个 Student 的 BC、DAgger 和 PPO，F10 验收 Actor-only。保留 371D 公共合同、canonical 24D Tanh Gaussian 和原生阈值。新增柔性状态使用独立版本。力学来源未经原生验证时保持 R2 和部署力反馈关闭。

`--plan --scope all` 查看旧任务和完整回归。原 `solid_cylinder` 配置、USD、物理参数、Checkpoint、动作合同和验收证据继续作为独立刚性回归。它的训练状态独立于柔性流程。

## Ubuntu 开发与 Windows 发布

Ubuntu 集成 Worktree 为 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration`。同目录下 `autowp1` 至 `autowp6` 分别使用 `codex/dh116-autowpN-20261111`。只在任务分支开发，经 CPU 验证后按明确文件范围接入集成分支。原始 Checkpoint 和原始数据按既有路径只读引用。

Ubuntu 调度器从本地已核验的远端跟踪引用或待发布队列尾部确认 HEAD，运行 CPU 工作包并提交到队列。默认 `--plan`、`--run`、`--prepared` 和 `--resume` 均为本地操作。

1. 从 Ubuntu 用 `--relay-plan --evidence-dir /home/xk/.local/share/dh116-autonomous-3c80166/evidence` 导出待发布提交链，再打包相应 Git 对象。
2. 在 Windows 裸仓库核对仓库、目标分支、提交父链、树 SHA、文件清单和 PR #2 Draft 状态，按顺序推送并逐提交读回远端。
3. 向 Ubuntu 的 `--relay-ack` 回传已核验回执。Ubuntu 写入发布清单并排队清单提交，由 Windows 再推送和核验。第二次回执清空队列并更新 Ubuntu 的本地远端跟踪引用。
