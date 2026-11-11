# DH116 工作包执行计划

基线为开发分支 `codex/dh116-stereo-privileged-dagger` 的 `3c80166da75fb5496701c9f3ff673296c5db5615`。最终验收分别在刚性代理和柔性 Cosserat 任务上执行，每种抓法至少 100 个独立原生初态，Lift ≥30 mm，Hold ≥5 s，Slip ≤0.5 mm。评估须排除外部支撑、Teacher/MPC 接管和仿真特权泄漏。

1. 在 `dh116_work_package_registry.json` 登记依赖、CPU 配方、待集成代码和原生/外部阻断。`dh116_release_manifest.json` 只记录已实际推送并读回的工作包提交。修改包先提交，再推送并读回真实 SHA，最后独立提交发布清单；清单记录它所验证的工作包提交。
2. `scripts/run_dh116_work_packages.py --plan` 只读检查当前与远端 HEAD、PR #2、工作区和可并行工作包。`--run --limit N --jobs J` 只运行有限个就绪 CPU 配方。测试可并行，提交与推送串行。已完成的开发改动以 `--prepared PACKAGE --file PATH ...` 接入，文件清单必须与工作区改动完全一致。
3. 测试前后核对保护源文件和历史 Checkpoint。报告记录实际命令、通过/失败数、退出码、私有日志 SHA 和后继依赖。测试失败保留日志与工作文件，只发布已审查的失败报告与状态。中断后用 `--resume` 恢复尚未核实的工作包或清单推送。正常推送发生远端分歧时停止发布并保留证据。
4. 并行优先推进 WP1 原色采样槽保留、WP2 训练模式及来源门槛、WP3 Teacher 离线合同与 WP4 PhysX 来源回归、WP5 Solver 来源、WP6 Actor-only 部署验证。新算法由独立工作区实际编写、测试并交给调度器，注册表中的已有 CPU 配方负责验证当前源码。
5. 原生同步采集、长时间训练、依赖安装和物理修改须独立批准。先准备单实例 PhysX 自然退出，后续接触资格依据逐 contact 的 A/B、作用侧、单位、切法向和外部支撑证据。MPC 371D/24D 同步重采是 Actor-only BC 的前置条件。真实 Solver 仓库、许可证和独立 Toolkit 12.8/SM89 构建是柔性任务验证的前置条件。正式 Teacher、BC/DAgger/Student PPO、最终原生验收逐级按证据晋级。

## Ubuntu 工作区与本机发布

Ubuntu 集成 Worktree 为 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration`。同目录下 `autowp1` 至 `autowp6` 分别使用 `codex/dh116-autowpN-20261111`。只在任务分支开发，经 CPU 验证后按明确文件范围接入集成分支。原始 Checkpoint 和原始数据按既有路径只读引用。

本机只进行 GitHub 发布，使用无工作区的裸仓库导入已提交的 Ubuntu Git 对象。依次推送、fetch 和核验提交、树 SHA、文件清单及 Draft PR。当前 Ubuntu 上的 CPU 运行用 `--local-only --evidence-dir /home/xk/.local/share/dh116-autonomous-3c80166/evidence` 形成待发布队列。远端验证结果回填发布清单，且发布清单本身也经过远端读回。
