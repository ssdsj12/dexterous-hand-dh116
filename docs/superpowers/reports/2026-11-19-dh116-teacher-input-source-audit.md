# DH116 Teacher 检查点前的原始输入与父进程截止审计

本工作包从 `11-17` 的 `d4aa566e2a6be97ef9d8b98f121784d7f3c2d42d` 开始。Antigravity 通过只读 SSH 做了三轮复核。独立比对 `tabletop_native_teacher_sources.py` 的 SHA-256 为 `ef6b8422bbb6fda6cc2d5c5fd9d93a4ec5225d4156dab18c3eed55826eed7f53`，Collector SHA-256 为 `1ca80dca5ed79a8bb0cb90081f8bd253d1c42f5d6763c67235a0e85b707252d8`。原 Session 的 SHA-256 为 `b38b7273f56d0a2ce470455f88d8e1c38ca1a1438cc554e5b33711dc5b35486b`。

复核确认，C04 owner 必须在已有物理场景和 R06 的 NativeOwners 范围内创建。Collector 内的即时哈希检查还缺原始物理行的事后对照。另外，父调度器在启动前计时，子 SessionBudget 在建立静态合同后计时，因此需要显式继承父进程截止。

## 检查点封存前的输入审核

`audit_teacher_input_sources` 读取不可变的训练账本字节。用原始 Reset 的杆和参数、各控制窗最后一个实际物理行的 `rod`、`material_director`、`rod_boundary`，独立核对 41 节点、状态和参数。动作决策、末端价值估计分别有独占的来源序号。使用原打包算术重建 371D 原始状态与有效位、41×13 节点特征、节点有效位、任务 context 与有效位。真实终止要求没有 bootstrap；预算截断仍要求同一 pre-reset 端点的原始 bootstrap。所有检查通过后，已验证的账本前缀 SHA 进入检查点封存事件。

`run_fresh_update` 只在真实 C04 所用的 `owner_source_unqualified` 范围进入此审核。旧 CPU rollout 的设施行为保留。新增审核会在检查点写入前拒绝原始输入不符，保留已有账本。它不认证真正的 GPU 所有权或外层退出状态。

## 运行预算

`build_actual_factories` 在构建实际场景前把租约的绝对截止绑定进原 `SessionBudget`。采样剩余时间为父租约和本地预算的较短者。过期租约不构建场景。既有非实际工厂继续使用原预算。原 24D 动作、奖励、PPO 更新算法、碰撞和验收阈值保持。

## 原生状态

本轮新增 Isaac、PPO 和真实抓取次数均为 0。Pinch 后续仍需在 `run_a` 内组合实际 SDK/DeformX 同设备采集、来源所有权、现有更新与 R06 自然退出。Tripod 仍需独立初态净空和双侧三指同钟反力；Power 仍需独立初态净空与掌面承载、五指包覆。抓取三种模式分别以净抬升至少 30 mm、连续保持至少 5 s、固定锚滑移不超过 0.5 mm 验收。

## 测试结果

原始来源新增 9 个回归，父截止新增 2 个回归，均在对应修复前复现失败。最终联合回归 231 项通过，包含新增测试、原 PPO 更新、原生情景对象工厂、旧 C06 产物审计、三指对向、R06 生命周期和调度器。测试使用 CPU 夹具。

命令在 Ubuntu worktree 以项目解释器与原解析环境执行。

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_actual_inputs.py tests/test_native_teacher_actual_update.py tests/test_native_teacher_training.py tests/test_native_teacher_actual_factories.py tests/test_native_teacher_worker_artifacts.py tests/test_dh116_work_packages.py tests/test_teacher_force_closure.py tests/test_fault_safe_lifecycle_v3.py tests/test_native_teacher_session.py
```

最终日志 SHA-256 为 `d33ee5064ac1785db7a7de4029a064fb7c8aec1206b1bb9d6c3857911d693e0f`。
