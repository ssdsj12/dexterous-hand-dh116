# DH116 State-Triggered Touch Grasp Implementation Plan

> 使用 executing-plans，按用户既有 Inline Execution 偏好在当前独立目录执行。原运行进程不热更新。

**Goal:** 到位低速后拇指轻触、食指对向闭合、平滑预载，然后验证真实夹持和抬升。

**Architecture:** 轻触子状态独立于原夹持证据观察器；分配器接收显式法向速度，零食指速度意味着保持实际接触点，不仅冻结关节。运行器只在实测满足条件后接管，并继续应用已有碰撞和反作用力路径。

**Tech Stack:** Python / NumPy / pytest / Isaac / DeformX。

## Global Constraints

- core0、数值库单线程、单 Isaac 实例；不修改原工作树运行版本。
- 位置 2 mm、法向 5 度、两臂和双手最大关节速度 0.02 rad/s，连续 50 ms 到位。
- 轻触目标 0.05 N，有效载荷至少 0.025 N 连续 50 ms；目标增载不超过 1 N/s。
- 保留既有夹持/滑移/抬升证据，禁止绑定，旧 pin 不可伪造兼容。

## Task 1: 独立轻触与到位组件

Files: `src/go2_pvcnn/control/dh116/touch_first.py`, `tests/test_dh116_touch_first.py`。

Interfaces: `ApproachReadiness.update(time_s, position_errors_m, normal_errors_rad, joint_speeds_rad_s)` 返回 ready；`TouchFirst.update(time_s, loads_n, valid_contacts)` 返回每手 phase、目标载荷与 normal_speeds_m_s，输入载荷/接触形状 (2,2)。

- [x] 写失败测试：低速到位去抖、速度缺失拒绝、10s超时；仅拇指推进、接触连续性、失去接触复位、双手独立阶段、预载限速。
- [x] `bash scripts/python.sh -m pytest tests/test_dh116_touch_first.py -q`，先看到新模块缺失，再实现。
- [x] 实现每手 touch/oppose/preload 与显式非有限/负载荷/时间连续性检查；有效接触必须由物理调用者提供。
- [x] 相同命令通过，不声称物理验收；与 Task 2 一并记录提交。

## Task 2: 协同速度分配

Files: `coordinated_closure.py`, `tests/test_dh116_coordinated_closure.py`（上述 src 控制目录及 tests 目录）。

- [x] 新增失败测试，给定速度 `[[0.0005,0],[0.0005,0]]` 时两手食指点三轴速度为零、掌心角速度为零；实际 USD Jacobian 和奇异输入都覆盖。
- [x] 给分配器增加受限显式法向速度输入，默认行为不变；复用 `A=vstack(J.reshape(6,10), .05*W)`，`b=concat((n*speed[:,None]+transport).ravel(), zeros(3))`，求解后检查残差，再对双手统一限速。
- [x] 回归：针对性 55 项通过，完整 579 项通过；记录组件实现提交，运行接线仍未完成。

## Task 3: 状态触发与实际运行接管

Files: `grasp_sequence.py`, `stage_runtime.py`, `scripts/verify_fixed_dual_panda_dh116.py`, 对应 sequence/runtime/runner 测试。

- [ ] 将原固定4秒测试改成实测到位、低速连续50ms；超过4秒不自动闭指，到位等待超过10s退出。
- [ ] 主循环传入两侧实测关节速度及接触有效性；在 approach->close 时用当前测量建立闭指初值，与前一指令按限速平滑接管。
- [ ] touch/oppose/preload 输出接入显式法向速度分配器；预载达标前不启动夹持计时。力与力矩仍从实际后端返回。
- [ ] 恢复重新建立 readiness/轻触证据，不继承旧计时，不更改旧 SHA；测试缺失速度、接管目标、恢复子阶段与旧证据隔离。
- [ ] `bash scripts/python.sh -m pytest tests -q --tb=short`，检查全部失败原因，不放宽安全或证据断言。

## Task 4: 物理验收与交接

- [ ] 确认旧 PID 已退出后，使用新目录、新输出路径和新指纹启动唯一 Isaac 实例。
- [ ] 记录真实轻触/对向闭合/预载/hold/lift，检查接触丢失、滑移、反作用力和抬升高度；失败保留全量证据并定位原因。
- [ ] 更新 README/CHANGELOG：阶段、改进、实际验证、未完成项；按用户要求上传时核对远程 SHA。
