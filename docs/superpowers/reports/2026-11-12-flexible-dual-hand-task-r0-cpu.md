# DH116 无相机柔性双手任务与 R0 CPU 验证

基线为 `2ef7f90ed9af9039dd32fb197063040ae11996f3`。新增 `configs/dh116_tasks/flexible_dual_hand_grasp_v1.json`，SHA256 `8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63`。任务类型为 `deformable_rod`。该配置锁定已有 DH116 和 DeformX 来源，`native_execution_enabled=false`。

## 原始物理和材料来源

| 项目 | 已核对的来源和值 |
| --- | --- |
| DH116 | 原左右 USD 从原始资产清单逐文件 SHA 校验。保留各自碰撞 link 和原始 24D 手指、掌平移、掌旋转合同。 |
| 原始任务与桌面 | `pose_reference.json` 的原桌中心为 `[0.6, 0, 0.25]` m，尺寸 `[0.8, 1.2, 0.1]` m，桌面为世界 Z=0.30 m。原任务配置的掌控制为 `driven_carriage`。 |
| 原手材料 | 原 `grip_compliant` 定义静/动摩擦 1.0/0.8、恢复系数 0、合并模式 max/min。原指腹弹性和阻尼为 10000 N/m、30 N·s/m。原代码明确这些是仿真假设或估算。 |
| 原状态和时钟 | 371D 合同和 24D 动作文件 SHA 校验，时钟 480 Hz 物理、60 Hz 控制。尚无本柔性任务的逐字段原生绑定。 |
| DeformX | 锁定后端提交 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`。实际 Ubuntu cp310 扩展 SHA `c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53`。重新只读校验原生 cooked hull 对应的 10 个实际 SDF 网格及注册摘要 SHA `db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab`。 |

Pinch、Tripod、Power 各有自己的配置，分别要求两侧原 2、3、5 个 distal pad。复用现有 `collider_plan()`，每环境计划使用一根自由杆，两手必须报相同 `rod_uid` 和不同可分辨的材料站位。载入时拒绝固定端点、零重力填值、未经确认的材料值、指腹身份漂移或源文件变化。历史零重力固定端标定只用于已锁定的后端和 SDF 来源。真实柔性重力、长度、半径、质量、密度、弹性、摩擦、solver substep、双手 base DOF、self-collision、桌面碰撞和杆桌耦合等尚未原生读回，均为 `null` 并列入阻断项。独立三种抓法的真实 Reset 与材料站位也尚未资格化。当前不存在 MPC 或 BC 的强制入口依赖。

## Reset 与逐步 R0

`FlexibleGraspEpisode` 接受现有 `DeformableRodStateProvider` 的 schema 和 backend 身份。旧 `FlexibleObject` / `WireRuntime` 仍使用 vendor 路径，未作为锁定 CUDA 后端的替身。episode Reset 要求 rod、joint、controller、filter、contact、camera_history 六个状态所有者的清理回调先执行，再读取新快照。CPU 验证了调用顺序、旧 episode 拒绝、anchor、hold、phase、奖励统计和计数清理。真实 SDK Reset 回调仍待绑定。每条测量与状态共用 env、episode、reset、rod、physics tick、rod tick 和 timestamp；重复、漏步及旧测量都拒绝。

每物理步记录十项 R0。正项为 approach progress、required contact、contact persistence、grasp geometry、lift progress、hold progress。负项为 material-anchor slip、joint limit、instability、external support violation。各项记录原值、单位、来源、有效性、系数、贡献及累计均值、总体标准差、最小/最大、episode sum、有效计数和来源身份。未知值及贡献为 `null`。仅在 CPU 显式注入诊断系数且所有贡献齐全时给出诊断合计。项目配置的系数未获批准，仍为 `null`，`training_reward` 始终为空。R2 关闭，逐指力未填充。现有 `compose_native_reward()` 保持原样，未来只能传入已原生验证且贡献闭合的 R0 结果。

Lift 使用全杆材料节点 Z 的中位值和最低节点相对 Reset 的净上移，并结合实际半径计算最低表面和桌面间隙。物体候选要求两种净上移均达到 30 mm。实际桌面接触、其他外部支撑、authored force、attachment 各需独立有效测量。固定材料站位使用初始 rest arclength，法向弯曲不会改变站位。沿材料方向和指腹切向滑移分别记录。0.5 mm 判据按误差区间判断，无法判定时 Hold 清零。Hold 从第一条符合条件的样本开始计时，必须有 2400 个连续合格物理间隔（2401 个合格采样端点）才输出 CPU success candidate。未知测量、失去对向接触、越限或支撑会打断该连续窗口。

## 验证

先加入失败测试，复现缺少任务入口、旧测量混入、滑移精度不足仍有效、未追踪 authored force 和过早累计 Hold。最后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_lift_hold_window.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py tests/test_privileged_teacher_reward_audit.py
```

160 项 CPU 测试通过，其中 12 项为新任务回归。528 个保护文件和 2 个历史 Checkpoint SHA 核验通过。本包未启动 Isaac 或训练。下一步是让真实柔性运行入口提供同一时刻的材料坐标、逐指 partner、外部支撑和 Reset 读回，并分别验证 Pinch、Tripod、Power 的 371D 字段来源、物理预算和原生抓取结果。
