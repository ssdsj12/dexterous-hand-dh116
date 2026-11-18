# Pinch Teacher PPO 场景交接

基线为 `ca73112686de6e805ae468397189660ddc6148e1`。延续已批准的 Actual Native Provider 计划，完成 L02 的 Pinch 新场景交接部分。

## 代码变化

- `tabletop_dual_pinch_builder.build_fresh_owned` 复用原 Pinch 场景、双手 Articulation、原 SDF 和单根 DeformX 杆。SDK setup 观察仍为 0→2，返回时原物理 episode 处于 `new`。原 `build_reset_owned` 仍按原流程执行一次 episode Reset。
- `tabletop_native_teacher_session.build_fresh_pinch_owner` 在已校验的观测后端 scope 构建场景，并把原 episode、setup 事件、几何回调交给同一个 `NativeEpisodeOwner`。施工失败的 setup 证据保存在 ownership 记录，Teacher 训练账本中的 setup 事件由原 owner 写入一次。阶段仍为 static → Reset → live → policy → optimizer/reward。
- Teacher 的初始几何回调要求实际 Reset 后且第一个动作之前。它读取原物理 episode 的同钟杆快照，验证 run、环境、episode、Reset、rod、SDK 时刻、rod tick 及 41 节点形状。原诊断入口保留原 proposal 几何路径。净空判据与现有原始保护沿用。

## 验证边界

CPU 夹具通过旧入口的一次 Reset、本入口中把 Reset 交给原 episode、同钟快照拒绝、后端临时切换后恢复和原单账本工厂接入。自测使用已安装 USD Python 工具和 CPU 替身，没有启动 Isaac、GPU Solver 或 PPO。之前 0/34 严格拓扑结果和历史 Reset 来源 SHA 拒绝仍有各自记录。

本包提供场景和 owner 构建接口。真正 A worker、同一策略步的 raw41 与模型输入证据、optimizer update、checkpoint 和自然退出仍待接通并验证。Teacher 准入继续以真实 Reset、物理证据、实际 worker 来源与批准预算为准。

## 回归结果

- 直接相关的四组回归共 46 个不同测试通过。原 Pinch 构建器组 19 个，观测后端作用域 1 个，C04 会话工厂 1 个，使用项目 `scripts/python.sh` 的 L01 实进程组 25 个。
- 更宽的初次回归为 298 通过、38 失败、3 个执行错误。标准脚本复验通过该批中的 20 个失败启动案例。其余案例中已核验的源锁阻断包括历史 R08 清单锁定的 builder SHA 与本次修改前的 HEAD 已有差异，以及 `/home/xk` 的证据路径经过别名。后者读取出的文件 SHA 与既有锁完全一致，但历史读取器拒绝非规范路径。历史源锁保留，无原生资格变更。
