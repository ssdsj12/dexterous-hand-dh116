# Pinch Teacher PPO 原工厂接线

接通 build_actual_factories()，复用 C04 的 NativeFactories、NativeContextTeacher、原 DeformablePPOTrainer 和 TrainingReward。只接收实际租约签发的同配置、同一 owner 的已验证 live 会话。模型创建前要求 provider 绑定真实来源能力。缺少该能力时拒绝创建模型。工厂自身不签发能力或租约。

Pinch 新场景可在同一实际杆 owner 上生成 Reset mesh plan，再交给 C04，调用继续读取后端验证过的 ropesim.sdf。C04 构造器读取历史单条成对 Pinch 来源时，先把现有已锁定路径解析到真实文件，再交回原读取器逐字节核对 SHA。只在构造器作用域启用。原历史锁、其余文件的路径检查和接触判据保持。

## 已验证

- 定向 CPU 通过 12 项，覆盖工厂顺序、租约/来源身份、实际网格计划接线、路径别名和旧哈希拒绝。原 PPO 的 CPU 物理夹具实算两次优化更新，权重发生有限非零变化。其物理输入是夹具。
- 本批次前置数值 Reset 原生回执另报。pre-action 账本读取实际 371 槽和 66 个有效字段，保持未知 305 槽的 validity 合同。与既有模型、奖励一起通过适用的静态来源核验。
- 工厂集成初次更广 CPU 运行 404 通过、3 失败。2 项是已复现的成对 Pinch 源路径别名，之后定向构造器回归通过。另 1 项由旧测试基线比较后继已发布配置文件触发，保留该结果。

## 晋级阻断

调度器的 actual 分支仍拒绝 actual_artifacts_not_implemented。待接 run_a 与同一策略步真实 raw41→模型的来源见证、实际 journal/Checkpoint 审计，生成独立冻结预算并取得真实父调度租约。本包 native PPO 更新数为 0。30 mm Lift、5 s Hold、0.5 mm Slip 继续各自原生验收。

[Readiness](../../contracts/dh116_teacher_ppo_worker_factories_v1.json)
