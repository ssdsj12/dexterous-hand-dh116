# DH116 保持预载与双臂共同平移核心

阶段 D4；父节点 [DH116 抓取](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty。按[计划](../../docs/superpowers/plans/2026-10-15-dh116-held-transport.md) Inline 执行，不提交，不热更已运行的 pilot6。

## 已验证的代码变更

- `coordinated_closure_increment` 增加默认零的显式共同世界平移速度，≤.01m/s。默认零与旧调用逐元素一致；保留原9约束10变量、奇异拒绝、臂.05/手.02rad/s限速。
- `coordinated_bimanual_increment` 两侧先求未饱和候选，再只做一次共同缩放，避免独立限速导致左右抬升不同速。实际新左右 USD 点J/掌心角J对照通过；不提供真实碰撞/保持资格。
- `GraspEvidenceGate` 增加可选显式预载门，仅加强 close→hold 入口。默认5mN旧行为不变；显式目标必须[.005,3]N。达到预载后仍须原.1s接触和1s保持、2mm滑移、双区30mm抬升1s。未来连续运行拟用新目标90%作为预载，不把刚接触等同可抬升。
- 运行脚本把已完成点/J探针持久化移入finally，后续碰撞失败也保留已测结果；仍将整体passed保持False。这项改变没有进入 pilot6，不能追称其失败后的未保存探针已恢复。

## 测试与边界

单核、数值库单线程，真实 USD 环境：

- 平移参数先RED 4失败，后核心16通过（含真实左右 USD）；首个RED环境没有pxr，实际USD用完整环境GREEN验证。
- 双臂共同缩放先RED 4失败，后20通过，再补真实双手同速对照；合并358通过。
- 预载先RED 5失败，后观察器/材料22通过。
- 失败保存探针先RED 1失败/12通过；最终 DH116 + moving-contact **364 passed in22.56s**，`/tmp/dh116-transport-preload-preservation-full.log`。

以上是纯控制候选、观察器与接线验证，不是实际保持或抬升。独立切触pilot6已在2483步被左食指非前区1mm保护停止；目前不晋级实际抬升，先筛出有足够闭指余量的姿态。连续接近到抬升的状态机尚未完整接入运行脚本。
