# Supported17 Tripod：接触 Collector 接线与独立输入审计

基线为 `300306f08b9e35f18f39d90bb6de90789b7b79e7`，分支 `11-17`。本包实际修改原执行链，首次把独立 Contact profile 的 Source、Session、Context、Reward 和原 Collector 连通；CPU 夹具能驱动原 Actor、原 PPO 完成一个窗口和一次更新。SDK、SDF、几何和参数 getter 在该测试中有明确 CPU 替身，**这不是新原生采集、真实接触通过或 Teacher 学会抓取**。Pinch、Power 不借用 Tripod 身份。

## 实际代码

| 文件/函数 | 本包行为 |
|---|---|
| source_priming.PrimedNativeActionHistory.record_priming/priming_event | 对真实 controller 返回的 tuple IDs 作无损 owned 规范化，保留原值、时钟、零动作及原始事件摘要；副本可读，内部篡改会失败 |
| policy_tripod_distance.capture_priming/capture | 验证精确 Primed history、全零 float64 原请求、真实 SDK targets、原几何与 material41；预热无 policy token，不算 PPO 样本；所有 sample 来源摘要随 history binding 更新 |
| sources.NativeContactTripodSources | 原 Reset/live 验证在 SDK2 完成；冻结 S0 后单独预热至 SDK3/rod12；验证参数→几何→提交接触 Context→原 Reward；所有原物理步骤写账本 |
| sources._validate_parameter_event/observe | 参数来源错误在接触进度之前拒收；后继被隔离，无 PPO transition 补零 |
| session / actual_factories / collector | 仅 CONTACT+Tripod 选择真实17槽 Source 和精确 Primed Context，原 Reward 绑定同一 owner；旧 SMOKE 路径保留 |
| preflight.required_capabilities / static_preflight | Contact 数值 gamma=.995，检查真实已加载来源字节；不要求未实现的 full Slip/Hold；Pinch/Power 未实现来源仍拒收，静态通过不授予 live 资格 |
| actual_inputs.audit_teacher_input_sources | 显式 profile，独立回放原始 contact 行，SDK3 起点，实际1～8行 terminal；预热及策略绑定必须同 Reset；第8步触桌同样必须 failure terminal/-40 |

桌面 Antigravity 共进行了四轮真实只读审查，并返回实读源码 SHA。Codex 独立复现了参数先推进计时、最后子步触桌可被伪造为 live bootstrap、预热 getter 和来源摘要缺口，再修正。完整原始意见及运行日志保留服务器私有证据目录；仅摘要/SHA 入库。

## 失败与验证

最终20文件联合CPU回归为 **313 passed、0 failed**（401.31s）。Antigravity只读审查源码和测试，实际测试由Codex在Ubuntu运行。另有Registry/调度器回归 **60 passed、0 failed**（16.73s），与313项联合回归分别报告。新包12项用例覆盖完整原 Collector/一次 CPU PPO 更新、第三子步提前触桌、最后子步伪造非终止、缺失中指 SDF、错误参数、预热原事件所有权、旧预算/旧 Context 拒绝、独立 raw 重放识别同步伪造的 Context/观测/哈希/计时。重复执行结果不累加。

第一轮旧联合回归为301 passed、2 failed、1 error。三处失败均选择了 ed086c 历史 Reset 证明，但其 physical_exchange 源码与当前基线不符；该物理文件本包未改，HEAD 与工作树同为02a773...。仅测试夹具选择独立原生 route-order-v3 证明 c123cc...；原证据未改、未重新绑定旧 SHA，当前严格 loader 已实际通过。增加负例确认历史证明仍被严格拒绝。修正后夹具子集6项通过。这个证明只覆盖 Reset，不证明接触或抓取。

## 预算、保护与后继

本包未启动 Isaac，原生物理/求解器/Teacher/PPO运行均为0。新 Contact horizon 准备按 SDK=`2+1+8N`、solver calls=`1+8N`，不能复用旧 SMOKE 的66/64预算。当前 worker recipe/semantic closure、actual_update 与 actual_artifacts 仍为旧 profile/SDK2 逻辑，**不能从本包直接启动 Contact 原生 worker**。

下一独立包先把原 worker 的 profile、完整来源集合、实际 SDK3/rod12 起点、提前终止与 checkpoint 审计接通并冻结新预算，再做有界原生诊断。Pinch 需自己的 supported17 Reset/Source；Power 需独立五指加掌面来源。三种姿态后续分别做真实接触、固定腕保持、离架 Lift/Hold/固定材料 Slip，仍用30mm/5s/0.5mm标准。

保留固定双RGB、左右手、原24D和371D合同、全部原物理资产和控制器。R2/Force Proxy关闭，正式BC/DAgger/Student PPO及成功专家池继续门控。发布前核验528个受保护文件与三个历史Checkpoint，经Ubuntu提交、Windows导入同一commit后普通push，PR#2保持Draft。
