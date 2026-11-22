# 三抓法Teacher服务器核查：真实入口与学习信号

核查基线57a25193b93b5e86d7f5e945eb7eaa891c149854，Ubuntu分支11-17与Windows读取的GitHub远端一致，PR#2仍Draft/open/unmerged。用户选择本轮仅服务器核查，因此没有恢复Antigravity桌面，没有修改生产代码、奖励、控制器或物理，也没有启动Isaac/PPO。上一轮已完成的Antigravity对话不能冒充本轮新审查。

## 三条真实入口

| 抓法 | 当前实际路径 | 尚未满足的条件 |
|---|---|---|
| Pinch | NativePinchSources与PolicyOwnedPinchGeometry，ROUTE_COUNT=5；原DualHandRodRuntime是双手各5个distal加table，共11mesh | verify_static_contract与NativeEpisodeOwner明确不向Pinch开放contact profile；需要自己的supported17 Reset/距离/历史/接触来源。不是把“13路”改名，也不能复制Tripod身份 |
| Tripod | NativeContactTripodSources，supported17、每手7条wrench路由，已完成一次真实8-control/1-update接线诊断 | 仍0手载荷，未抓住；接触版Close-only不含完整Lift/Hold资格，尚需有效接近信号和完整后继来源 |
| Power | 当前PowerSharedRodRuntime每手登记5 distal、base_link、finger52_link | verify_static_contract、source工厂与PrimedContactContextOwner均拒绝Power；已测开掌palm witness不能作为拳握。7条路由不代表所有指骨表面，缺失接触资格仍NA |

关键代码：tabletop_native_teacher_session.py:120–124、390–416；tabletop_native_teacher_actual_provider.py:76–99；tabletop_native_teacher_collector.py:114–126；tabletop_native_teacher_policy_pinch_distance.py:52–54；tabletop_power_shared_rod.py:22–44；tabletop_native_teacher_contact_context.py:20–24。完整SHA记录于readiness。

另外，verify_static_contract仍只接受smoke或Tripod contact-only，完整来源会返回full_shared_contact_slip_hold_source_adapter_missing。不能将一次PPO更新、低KL或CPU通过描述为三抓法完整Teacher。

## 距离势能的实际影响

生产smoke_potential位于tabletop_training_reward.py:106–111，TrainingReward._potential在770–775行调用；固定dref=5.804043651mm。对既有真实241行零指令与64行策略记录重放，全部305行distance potential均为0。原raw、S0和阈值未改。这只说明该距离项在这些状态被clamp，不说明PPO参数梯度全部为0，代价/价值/熵仍可产生梯度。

只作数值研究的候选是：d/dref<=1时逐值保留原公式；大于1时取-2*tanh(d/dref-1)，总范围[-2,2]，不移动dref。参考内5个采样值与原公式严格相同；零动作末端候选phi=-0.279837，旧策略末端-0.072969。SDK67策略相对零动作的候选差约0.00227670，这是同cohort数值差，尚无误差校准或抓取效果资格。

使用生产control_total进行三个纯代数案例：非终止bootstrap、三物理步后失败终止、time-limit的pre-reset bootstrap。贴现telescoping残差最大3.66e-15；按V_new=V_old-deltaPhi作一致变换后，GAE代数残差最大1.53e-16。负对照不变换value时差异显著，因此“改奖励但照用旧critic”不能宣称数值等价。失败终止样例总reward仍-39.920668，并非正奖励。

这些GAE比较是显式的synthetic algebra，不是调用真实PPO训练得到的结果；没有创建OnPolicyRollout或伪造策略token。原PPO/GAE另外运行实际CPU回归。代数一致不证明signed候选能更快学会抓取，也不授予设计或原生资格。

## 选择边界与验证

另一条路线是独立静稳task_start，但上一轮零动作0.5s末仍有12.626mm/s最大节点速度，末0.1s平均Z变化0.89188mm，尚不能建立合格静稳起点。当前不能偷偷平移Reset参考或在episode中途重设dref。

候选若进入实施，须独立配置、schema/来源SHA、checkpoint身份和实验目录，接通原Context/Source/Collector/worker并验证完整terminal/GAE合同；不能只换一个势能函数或让旧profile悄然改变。Pinch、Power各自缺失的实际来源也必须分别解决。保留原24D/371D、双RGB、资产/参数、30mm Lift/5s Hold/0.5mm Slip，R2/ForceProxy关闭，正式BC/DAgger/Student仍受门控。

本轮生产奖励/接触/session/preflight六文件383 passed（94.96s）；原结构化PPO与rollout admission两文件27 passed（6.22s），共8个不同文件410 passed、0 failed。上述3个代数案例不计入pytest数量。native_runs=0、ppo_updates=0。完整命令、日志及分析SHA见readiness；私有原始数据和分析脚本留服务器，保护文件及历史模型在发布前核验。

发布核查：Registry回归60 passed（14.37s），单独记录，不计入上述410项；528个保护文件和3个历史Checkpoint SHA均未改变。PR#2保持Draft/open/unmerged。
