# Contact Tripod fresh8：第一次真实有界 PPO 与零接触诊断

实际测试HEAD：`bbaacaad5cf960f0bee63c0b1d4993d6e435b653`。执行指纹d4ea1e10c76994c76532b3b6b7a6f6e5072be9d609a76030e0470e3b1d1d23e0，attempt-2eprrla7。经Windows发布、资源/单实例/未消耗检查及用户持续有限范围授权后，仅运行一次原配方，没有追加预算或强制结束。

## 实际执行和独立复核

2个SDK初始化步，1个独立零动作source prime，64个策略物理步，总67 SDK、65 solver calls、780内部子步；8个完整decision/settlement/transition、16个真实pre/bootstrap来源包、41个材料节点。Reset2/S0保留，prime到SDK3/rod12；第一个策略物理终点4，最后终点67。原Collector/Actor/PPO接受一次更新，一个minibatch，参数差L2=0.003945996284，checkpoint SHA `b2242925b780569fe4917662b9fcf06b0688136baaada9cb8b257893bafd5196`。这是真实仿真数据驱动的原PPO更新，Actor/PPO计算设备为CPU，物理/DeformX使用GPU3；不冒充纯GPU训练。

原Repository.verify_actual_diagnostic使用准确receipt SHA重新核验执行身份、源字节、实际7文件、同钟来源、模型输入、Reset、预算、Checkpoint/Adam/RNG及自然退出，返回diagnostic_verified=True。原scheduler receipt保持actual_receipt_audit_pending/native_qualification=False，不改写；Registry另记录有限诊断通过。回执SHA `a8c9b362e248188e5cc19d4c489eb0cc333ff4ef623d6470d13899054bb25edc`，独立readback SHA `55246ee28abbb3b898b62b9fdc027847b3bded19fd58e60b11c725d753bbd8d1`。

## 抓取没有成功：真实接触仍为零

原contact reducer离线重放这次64行原生physics（不是第二次采集）：左右手各0/64 loaded，法向载荷峰值均0N；双手承载接触0/64，support-free0/64，table接触0/64。支架0/1峰值分别0.094233555N与0.093937434N；杆最大节点速度0.071833082m/s。线缆由支架承托，不能称为抓住或Lift。

8个奖励窗口的phi_pre/phi_post_raw/endpoint_phi全部0，只产生时间/动作代价；contact、geometry、persistence没有正贡献。原固定Reset距离基准0.005804043651m，实际endpoint距离0.005814701784～0.006015894236m，均大于基准，原distance潜能因此被其设计clamp到0。这不等于distance项没接入；本轮不移动基准、不放宽成功标准、不加手工Approach、不修改物理。下一步针对动作→实际目标→关节反馈→各指腹材料距离的可达性和有界学习方案取证。

## PPO与关闭

原LR1e-4第一次full-rollout KL=0.135870636超过0.02，原KL Guard回滚并以LR1e-5接受，最终exact KL=0.001356620，未调大KL门槛。预裁剪梯度范数68.246826，Entropy=-21.681309，重算logprob最大差1.90734863e-06。完整24D分组std和minibatch/epoch原日志保留私有目录。

外层192.874823s<240s，inner before/after剩余38.047113s/2.703643s；两种时限分别核验，没有把外层192秒误判inner180超限。外层和GDB inferior exit0，无signal、无kill、无retry。App.close返回，14个被跟踪weakref全死，GPU3进程已释放；内部general native_release证明仍false，不将一次自然退出夸大为全部生命周期已解决。

## 准入与后继

本次仅0.1333秒策略运动，无法证明5秒Hold。teacher_admission、native/grasp qualification均false；Pinch、Power与正式BC/DAgger/Student继续各自门控。支架、固定双RGB、左右手、原24D/371D、资产、关节保护及30mm/5s/0.5mm保持原样；R2/ForceProxy关闭。下一步先用本次实际动作和几何账本定位无接触，再推进固定腕接触保持和原profile恢复，不能用重复同指纹运行制造样本。

本包没有追加Isaac运行，只有本次结果的独立离线复核和CPU回归。Antigravity仍额度受限，未记新复审通过。原始数据/日志/模型保留服务器，不上传。528保护文件和3历史Checkpoint在发布前重新核验。

## 完成后的CPU回归

真实配方参考审计、原contact账本审计及Registry联合回归67 passed、0 failed（41.82s）。这是CPU/离线回归，不是另一次原生采集或训练。

登记有限诊断结果后，再次独立核验原回执仍通过；最终Registry另60项通过（13.66s，与67项重叠，不累加）。
