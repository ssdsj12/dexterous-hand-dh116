# 真实目标与关节跟踪审计：零接触问题的动作链取证

基线HEAD：`f3679e7081f03e5f634760876997176f0996b129`。本包只修改原有审计器和对应测试；不启动Isaac、不做PPO、不修改控制器、Reset/S0、奖励、速度、资产、关节范围、固定双RGB或成功标准。

## 实际修改

在scripts/audit_dh116_teacher_contact_ledger.py的_joint_tracking中，从同一physical event的pre/post实际PhysX backend_targets、q/qd、joint_names与active_indices建立逐关节映射。SDK slot顺序不作为24D动作顺序；右finger12仍是canonical action7。保留主动/被动关节实际q/qd，排除被动关节的未命令target槽，防止把默认0误判为跟踪误差。腕部只在相同articulation joint坐标内比较q与SDK target，不把world palm请求直接减去local joint target。

_tracking_summary新增full_interval_coverage、missing_sdk_steps、coverage_scope、residual_scope。可选target渠道缺失时保留NA、保留全部contact账本；统计只针对有效行，禁止把部分覆盖描述为全段合格。有效来源跨帧换owner、左右alias、映射错误、bool/text/非有限target仍严格拒绝。_finite在转换float之前核验真实数值dtype。audit_file原SHA、连续clock、contact资格规则不变，新增tracking命名空间，不授予控制或抓取资格。

## TDD与CPU回归

新增测试先复现缺失字段、owner/alias逃逸及bool/text转换问题；记录首次红灯、夹具时钟修正后的有效红灯与最终绿灯。专用16项覆盖乱序关节、action7、被动target NA、腕部坐标与单位、缺失target、不完整覆盖、左右alias和跨帧身份变更。最终7文件联合回归134 passed、0 failed，111.45s；16项包含其中，不重复累加。测试期间两份源码SHA保持不变。完整命令、失败日志SHA及最终日志SHA见readiness。

## 旧原生数据重放结果：不是本包新增采集

严格以raw physics SHA 98770ef33be25d204589f9511a7baa2dc72e2cd2167118b58cc732e2de76b571重放既有Tripod Contact8，SDK4～67共64行，原始文件未改写。左右target渠道均64/64，missing=0，完整覆盖仅指这64行。左右所有受控角关节最大残差0.00402523/0.00492120rad，受控平移关节最大残差0.025370/0.024789mm。主动finger单独最大残差0.00153814rad；请求finger target与实际backend target最大差约2.94e-8rad。不能由这些条件残差单独授予控制性能资格，也不能据此证明力接触。

双手loaded、bilateral和support-free仍全部0/64，手部法向载荷峰值0N。两支架峰值0.09423356/0.09393743N，table0N。这说明“指令完全没发到SDK”的假设没有在本段证据中得到支持，抓取仍未形成。

同一0.133333s记录中，rod节点平均Z下降4.671094mm；实测腕平移只有每轴约0.04～0.22mm。左三指pad gap起末为2.411→3.413、1.605→2.392、1.851→1.682mm；右三指为6.678→6.909、9.879→9.971、12.417→11.728mm。gap同时受杆、腕、指运动影响，当前是相关观测，尚未隔离因果。单轴腕平移速度上限不能证明总pad可达性；距离势能被clamp到0也不代表PPO参数梯度为0。

## Antigravity实读与独立审查

本包已完成R1/R2真实桌面对话，核对双方仓库、HEAD及实读源码SHA。Codex否决“8步物理上必然不可达”和“PPO梯度全0”的过强结论，Antigravity在R2明确接受纠正；双方确认target缺失应保留NA而非丢掉contact证据，并加完整覆盖标记。R3真实复审完成，两份修改源码SHA与Ubuntu完全匹配，未发现审计包阻断项，建议关联回归通过后发布；对应134项最终回归已通过。R3同时确认真实参数→几何capture及独立无Actor入口。

原NativeEpisodeOwner将physical历史绑定真实策略decision，不能用假policy token运行无Actor对照。后继应复用build_fresh_supported_owned、原DualHandPhysicalEpisode、c123 Reset和既有R06生命周期；真实NativeParameterEndpoints须先于几何capture，prime必须持久化并更新rod boundary，所有已推进步骤计费。候选2setup+1prime+240测量=243SDK/241solver/2892substeps只属于准备方案，尚未冻结或执行，不复用已耗尽67步授权，也不假定240步足够静稳。

## 准入、保护与后继

本包native_runs=0、ppo_updates=0；Teacher/grasp/native qualification均false。Pinch和Power需要自己的接触来源，不能借Tripod证明；Power仍须掌面+五指包络。正式BC/DAgger/Student继续门控，R2/ForceProxy关闭。保留30mm Lift、5s Hold、0.5mm Slip。原始模型、数据、日志只留服务器；发布前检查528保护文件与3历史Checkpoint。下一工作包实现并测试真实零动作沉降入口，随后另冻结预算与墙钟，不先修改奖励或Reset基准。

R3建议的prime顺序仍需独立修正：沿原prime_supported_sources，必须先计费并持久化已推进的raw row，再调用可能拒绝的boundary.capture，最后更新episode._rod_packet；不能因边界校验异常丢失真实推进证据。外层480s/内层360s仅为后继候选估算，尚未冻结或执行。

登记后Registry另60项通过（14.61s，与134项重叠，不累加）。
