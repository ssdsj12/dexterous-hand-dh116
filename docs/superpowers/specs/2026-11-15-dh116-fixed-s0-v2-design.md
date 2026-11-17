# DH116 fixed S0 独立 v2 设计

状态：用户已审阅方案并明确选择“采用独立v2，按文档顺序推进”。本文是实现设计，不是原生运行许可或成功证明。文档基线为正式731a97298c2a1498f9fddd7a2ce7f35ce3dfa0b9；C04 own14已来自503d9262（正式code0d3653），随后C05经独立审查与86项root CPU回归，已正式集成5b7f13ca10c8bebf1ae4693b01c6cdd09391d533；Windows/GitHub manifest为3f277ed219248280b1640b11a4bab78c4080d981。此为CPU工程资格，C06真实provider与原生更新/恢复尚未通过。原DOCX SHA256=1ea4da5342de9a64e5adb3ef5c0a50482d2158bd9a9d3e7f5fab7f6b04e318dd。

## 不变量与验收层次

保持两只无机械臂DH116、同一自由柔性线、每手6主动+5从动、24D速度动作、480Hz物理/60Hz控制/12杆子步/8迭代、原材料/增益/力矩限制。严格成功仍为整线净升30mm、无外援且真实双侧支撑/离桌、连续悬空5s、材料与切向滑移不超过0.5mm、数值健康。CPU合同、真实训练链、长程Reset、严格物理成功、性能各自验收。所有未验证模块维持native_ready=false。

旧C01/C02/C03及approach_wiring_smoke_v1保留独立身份和字节；不重解释旧context50，不迁移旧checkpoint冒充同模型续训，不覆盖历史模型/R0账本/11-10/528保护源码与2历史checkpoint。W0–W2继续既有授权修正，不重建PPO算法。新v2独立文件/配置/模型/checkpoint命名空间。

## W3：执行阶段与context51

新执行owner的PhaseSampleV2只定义CPU数学/fixture输入；typed dataclass、state.source_scope、caller True均不能证明native来源。生产context/pack必须有W4同一真实owner实际source adapter签发并复核的PhaseSourceWitnessV2（同snapshot/phase receipt/epoch/clock/source SHA绑定），缺少时具体拒绝full_v2_phase_source_adapter_missing或full_v2_phase_source_witness_missing。新执行阶段顺序固定Close→PreliftHold→Lift→LiftHold，phase_source使用独立execution_phase_owner_v2而不伪装原task.step。Reset只恢复经解析验证的S0，不调用Approach轨迹。S0由tabletop_native_teacher_preflight.load_reset_manifest(config, project=...)验证original_receipt、reset_readback、numeric_reset和exact数值/mesh来源；照片、pose preview和中途帧均不构成S0。

Close建立required pads及原opposed判据。PreliftHold稳定计时从第一个合格端点零开始；prelift_stable_s=0.25是独立配置候选，须通过真实测量后才准入，并非既有成功硬门槛。稳定谓词必须来自实测接触、参考跟踪/相对运动条件；没有来源时拒绝full。接触丢失返回Close且预抬/最终保持计时清零；锚由W4持有，不随阶段回退而重建。Lift到达原高度/支撑/离桌/滑移健康条件后进入LiftHold。首次合格端点为0秒，2400个完整合格间隔及2401端点才可满足5s；中断回Lift，接触丢失回Close。PreliftHold绝不计入最终5s。

新context51为独立schema：0:4=[Close,PreliftHold,Lift,LiftHold]；4:50沿用旧context字段的名称、单位和归一化含义；index50专门为prelift_stable_duration，按新配置显式正scale归一化。新metadata包含phase contract/source receipt、profile、clock、epoch、fixed-S0 SHA、context schema digest。旧slot7仍是最终continuous_hold_duration；slot6仍双侧持续接触塑形历史，与index50不同。新packet不可通过旧C02 validate_context，也不向旧TaskContextOwner伪造receipt。

模型先仍六键：common371/common_valid、rod_nodes/rod_mask、task_context/task_context_valid；context宽度51，encoder输入102（值与mask）。common原371中305未知槽继续NaN/false，网络只取已验证66；rod41×13全部节点要求不变。独立模型/输入/context/phase身份纳入checkpoint；旧50D模型、旧动作空间模型或旧checkpoint全拒绝。W8的controller_state50D/74D是未来独立八键版本，与context51不混用。

## W4：同钟来源、slip生命周期和独立评价

当前实冲突：19d4593 tabletop_deformable_grasp_task.FlexibleGraspEpisode._slip:384–414失双侧接触时保留anchor但slip=None；tabletop_training_reward.TrainingReward._read_anchors:550–589一旦建锚便要求左右×material/tangential四组有效drift/uncertainty。独立v2必须显式表达never_anchored、detached、contacting，旧v1语义不变。

一个owned measurement snapshot绑定owner/episode/reset/rod/SDK/rod tick/实际时间/来源SHA；同时持有实际接触、支撑、节点、固定材料锚、原guard、禁止辅助与健康。unknown是缺/坏来源，抛SourceFailure且不形成PPO transition；proven NA是某量在已证明物理状态下无定义；真实无接触、回落、失稳/滑移超限是可学习负例，沿原任务定义判断是否终止。

首次合格双侧接触冻结原材料station/UID/tick、切向参考和不确定性。detached保留这些exact reference；接触滑移可为有源NA，理由fixed_anchor_detached；没有建锚用no_bilateral_contact_since_reset；contacting必须给全四组有效drift/uncertainty且不重锚。固定锚几何漂移是单独命名的量，不伪称接触滑移；如果现有实际geometry reader能明确定义并测得，它在detached仍有效。不存在来源/算法证明时保留明确unavailable并阻断依赖它的full profile，不能补零、就近换材料点或从奖励反推。

奖励与独立evaluator分别从同一snapshot重算自己的资格/时间/结果；evaluator不接收reward.success。v2奖励保持C03原数值权重、单位积分、势差、gamma/GAE/terminal Phi0/truncation/bootstrap、事件一次及失败优先。detached接触slip成本按新schema明确不适用，contact persistence/lift/hold资格失效按原公式归零，coverage/opposed按实际剩余接触更新、时间成本继续。slip超限不会自动新增一律硬中止。完整来源adapter/native资格属于独立核验门，CPU fixture不得提升为actual source。

## W5：真实长clock与Reset v2

旧tabletop_mesh_reset_order._clock:27–39明确绝对SDK≤240；C04 NativeEpisodeOwner.reset_clock_preflight先于mutation保留该拒绝。它不意味着第241连续步必失败。setup2+32×8=258说明32-window后Reset需要新reader；5s保持2400间隔，Close/Lift另计。

独立ClockPolicyV2有显式max_sdk_step和误差算法身份，按真实SDK dt与整数推进检查actual timestamp；保留SDK绝对步、episode相对间隔、rod_tick/solver_time、reset_epoch。误差包络从已知起点按真实float32/float64累积模型计算并绑定验证范围，绝不覆盖sim.current_time=step*dt。不明确的SDK计时ABI先只读核验源码/fixture/既有receipt，再实现独立policy；不能臆测SDK如何累积。安装源码已确认Python累加callback step_size且整数步递增，但pyi仅给Python float不证明底层精度。≤9600间隔double-dt累加数学最大误差2.434e-12小于旧1e-9，因此实际源确认此模式时保留旧连续控制reader；FP32-dt模式9600偏差约1.043e-6只作为需核验的候选，不据此先重写控制栈。literal240是Reset确定硬门。

Reset v2独立实现/显式注入clock reader，保存冻结v1源文件，不全局放宽v1。恢复S0 q/qd/腕targets/rod位置速度director、11mesh current/target/teleport、反力与composer、phase/contact/hold/anchor/history和参考控制状态（W8前该分支不存在），clear_meshes后借用view旧epoch失效。setup/reset/已隔离步数消费不可返还。CPU至少覆盖240、258、>2400实际推进后Reset且再执行下一episode；native采用相同边界但须独立批准预算与冻结worker，CPU测试不授权运行。

## 文件边界与交付

三个实现计划分别位于：
- docs/superpowers/plans/2026-11-15-dh116-fixed-s0-phase-v2-plan.md
- docs/superpowers/plans/2026-11-15-dh116-slip-source-v2-plan.md
- docs/superpowers/plans/2026-11-15-dh116-long-reset-v2-plan.md

新模块采用*_v2.py或新的phase_contract_v2/source名称；旧C04/C05工厂不改默认profile。v2软件接线需独立adapter合同，不能给existing NativeFactories传qualification bool绕过exact-type检查。阶段W3可以先做CPU/模型合同；W4补真实测量来源前full保持拒绝；W5可并行做CPU时钟/Reset，但native full取W3+W4+W5交集。root进行独立SPEC/QUALITY后按包集成。

## 后继集成包 V2-I01：新合同进入完整训练入口

规划注册项native.fixed_s0.v2.integration（V2-I01），状态planned；本次仅记录规划，不改正式registry/readiness。位置为W3+W4+W5之后、正式v2 update/save/resume验收之前，完成后再进W6/W7。依赖正式C05 checkpoint/training与C06实际worker/provider精确来源；合同型C06或fixture不替代真实provider。W3/W4/W5各自完成不代表工厂已可用，旧C04/C05 exact-type仍应拒绝新model/packet/epoch。

未来新增文件（非现存来源）：src/go2_pvcnn/control/dh116/下tabletop_native_teacher_session_v2.py、tabletop_native_teacher_collector_v2.py、tabletop_native_teacher_checkpoint_v2.py、tabletop_native_teacher_admission_v2.py，以及scripts/train_dh116_native_teacher_ppo_v2.py和tests/test_native_teacher_integration_v2.py。复用W3 model/context、W4 SourceAdapterV2/RewardEngineV2/EvaluatorV2、W5 FixedS0PhysicalEpisodeV2/ResetPreflightV2/guarded_exchange_factory_v2；原PPO算法、正式C05原子存储与验证helper、C06资源/生命周期机制复用，不放宽旧exact-type或改旧入口。

未来API在此固定，V2-I01实施时在上述对应新模块定义：

- NativeSessionConfigV2是frozen dataclass，字段phase_config/model_config/reward_config/source_config/reset_config为owned Mapping，resolved_reset_manifest/budget/authorization/resource_reservation/worker_binding为path+SHA绑定；backend_profile为真实BackendProfile。native_ready布尔不是来源。
- NativeEpisodeOwnerV2在session_v2.py定义，借用且只持有一个FixedS0PhysicalEpisodeV2/sim/rod，公开reset(seed,*,episode_uid,owner_resets)->Mapping、control(action)->Mapping、pre_action_state()->Mapping、close()->None；reset使用显式v2 reset/exchange factory与实际source/live验证，close只解除borrowed callbacks，原生释放仍由C06/R06负责。另持有同一history/phase/source/context/ledger与reset_epoch；不另建物理scene。
- NativeFactoriesV2是frozen dataclass，包含episode/model/trainer/reward/sources五个callable；episode(contract)返回上述NativeEpisodeOwnerV2，model(contract)返回NativeContextTeacherV2，trainer(model,ppo_config)返回原DeformablePPOTrainer，reward(config)返回RewardEngineV2，sources(owner,source_context)返回实际SourceAdapterV2。工厂由源锁resolver核精确源字节/返回类型，任意caller callable不授native资格。
- create_native_collector_v2(config:NativeSessionConfigV2,factories:NativeFactoriesV2)->NativeTeacherCollectorV2依次执行静态source/phase/model/reward/clock/S0/预算→单owner→实际v2 Reset→live/readback/numeric/guard→真实source adapter与初始phase witness→新model→原trainer（在此restore）→reward/evaluator→collector。无第二scene/额外step；部分构造清理复用已修C04/R06借用生命周期。
- NativeTeacherCollectorV2定义collect()->RewardPacketV2或None、update()->Mapping、restart_episode(factories)->NativeTeacherCollectorV2、close()->None。RewardPacketV2由W4定义，保留C03全部结算字段和新source schema。每decision一次原control，每physics一次snapshot/witness；history结算→reward→context51→pre-reset value→原record→context commit。token/issued/latent/logprob、隔离及budget消费保留。v2 context显式接受并源锁核验现C04 NativeAction24HistoryOwner reset-bound版本，不能只借同名接口；episode_uid/reset_id/rod_uid/clock贯通新Reset，不改time值。
- checkpoint_identity_v2(trainer,training_contract)->Mapping、save_checkpoint_v2(path,trainer,training_contract)->Mapping、load_checkpoint_v2(path,trainer,training_contract)->Mapping仅接受新model/input/context51/phase/slip/source/clock/reset/gamma/action身份，保存Adam/计数/RNG与clean accepted-update边界，旧C05 checkpoint双向拒绝。复用正式C05 single-bytes snapshot/安全反序列化/exclusive写入与验证helpers，不能patch旧_trainer后借旧namespace保存新模型。
- actual_admission_v2(config)->Mapping逐项输出静态/live/source/clock/checkpoint/provider/budget/reservation条件；execute_native_v2(config_path,output_path,*,checkpoint_path=None,additional_updates=1)->Mapping由新CLI调用。执行前核正式C06真实provider的完整entry callable、source SHA、实际lease类型、同owner SDK/rod资源、预算/锁/R06 teardown。worker_binding只能从已核C06注册入口解析，不能接受JSON自报provider或任意callable；无法解析时拒绝native_v2_actual_worker_provider_missing，预算/资源分别拒绝native_v2_budget_missing/native_v2_resource_reservation_missing。

V2-I01先RED证明旧factory/collector/checkpoint拒新身份；再以新入口通过CPU原SDK/rod doubles：32 transitions→accepted update1→clean save→独立新进程v2 Reset/live→load→additional_updates=1→accepted update2，检查finite非零权重变化、实际计数和same-byte origin。现存producer逐字段真实调用、伪造phase witness/foreign epoch/第三步source损坏、post-SDK clearance第三步终止无第四步、258/>2400 Reset后下一episode都是完成门。独立SPEC/QUALITY与528+2保护通过后只给software完成；native同样闭环仍需正式C06 provider、resolved S0/全source/clock范围及单独批准预算资源；grasp/三抓法/5s另验。无新入口、实际source接线或独立进程恢复证据时，规划项不得标complete。

## W6–W10依赖

W6 GPU exchange先证明同S0/动作/物理下逐step q/rod/contact/wrench/reward一致；保持上一帧反力延迟、COM扭矩坐标/单位、owned view/stream-event/reset生命周期。当前静态可见.cpu().numpy、host force/pair getter只是profile候选，未测排名或加速倍数；不得假定上游96b8514接口在当前独立backend ABI全部可用。

W7来源context/分块日志必须保留生命周期边界SHA和动态owner/epoch/clock；当前C04 NativeJournal每emit都fsync，旧journal仅边界fsync，须分别计时。有界ring、producer completion event、writer acknowledgement、背压及checkpoint flush通过对照后才启用。

W8在既有24D坐标/限幅内增加v_ref/q_ref/pose反馈；0.1–0.2s ramp只是候选，jerk再加a_ref。safe_z与限幅同可行域，冲突明确limit_conflict。需新controller50D或74D+valid八键模型、Reset和checkpoint，不修改本轮六键context51。

W9先迁单环境storage/GAE/model/reward/optimizer与历史接口，再N=1→2→4递增及subset Reset隔离；不只.to(cuda)。W10 MPC是可选专家初始化/参考，不是永久PPO前置；在线修正需三层动作和正确logprob。既定固定前方双目保留，腕部相机需单独传感器选择和新外参合同，不能悄悄替换。

## 设计完成标准

四份文档只构成可审查设计；实施前记录base SHA/独立worktree，按各计划TDD和相关CPU回归、保护528+2、独立SPEC/QUALITY。设计阶段无App/GPU/Isaac/原生/安装/下载；完成审查后按既有Ubuntu Commit→Windows导入同一Commit→GitHub发布流程提交。native budget/worker、真实S0/来源、长clock误差包络、三抓法和严格成功各自等待实际证据，不能用一个native_ready布尔值替代。
