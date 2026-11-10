# Privileged Teacher reward：Phase 0 基线审计

本文件记录 `base11-10` / `6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b` 的实际代码。审计范围是源码、已签入 launcher/config 和纯 CPU 算术测试；未运行 Isaac、训练、GPU、原生采集或 checkpoint 修改。**不能据此宣称任何 Teacher、Student、support/friction 或物理成功已通过 native qualification。** 后续实现需要保留此基线身份，再建立派生合同。

机器合同为 [privileged_teacher_reward_contract.json](contracts/privileged_teacher_reward_contract.json)，含源文件精确字节 SHA256、profile、系数、默认值、启用条件和成功条件。`active_run_profile=null`：本审计没有读取并绑定一个实际 active run 的命令、monitor、checkpoint 与物理 trace，因而不把 launcher 默认值写成运行事实。规范中 `teacher_r0_existing/r1_geometry/r2_support_recovery` 是后续要求；本基线没有这些 profile。

## 1. 调用链与默认值

以下路径均相对于仓库根目录；行号对应上述基线。

| 入口 | 源码事实 | 影响 |
|---|---|---|
| `scripts/train_dh116_tabletop_gpu_ppo.py:179` | imitation=`legacy`、reward=`baseline`；surface/free-wrist/vertical/phase-force/feedback 均为显式 flag | 直接 CLI 默认不构成 371D surface task |
| `scripts/smoke_dh116_surface_grasp.sh:5` | 显式 surface+free-wrist+free-grasp+phase-force+continuous vertical force，480 Hz；不传 reward profile、stable_hold_v2 或 feedback | 签入 smoke launcher 是 24D surface / baseline / legacy；不是 V4 全项，也不是 371D feedback |
| `scripts/run_dh116_surface_student.py:25`、`:154` | 默认 baseline/legacy；读取 smoke 命令，再显式转发选项 | 不执行本脚本以推断默认，使用 AST 和源码审计 |
| `scripts/train_dh116_surface_multistyle.py:35`、`:153` | 默认 baseline/legacy，非 baseline 显式向子进程转发 | stable_hold_v2 与 reward profile 是不同开关 |
| `tabletop_tensor_training.py:730` | 建 `TensorTask`；`grasp_shaping = args.grasp_shaping and not free_wrist_grasp` | surface 继承的 `--grasp-shaping` 字符串不代表 `GraspShaping` 真正启用 |
| `tabletop_tensor_training.py:752` | stable-lift 可选；CLI `train_dh116_tabletop_gpu_ppo.py:359` 明确与 free-wrist 互斥 | 当前 surface 无旧 `stable_lift_hold` |
| `tabletop_tensor_training.py:757`、`:767` | phase-force→free-grasp→free-wrist→surface 的安装顺序 | 后层 reward 可以覆写前层 term，不能把各文件出现的项全加起来 |
| `tabletop_tensor_training.py:771` | imitation=`stable_hold_v2` **或** vision_student 时 `enable_stable_hold_rewards` | baseline 也可能有旧 hold/slip，取决于此开关 |
| `tabletop_tensor_training.py:774` | profile 非 baseline 才安装 V4；vision-lift 非 R0 再安装实验 | `baseline` 不创建 `RewardAblation` |
| `tabletop_surface_exploration.py:51` | 非 baseline 要 feedback 且 stable_hold_v2（vision 路径例外）；support 还需 vertical+phase-force | 校验软件输入可用性，不等于力物理资格认证 |

`configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json` 的 12 N 是仿真候选 capacity；状态是 `candidate_requires_physics_validation`、`simulation_estimate`。surface `enable` 将 `f_requested/f_target/f_offset/f_boost` 清零，24D 没有 force action，不能把这个 load profile、固定 10 N shaping scale 或夹持法向 load 当成已测向上承载。

## 2. 从物理子步到返回 reward

依据 `tabletop_tensor_task.py:105,267,467,800,804` 和 `tabletop_tensor_env.py:331,382,438,485,554`：

1. `begin_step` 每个控制步将 `reward/terms/used/nonfinite` 清零，保存 `old_command`。continuous-motion 先把有限动作裁剪到 [-1,1]；NaN/Inf 对 live 行记一次 invalid-action -25。
2. `TensorEnv.step` 每个 physics 子步计算 `live=~task.done`，应用动作和 guard，读取真实 measurement。480 Hz / decimation=8 的签入配置对应 60 Hz 控制步；这里没有把每个子步再除 8。
3. `consume` 得 `valid=live & finite`；finite 检查基础 measurement、force/weight（启用时）、load 非负与 link/group/hand 和一致性、observation 等。原始异常数据不被改成“真实 0”；被排除的 scoring 行局部中性值仅用于安全调用 helper。
4. 每个项先产生 `[N]` **该子步 point contribution**，最终 `terms[name][valid] += value[valid]`，`reward[valid] += value[valid]`。rate 项已乘 `dt`，事件/差分项不能再次乘 `dt`。`used[live]+=dt`。
5. 物理终止当步仍可能有 time/force/slip/smoothness/downward 等成本和 success/failure 事件；大部分正向 surface rates 要 `failure==0`。V4 常规项使用 `valid & ~done`，最终成功样本仅 milestone 有 `valid & failure==0` 例外。
6. 第一次 terminal 后当前 tick 的剩余子步 `live=False`，不会重复计成功或 timeout。物理立即局部 reset；task snapshot 保留至 `end_step`，随后 task reset。必须沿用此 caller mask，直接错误地对 done 行传 `live=True` 会重复计算 failure/time，`consume` 自身不是幂等接口。
7. `end_step` 只为可选旧 GraspShaping 添加一次 force potential（surface 实际关闭）；然后返回累计值、各项累计值、used_time、done 和 terminal 状态。
8. `TensorEnv` 另加入 `velocity_excess` 到 reward，并在 extras.terms 加同一累计项；可选 acquisition 在 env 末尾加，但 surface CLI 排除此分支。

记 `r[e,k,i]` 为环境 e、控制步 k、子步 i 的各项贡献，则 `r_task[e,k] = Σ_i Σ_term r_term[e,k,i] + once_at_end`；`r_env = r_task + velocity_excess [+ legacy_acquisition]`。一控制步每个 term 的 total 是所有有效子步之和；跨环境总贡献是 `sum_e`，与训练 logger 的 `mean_e` 不同。训练 `tabletop_tensor_training.py:1276–1283,1325` 将每步 `reward.mean()` 和每项 `.mean()` 累加后再除 rollout steps，不能把日志均值当 episode return。此 Phase 0 没有实际运行贡献数值，任何 real per-env/total 均为 **NA**；测试只验证算术闭合。

## 3. 记号、来源与单位

下表每个公式均是**单个环境的最终 contribution**；跨环境或时间的 total 按第 2 节相加。`C(x)=clip(x,0,1)`，`x+=max(x,0)`，`I(P)` 是布尔 gate。输入 `bottom/reference/slip/drift/depth/point` 为 m，`velocity` 为 m/s，角度为 rad，load/support/weight 为 N，`dt` 为 s，最终 reward 为 points。rate 的 coefficient 单位是 points/s（若另乘 N，则 points/(N·s)）；势能差和事件系数为 points。

`h=bottom-reference_bottom`；`L_h=sum_digit effective_pad_loads`；`O_h=rejected_contact_loads`；`M=max(object_link_loads)`；`v=norm(object_velocity[:3])`。`paired_h` 由 native accepted pad normals 的对向面分配产生。`Q = paired_left & paired_right & O_total/(L_total+O_total)<=.05 & max(depth)<=.0008`；Q 是当前 measurement gate，正式成功还要求每个 raw physics frame 和完成控制窗。

基础来源：`tabletop_tensor_world.py:287–303,374` 的 Wire-filtered normal/friction 和真实物体/手腕位姿；`object_weight_n` 在 `:176` 为 runtime mass×(-gravity_z)。法向 magnitude load、signed normal Fz、normal+friction Fz 是三个不同量；前者不能替代后两者。FK pad reference points 是几何；不作为实际接触。此审计只追踪软件来源，不认证 PhysX 作用侧、friction partner attribution 或 per-digit 实测资格；这些物理量仍须保留 **NA / unqualified** 边界。

## 4. TensorTask 基础 reward 与实际覆写

来源 `tabletop_tensor_task.py:691–802`。下列基础项最终都受 `valid` 累加；额外 gate 单列。`p` 指 previous_phase，不是当前 phase；surface free-grasp 将内部 phase 固定为 3，仅供兼容。

| term | raw→normalized→coefficient，最终公式 | dt、phase/mask 与当前 surface 状态 |
|---|---|---|
| `contact` | `Δ[5*C(sum(hand_object_loads)/10N)]`；p=close 用全差，p=hold 仅负差 | 差分，无 dt；surface phase-force 覆写为 0 |
| `lift` | `I(p=3)*5*ΔC(h/.03)` | 差分，无 dt；free-wrist `add_feedback` 无条件覆写为 0，不能再与 dex/V4 lift 重加 |
| `contact_dwell` | `I(p<3)*ΔΦphase`；close `Φ=C(contact_time/.2)`，hold `1+2*C(contact_time/.3)`，lift=3 | 无 dt；surface 固定3、free-grasp也置零 |
| `lift_dwell` | `I(p=3)*2*ΔC(success_time/hold_seconds)` | 无 dt；surface hold_seconds=5，保留；保持中断可为负，整个稳定窗最高净+2 |
| `slip` | p=2 时 `-min(((slip-.003)+/.003)^2,25)*dt`；否则 bottom>ref+.005 时 `-min(((slip-.02)+/.02)^2,25)*dt`，否则0 | rate，当前 surface 仍存在，但 raw slip 见第8节，并非正式 anchor slip |
| `smoothness` | `-.05*min(mean(((command-old_command)/.1rad)^2),25)*dt` | old_command 固定为控制步开始时的 joint target；不是 measured/executed smoothness，也不是逐子步 target difference；surface保留 |
| `force` | `-.1*min(((M-15N)+)^2/(225N²),25)*dt` | 单link normal magnitude excess，非支撑；surface保留。M>30N还会失败 |
| `time` | `-.1*dt` | 含有限terminal样本；surface保留；timeout 是failure，不是再多一个超时reward |
| `success` | `25*I(event)` | 无dt；surface event来自正式tracker；原生caller只支付一次 |
| `failure` | `-25*I(failure!=0)` | 有限失败样本一笔；invalid-action在begin写-25、nonfinite在consume/env专门写-25，正常caller使它们不再进入同一行常规failed piece |
| `downward_force` | `-.5*(-signed_total_normal_Fz)+*dt` | force_scoring_started，或close-contact特例；surface free-grasp从开局started=True，保留全期向下成本，未设episode cap |
| `close_downward` | close→lift且当前Fz<0时 -3 | 无dt；free-grasp显式置0；即便launcher有flag，surface不支付 |
| `approach/mpc_tracking/stage/grasp_ready` | 注册初值0；上述surface路径不赋正值，free-grasp清stage/grasp_ready | 不得因字典存在就声称已启用 |
| `force_tracking/contact_region` | surface跳过旧force-region reward；free-wrist将tracking置0；contact_region保持注册0 | 当前不计，旧分支见第7节 |

`downward_force` 的 positive contact 对应项在 surface 路径**不启用**：`if contact_reward_window is not None and not surface_force_sharing` 跳过 window/continuous upward credit。`contact_reward_clock/contact_reward_paid` 留在 observation 不表示仍有 +0.5Fz 奖励。

## 5. 当前 surface rates 与 phase-force

来源 `tabletop_surface_exploration.py:165–253`、`tabletop_free_wrist_grasp.py:169–185`。所有 rate 乘物理 `dt`，gate 为 `valid & failure==0`。surface没有额外阶段门；current paired 来自实际pad数据，不使用指定thumb/index。

| term | raw、归一化与系数（表中为 points/s，再乘dt） | 附加有效条件/来源 |
|---|---|---|
| `dex_reach` | `3*mean_hand(top2_digit(exp(-d/.008))) + mean_hand,digit(exp(-d/.008))`；`d=abs(radial_norm-.008)+max(abs(along_axis)-.25,0)` | FK reference points相对真实物体中心和旋转后Y轴；纯几何 proximity，最大4/s，不是接触或成功 |
| `dex_contact` | `2*mean_digit,hand(I(load>.05 & effective_valid)) + 2*mean_hand(paired)` | 最大4/s；单手可获部分shaping，不能用于双手成功 |
| `dex_lift` | `60*C((h-.0005)/.03)*gate` | legacy gate=两手paired；stable_hold模式gate=Q。绝对高度rate会持续给分；V4含LIFT才置0 |
| `dex_force/dex_balance` | 0 | surface明确不追固定force请求；face/within balance另记 |
| `dex_action_rate` | `-.5*mean(clip((actions-previous_actions)/2,-1,1)^2)` | 使用提交给task的裁剪后normalized action，previous_actions为上控制步；非真实执行量；一个控制步内每子步乘dt |
| `surface_between_faces` | `-2*mean_hand(abs(face0_load-face1_load)/total_face_load)` | 仅paired行有ratio；缺接触的0不能解释成已平衡 |
| `surface_within_faces` | `-2*mean_hand(sum_digit(abs(desired-load)*assigned)/total_face_load)` | desired是每面对半、面内均分，源于当前实测load；非新force target |
| `surface_missing_opposition` | `-.5*mean_hand(I(accepted_load>.05 & ~paired))` | 当手有load但无对向面 |
| `ordinary_collision` | `-mean_hand(O_h/(L_h+O_h).clamp_min(1e-12))` | 非接受pad的normal magnitude比例；不是表面反力 |
| `surface_hold_progress` | `120*(clip(native_formal_duration,0,5)/5)^2 * Q` | 仅stable_hold_v2或vision；用上个已完成60Hz正式30mm窗口duration，在每physics步付rate；**V4不禁此项** |
| 旧 `surface_slip_window` | `-160*min((measured_drift/.0005)^2,4)*Q*I(h>=.003)` | 仅stable_hold；独立3mm tracking的last-completed-window drift；每physics步乘dt，原生window无效时measured_drift=0。V4 SLIP会同名替换 |

分配公式 `tabletop_surface_force_sharing.py:12`：bearing=`load>.05 & valid`，有效normal非零，挑最对向pair（dot<=-.25），每个bearing都必须可分到两侧（对axis投影绝对值>.2），两侧均非空。balanced 的1.5比值是额外诊断/软奖励目标，**不是当前正式成功门槛**。

phase-force 来源 `tabletop_phase_force_reward.py:34–96` 与 `TensorTask.consume:770–794`：

* 对每只手，`n_h = verified_support(...,legacy=True).normal`，`f_h = ...support`；`W=runtime weight>0`，`a_n=clip(2*max(n_h,0)/W,0,1)`，`a_f=clip(2*max(f_h,0)/W,0,1)`。
* surface hand mask=`paired_h & effective_load_h>.05`，没有旧2..12N限制（unrestricted_load=True）。normal总和>=0才 `grasp_upward_force=2*mean_h(a_n*mask_h)*dt`；support总和>=0才 `lift_upward_force=5*mean_h(a_f*mask_h)*dt`。
* free_grasp `phase_free=True`：两项均全回合开启，不是close只normal、lift只friction。单手至多拿一半rate。这是已存在组合，不应在Teacher再加一遍。
* legacy support算法：`normal_h=min(-sum_digit(pad_vector_z), whole_hand_normal_Fz)`；`friction_h=whole_hand_lift_Fz-whole_hand_normal_Fz`，仅该手 rejected_load<=1e-6 时接受friction；mixed-contact路径legacy直接不加friction。V4 support使用更保守的非legacy版本（第6节）。

## 6. Surface V4 profiles 与八个因子

来源 `tabletop_surface_reward_ablation.py:6–159,222–268`。这是现有代码的常量及其实际乘法，不是抄用规范建议。`baseline`不安装V4；`v3_reference`安装空terms observer但保留旧baseline+stable_hold组合。

| profile | 实际新term（加在已有任务组合之上） |
|---|---|
| `v3_lift_only/early_hold_only/milestone_only/contact_break_only/slip_only/relative_motion_only/support_only/smoothness_only` | 名称分别带统一前缀 `v3_`；每个仅启用下面一个对应因子 |
| `lift_progress` | LIFT |
| `early_hold` | LIFT,HOLD,MILESTONE |
| `contact_break` | LIFT,HOLD,MILESTONE,BREAK |
| `slip_relative` | LIFT,HOLD,MILESTONE,BREAK,SLIP,MOTION |
| `support_deficit` | LIFT,HOLD,MILESTONE,BREAK,SLIP,MOTION,SUPPORT |

**没有一个 cumulative profile 包含 SMOOTH；没有“全部八项”profile。** 精确15个profile字符串及enabled terms列在JSON机器合同；旧version名 `v3_*` 不改变返回合同version是`surface_stable_hold_reward_v4`。

`V=valid & ~done`（且bottom/reference有限）；`qualified=Q & V`；`airborne=h>=.003`；`early=V & window_completed & window_native_valid & early_window_good`。early observer用 **3mm** 起始高度、raw成功算法，独立于正式30mm tracker；window周期 T=1/60s。

| 因子 / term | raw→normalized→coefficient，最终每环境贡献 | dt、mask、episode cap |
|---|---|---|
| LIFT `surface_lift_progress` | `ratio=C((h-.0005)/.03)`；`best_new=qualified ? max(best,ratio) : best`；`8*(best_new-best)` | points差分，不乘dt；episode high-water，最多8；30mm仅7.8667，30.5mm满8；非qualified不更新best |
| HOLD `surface_early_hold_rate` | `I(early)*min(max(10-paid,0),2*T)` | 2 points/s、只在完成窗支付T，不乘physics dt；最多10points/episode |
| MILESTONE `surface_hold_milestone` | `sum_j [1,2,2,3,4,5]_j*I(duration>=[.1,.3,.5,1,2,5]_j & ~paid_j & good_window)` | 每级一次、无dt；总17；duration是3mm early window，不是正式成功；terminal success允许付最后5points，failure不允许 |
| BREAK `surface_contact_break` | armed后连续两个 `native & ~early`完成窗时 -4 | 无dt；hold>=.1s rearm；最多2笔即-8。无效native窗重置bad计数，不是物理break事件 |
| SLIP `surface_slip_window` | `-24*min((reward_drift/.0005)^2,4)*T` | 仅V、airborne、完成窗、native、slip_observable、finite nonnegative drift；最大绝对rate96/s；滑到破窗仍可罚，断接触则anchor清掉 |
| MOTION `surface_relative_motion` | `-2*min((relative_tangent_speed/.002)^2,4)*dt` | qualified、airborne、finite；最高8points/s成本；真实obj/pad body velocity在实际pad point计算切向速度，按valid load加权，不是centroid移动 |
| SUPPORT `surface_support_deficit` | `deficit=clip((W-S)/W,0,1)`；`-1.5*deficit*dt` | qualified、airborne、finite support、finite W>0；负support饱和为全部不足。无资格support=NA不能作0传入或claim此项物理有效 |
| SMOOTH `surface_executed_smoothness` | `-1*smoothness*dt` | V、finite；full 24D measured position增量与投影后的真实target增量，见下段；不替代旧smoothness/dex_action_rate |

MOTION：`v_obj_at_pad = v_obj + omega_obj × (pad-center_obj)`；`v_hand_at_pad = v_body_COM + omega_body × (pad-body_COM)`；切掉沿native pad normal的分量；以valid effective_pad_loads加权平均其norm。需要实际digit body速度，不可把整手力或速度广播成每指测量。

SUPPORT（`support_components(...legacy=False)`）：先仅保留accepted pad的normal z；`n_h=min(-sum z, whole_normal_Fz_h)`；`f_h=whole_lift_Fz_h-whole_normal_Fz_h`；若该手无rejected且有accepted接触，加完整friction，否则仅加`min(f_h,0)`；再取`support_h=min(n_h+friction_credit_h,whole_lift_Fz_h)`；`S=sum_h(support_h)`。代码有符号和whole-hand上界，但资格仍取决于native读数与partner attribution，不能从函数名字推断已qualified。

SMOOTH（`:162–186`）：rate向量=[.3rad/s×12,.01m/s×6,.5rad/s×6]；`measured=(actual_position_now-previous)/(rate*dt)`；`applied=accepted_target_delta/(rate*dt)`；`smoothness = I(ready&valid)*mean(clip((measured-previous_measured)^2,max=1)) + .1*mean(clip((applied-nominal_action)^2,max=1))`。first/reset样本不付motion差，但仍有projection residual。实际position为q+真实wrist xyz+真实wrist joint angles；target为最终joint command+world palms+wrist angle target，非actor request。每component cap1，整体最多1.1；旧smoothness和dex_action_rate依然独立计，因此不能将三个名称混成同一个执行平滑惩罚。

替换表只有：LIFT出现→禁 `dex_lift` 和旧 `lift_milestone`；SLIP出现→把旧同名 `surface_slip_window` rate清零、V4 piece覆盖。**未禁 `surface_hold_progress/lift_dwell/grasp_upward_force/lift_upward_force/旧smoothness/dex_action_rate`。** 当前371D free-wrist路径旧milestone/force_progress本来没装；不能因此宣称真的发生旧milestone双计。V4 early hold+milestone 与旧hold/dwell的语义重叠确实存在，新增Teacher必须透传最终env reward一次，不能再复制这张表相加。

## 7. 其它 opt-in / 旧分支：不能冒充当前371D活跃项

* 非surface `window_credit`（`tabletop_contact_window.py:18`）：`5*C(force/10N)*min(dt,max(1-clock,0))`，eligible且started才付，episode<=5；start=`first_contact`达到.02N启动，或`close_start`开局启动，不重启。continuous模式覆写为`.5*clip(normal_Fz,0,10N)*dt`，无window或episode cap，开始时机受force_scoring_started/force_reward_start控制；一次grasp_ready +3。surface跳过整块。
* 非surface force-region（TensorTask`:736`）：`contact=.5*clip(sum(weighted_normal_Fz_h),0,20N)*dt`，contact_weights为below2/above.5/invalid0，loads>12N手不正奖；`force_tracking=5*mean(exp(-abs(loads-requested)/2N))*dt`，需两手below、2..12N、sum normal>=0；`contact_region=-.5*clip(sum(invalid_region_loads),0,20N)*dt`。不是surface的contact_reward。
* 旧 `GraspShaping`（`tabletop_grasp_shaping.py:26`）：`force_ratio←EMA(force/W,tau=.05s)`；`Phi=2*tanh(force_ratio)`；每control transition `force_progress=.995*(done?0:Phi_after)-Phi_before`，只begin时live，end只一次。lift_milestone是absolute bottom>=.303/.310m、stable+lifting+eligible持续.1s时各+2/+3、episode各一次，无dt。**其absolute .30m不是raised support baseline；当前surface构造禁用，不能重新装入当前任务。**
* 旧 `StableLiftReward`（`tabletop_stable_lift_reward.py:16`）：good=eligible、lifting、bilateral_load_ready、v<.03、raw slip<=.002、signed vertical>=0、finite；`5*Δbest_height/.03`，best基于`clip(bottom-.30,0,.03)`；hold需bottom>=.303，付`4*min(dt*height/.03,.5-credit_s)`，最多2points。替换旧lift，和当前surface互斥。
* 旧 `grasp_acquisition`（`tabletop_grasp_acquisition.py:55,88`）：finite-cylinder真实pad relative SDF，`Phi=10*mean_pad(exp(-abs(sdf)/.015))+10*min_hand(mean_pad(proximity))+20*clip(min_hand_load,0,1)`；env末尾加`.995*(done?0:Phi_after)-Phi_before`；force-region另换potential。当前surface free-wrist CLI禁止此选项。
* 额外vision-lift实验（`tabletop_lift_reward_experiment.py:18–249`）：R0空，R1=height_hold，R2=upward，R3=drop，R4=hold+up，R5三项；只许`support_deficit` base且显式非R0。**这是已有vision实验R0/R1/R2命名，不是规范新增Teacher profile。**
* 实验quality `q=clip(1-speed/.03,0,1)`，须完成native窗、真实reset h>=.003、两手opposed、drift<=.0005且observable、support>=W、finite、无failure、good_window。`height_hold=1*C((h-.003)/(.03-.003))*q*T`；`upward=.5*C(min(h-prev_h,h-qualified_peak)/(.005*T))*q*T`（负值clip0），已支付高水位不能来回循环刷分。
* 实验drop是独立live物理子步ledger：同一连续下降段anchor取高水位，超过.0005m deadband的新下降距离`d`付`-.5*C(d/(.005*dt))*dt`；包含有效terminal failure样本，invalid-live打断桥接，inactive frozen，不要求有contact；完成窗update产生的drop被子步ledger结果覆盖，避免同段双扣。这个实验同样不认证force source。
* env `velocity_excess`（`tabletop_tensor_env.py:248,485`）：`-.1*min(((max_all_joint_abs_qd-3rad/s)+/3rad/s)^2,25)*dt`，scoring_live且finite；可在terminal当步付。finite速度超限为reward成本，不单独构成此处成功门槛。

## 8. 正式成功与不同 slip 的逐层对照

正式实现链为 `surface_exploration.enable:145–150` → `TensorTask.consume:654–680` → `SurfaceLiftSuccess.update:101–196` → `hold_window_step:5–20`。`surface_success_hold_s=5`，正式tracker默认minimum_lift=.03；返回事件是duration首次跨5s的pulse。

| 条件 | 正式代码真实条件 | 规范及其它指标的关系 |
|---|---|---|
| 净lift | **每个raw子步** bottom>=`world.lift_reference_bottom_m + .03` | reference来自原支撑baseline，含env origin；不是palm高度、COM高度或absolute .33m |
| 持续hold | 完成1/60s窗才更新duration；连续>=5s；480Hz时300个完整窗口/2400个raw样本 | .1/.3s early milestones和旧Teacher .3s不能代替 |
| 速度 | 每raw子步真实object linear norm **<.03m/s** | 未附加angular速度门；不能私加或放宽 |
| depth/数值 | 所有raw测量finite、quaternion非零、loads/ordinary/depth非负；每raw最大accepted depth<=.0008m，active有效 | 还受env独立机械/force/table/nonfinite guard，不是只看均值 |
| 双手对向contact | 对load/vector做真实dt加权控制窗mean，再以每手native opposed faces判定paired | 不要求每raw单独双面，1.5力均衡/force precision是诊断而非成功gate |
| ordinary contact | mean ordinary/(mean ordinary + mean accepted total)<=.05（数值容差1e-12） | 不等于“桌面支撑比”；检测本身不证明物体已离所有支架 |
| 正式slip | rod本体三点 `[0,-.2,0]`,`[0,.2,0]`,`[.008,0,0]` 转到**每只实际wrist/body link**坐标，取全部raw样本、两手、三点到固定hold anchor的最大欧氏偏移<=.0005m | 是完整三维位姿漂移，包含旋转效应，不是contact centroid、relative center、窗口bbox或每步差 |
| anchor reset | 首合格窗起始raw sample设anchor；slip>.5mm整窗duration归0，并用最后raw pose为未来窗新anchor；不合格物理窗同样打断 | 稳定后可重新开始完整5s，不保留破窗前duration；不能靠每步移动接触中心“更新正式锚” |
| signed support/weight | **SurfaceLiftSuccess没有Fz或weight参数** | TensorTask`:646`先算的`stable`含force>=0，但surface分支完全不用stable；不能声称support>=W、signedFz>=0是独立正式成功gate。支撑物理证据仍须额外验收 |
| 无恢复/完整首回合 | 不在SurfaceLiftSuccess函数输入内；见demonstration接受层 | 回合内恢复或后续回合成功不能标作无恢复首回合成功 |

原始 `TensorTask.slip = norm(object_relative - captured_grip)`（`:566–574`）是相对两腕均值的物体center偏移；grip按旧phase建立/双边contact loss重捕获。当前success分支不使用它作正式slip门。旧`surface_slip_tracking`是minimum_lift=.003m的独立tracker，last completed native窗口的`measured_drift_m`；V4 `surface_early_hold.reward_drift_m`又有独立reward anchor，允许在speed超限导致hold不合格时仍测量slip，超过.5mm后重锚、断contact清锚。它们与30mm正式hold anchor的起始时刻、有效mask均可能不同，数值不可互换。

`tabletop_surface_demonstrations.py:62–103` 的接受层累计 `tainted |= extra.recovered`，只接受 `first=done & ~ended` 且 `success & failure==0 & ~tainted`；报告 `optimization_performed=False, trained_policy=False`，`success_source=ordinary_TensorEnv_real_native_success;no_script_target_override`。`load_teacher_actions`的metadata qualified_lift/action_size检查还不是本次新Teacher的证明，须复验source SHA、动作语义、物理配置、原生完整轨迹与正式成功窗。

尤其 `tabletop_surface_teacher.py:33–45` 旧合同是 **26D / 261D / minimum_height=.01m / hold=.3s**，`original_formal_task_success`字符串还写30mm/.5s。`tabletop_surface_teacher_control.py` 是脚本servo/controller，不是Privileged PPO policy；旧support servo也不是这个PPO reward。它们的旧资格/文案不能覆盖本任务24D/371D、30mm/5s/.5mm标准。本审计按实际surface成功代码解释规范“支撑、速度、有效性、无恢复”概述；未降低任何原生阈值，也未捏造代码没有执行的force gate。

## 9. Reward hacking 风险与 Phase 1 绑定

| 风险 | 已有约束与剩余缺口 |
|---|---|
| 只抬掌不抬物体 | lift只用真实bottom、基准ref；reach和动作项可能仍变，但不是lift。CPU反例固定物体world center只移动wrist时lift=0 |
| 桌面/支架仍承托 | initial support baseline高度本身h=0不给lift；正式还需净30mm。但Q只验证手接触、ordinary hand load比例与depth；不能证明全部端支架反力为0。需native支架/partner证据，当前NA |
| 增大夹持力当向上承载 | dex_contact可正奖但最多rate，phase-force使用signed Z及runtime W；native作用侧/摩擦资格未认证仍禁止新R2。normal magnitude不能补写Fz |
| 单手获分 | contact/phase-force可部分shaping；Q、正式窗口必须双手paired，部分reward不等于成功 |
| 旧新lift/hold重复 | free-wrist先清旧lift；V4 LIFT清dex_lift，SLIP同名替换。旧hold/dwell与V4hold/milestone仍同计；surface旧GraspShaping本来关闭。新R0只透传final env.reward一次 |
| terminal/timeout/reset刷分 | live=~done、事件pulse和per-episode paid防同episode重付；reset重置budget后新episode仍可再赚early奖励，不能用均值高reward验收完整hold。首回合、无恢复验收独立 |
| 超大request裁剪 | continuous-motion有限超界先clip，因此invalid_action不会拒绝此类输入；测试/日志要保留raw actor request与实际clip/guard执行差，不能由bounded action推断raw policy合规或native成功 |
| requested当executed | 三个平滑项不同，只有V4 executed读取实际position+applied target；R0禁止把raw request变化伪作实际motion |
| force NA或广播 | helper要求N×2 hand force、N weight、N signed sum；不接受scalar广播。任何未知真实support/friction保持NA，不用0、不把whole hand平摊per-digit、不用固定10N代runtime重量 |
| centroid或诊断drift掩盖slip | 正式锚是物体三点在真实两腕坐标，独立于contact centroid；reward tracker必须带独立名称/anchor来源，不能混写正式slip |

Phase 1 建议：`teacher_r0_existing`绑定准确version+profile+全部父开关和source SHA，只返回原生`TensorEnv.reward`，附带现有extras.terms用于闭合审计，不重建计奖。R1在目标几何资格通过前关闭；R2在force作用侧、单位、坐标系、partner attribution与per-digit native来源全部通过前关闭；不因本CPU测试通过就开放。支持缺失、文案不一致和hold重叠需要显式报告，不通过改成功标准或默默禁用旧项解决。Teacher PPO、脚本Teacher/MPC、Student与DAgger成功必须各归其主。

## 10. 本次验证与边界

`tests/test_privileged_teacher_reward_audit.py` 使用现有CPU Python、torch、pytest，实际构造 `TensorTask.__init__` 并经过begin/consume/end与纯Torch helpers；synthetic测量仅用于计奖/契约测试，没有把fixture当native。

初次运行因旧fixture间接依赖未安装的gymnasium而collection失败；已将新测试改为独立SimpleNamespace模板，不mock或安装gym、不改生产源码。随后先验证合同缺失导致两个预期FAIL，再创建JSON；执行覆盖source/contract hash、CLI AST默认、精确profile/系数、8physics子步求和、done/timeout不重付、正负force独立成功分支、各8因子mask/dt、raw/early/formal不同锚、单手部分shaping、force shape/缺失、support不per-digit广播及executed residual。

最终命令（PowerShell，关闭GPU与bytecode，不安装包）：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest tests/test_privileged_teacher_reward_audit.py tests/test_surface_success.py tests/test_surface_reward_ablation.py -q -p no:cacheprovider
```

实际结果：**75 passed，exit 0**（新审计测试26例，既有success/profile测试49例）。现有旧surface integration tests还间接需要gymnasium/scipy；本任务未安装依赖、未导入Isaac，没有将它们标作已通过。
