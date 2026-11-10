# Phase 1 Privileged Teacher 实现与验证记录

当前状态：Phase 0 已由主任务封存为 `497e290`；PPO/模型/checkpoint与默认R0软件路径已实现。snapshot02又在独立 `tabletop_privileged_shaping.py` 补全规范4.3的R1四类和R2五类纯算术因子，并保留原support deficit一次。**Phase 1 尚未整体完成：原生measurement factory、逐项实际source/tick绑定、独立geometry资格和正式native验收仍未完成。** 本文没有宣称 native 训练、完整原生成功或 force qualification，具体代码与剩余边界见文末。

## 设计与边界

保留 Phase 0 审计的371D state、24D action和原生reward/success源码。本阶段新增五个 `tabletop_privileged_*.py` 模块，纯PyTorch，不导入Isaac、RSL或TensorDict。root负责独立CLI及原生env适配；模块不接管旧surface controller。

`PrivilegedTeacher` 的actor/critic分别为371→256→256→24/1，独立参数，ELU激活；可训练24D log_std从log(.1)开始，优化后投影std到[.001,.3]并仅清理改变坐标的Adam moments。动作是tanh(z)，始终保存原始z，复用既有 `tabletop_squashed_gaussian` 的稳定log-Jacobian、likelihood、MC entropy、latent KL。

模型接口：`latent_mean(actor_obs)`、`deterministic(actor_obs)`、`value(critic_obs)`、`sample(actor_obs)`、`evaluate_latents(actor_obs,critic_obs,z)`。generic PPO仅要求model有latent_mean、log_std、value；actor_obs可以是Tensor或dict[str,Tensor]（允许batched history），critic始终独立371D Tensor。Student bridge以后只需把dict重建成自己的history结构，不能把critic输入送进actor。

## PPO 生命周期

`PPOConfig` 起点：gamma=.995、lam=.95、clip=.2、lr=3e-4、epochs=5、minibatches=8、entropy_coef=.005、value_coef=.5、max_grad_norm=.5；这些是起始配置，未经native实验不能称最优。

```python
trainer = PPOTrainer(model, config, rollout_steps=32, num_envs=n, device='cpu')
sample = trainer.collect_step(actor_obs, critic_obs371)
# 仅把sample['actions']作为归属于当前policy的normalized issued24提交env.step。
trainer.record_step(reward, done, next_value,
    executed_actions=sample['actions'], action_owner='policy')
stats = trainer.update()  # 只接受完整rollout且没有pending transition。
```

collect生成collection token/session及policy_version并内部保存detached clone；返回sample的调用方改动不改变pending。record不能没有pending，不能插入MPC/Teacher外部action，不能把底层SDK projection替换成PPO action。存储cpu FP32与独立actor/critic输入；保留z/action/logp/old_mu/old_std/value/reward/done/next_value/bootstrap mask。检查actions与tanh(z)、冻结policy的old likelihood/mean/std；任何收集期间的policy修改在update前报错。

dictionary history的bool mask和integer frame id保持原dtype，只有floating tensors转FP32。无效depth/point的NA须由上层按真实mask中性化后再打包，PPO不猜测mask含义、不读取模拟器补值。sample/action相等检查在原collection device上重算tanh，避免跨CPU/GPU kernel舍入被误判。

done与bootstrap分开：默认`bootstrap_mask=~done`；真实success/failure terminal禁止bootstrap。截断必须明确`truncated=True`且显式bootstrap_mask，next_value来自terminal observation而非自动reset observation。GAE时间递推在done处停止，避免跨reset桥接；advantage与return detached，全rollout归一化使用population std，singleton有效。minibatch使用tensor_split/randperm覆盖全部rows，batch小于8时减少minibatches。所有优化后投影noise；统计记录实际samples、optimized_samples和minibatch_updates。

Actor BC使用独立optimizer，仅更新actor；critic和PPO Adam不因BC初始化而改变。BC只允许PPO clean boundary（pending空、rollout空）。专家成功池还须独立dataset资格验证，schema-only CPU arithmetic调用永远不构成dataset/native资格。

## Reward、dataset 与checkpoint

`compose_native_reward(native_reward,native_terms,profile='teacher_r0_existing',...)`返回reward与贡献trace。R0逐值透传最终native env reward一次，验证extras term总和与reward闭合，不复制旧term重新计奖。R1/R2须显式profile与qualified provenance；缺少几何或force资格、NA、shape错误等必须fail closed，不补0或固定10N。每个新增项记录raw、normalized、coefficient、dt、mask、source、per-env贡献和total。

R0 raw/normalized保存native已经缩放的contribution，coefficient=1，明确`raw_kind=native_contribution_not_original_physical_measurement`；`physical_raw=None`附unavailable_reason与Phase0公式审计引用，不伪造原始物理measurement。R1/R2还要求`expected_identity`匹配当前state/action SHA、qualification report SHA、native qualification run exit0/sample_count>0及原始native evidence文件SHA。布尔qualified字段本身不足以启用。

初始接口只有qualified geometry potential差分与signed support deficit（-1.5points/s×clip((W-S)/W,0,1)×physical_dt）。snapshot02补全下文所列独立typed纯算术因子；generic势能接口与typed因子只能选一个owner，support deficit仍只计一次，已有native同owner则报错。所有新增系数均为默认关闭的候选值，没有native调参或效果证据。R1/R2仍enabled=false，不自动启用或迁移成功标准。

专家dataset只接受完整首回合、未恢复、未从checkpoint恢复的native成功，带episode/source/物理配置/成功proof SHA和当前canonical action/state身份；`episode.source_path`必须实际文件且SHA匹配，不接受visual/26D/261D旧合同。成功proof须绑定同一episode与身份，数值有限的30mm/5s/.5mm及native_task_success标志，并有native_isaac_physx来源声明。schema-only fixtures始终不能进入成功池。此结构校验不是本次已经完成物理资格的声明，native collector仍须对真实trace负责。BC group loss对joint12/translation6/rotation6各先按有效维归一化，再等权，避免维数多的joint组支配loss。

checkpoint API：

```python
identity = build_identity(config, state_contract_sha256=actual_state_manifest_sha,
    source_sha256=source_manifest)
save_checkpoint(path, trainer, identity, history_reset='clear')
load_checkpoint(path, trainer, identity, history_reset='clear')
```

path必须exclusive create，不覆盖旧checkpoint；包含新version、canonical action SHA、371D state manifest SHA、实际config、source SHA、policy/Adam、RNG和counters；build_identity自动加入当前五模块真实文件SHA，外部来源可追加但不能覆盖错误SHA。加载严格比对身份并拒绝旧视觉checkpoint，在改变live model前验证Adam moments/order/shape、noise范围和完整RNG。保存/恢复只在clean update boundary，history reset策略必须显式声明。checkpoint保存优化状态，没有保存或复现原生物理world；恢复PPO训练状态不构成无恢复原生成功证据。

## 验证计划与已执行结果

RED：现有CPU Python、PYTHONPATH=src、CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1，`pytest tests/test_dh116_privileged_teacher.py -q -p no:cacheprovider --maxfail=1`实际返回 **1 failed，exit 1**；首个失败断言是缺少`tabletop_privileged_teacher`模块，非collection error。未为RED修改生产源码。

随后完整 RED：27 tests failed，exit1，全部因对应新模块尚未创建。实施后增加坏checkpoint与精确resume测试，当前新测试33例：拓扑/likelihood一致、tanh饱和保存z；旧维拒绝；PPO pending/owner/issued-action隔离；storage detach/全batch/singleton；frozen likelihood；terminal与truncation GAE；generic history dict的bool/int保持及独立critic；noise投影Adam清理；BC optimizer隔离/group归一化；不合格专家拒绝；R0精确透传及未qualified R1/R2拒绝；checkpoint独占/身份/RNG/Adam/脏buffer/坏moment/坏noise原子拒绝；resume后下一PPO update的stats/counters/全部参数逐值一致；config opt-in。

最终纯CPU命令：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest tests/test_dh116_privileged_teacher.py tests/test_squashed_gaussian.py tests/test_privileged_teacher_reward_audit.py tests/test_surface_success.py tests/test_surface_reward_ablation.py -q -p no:cacheprovider
```

最终结果：**116 passed in 12.43s，exit0**（新测试33例、既有tanh分布8例、Phase0/success/profile75例）。`git diff --check` exit0，所审旧state/action/control/reward源码无diff。所有测试为CPU fixture或算术/合同检查，没有创建合格native force/support或成功示范，未安装/升级依赖。

本轮review修复：dictionary bool/int mask转型问题；native trace原始物理量NA必须有理由；checkpoint坏Adam shape/缺moments/RNG/noise在live变更前拒绝；resume精确恢复下一次PPO update；qualified optional reward要求current identity与原始evidence SHA。剩余物理验收及R2未实现扩展如上明确保留。

后续跨阶段审查发现：仅source/proof文件SHA加自称native JSON仍可能把任意bytes当成功证据。已新增回归先观察`DID NOT RAISE`，再将正式成功池收紧为每episode必须有`qualification_evidence`，由公共`tabletop_stereo_promotion.verify_native_episode`重读完整480Hz raw episode、producer/config/kwargs、正式成功窗及独立force/external-source资格，并精确匹配model identity和episode UID。缺包或verifier不可用时成功池关闭。R1几何也必须经该raw gate，R2支撑必须经`verify_force_qualification`原始per-contact审查；report自称qualified不授权。新Phase1测试数增至34，单独执行34 passed/exit0；schema-only BC仍只用于明确算术检查，不产生成功池资格。

## 规范4.3纯算术补全与剩余native缺口（snapshot02）

新增 `MeasuredTensor(value,valid,unit,source,role)`、`GeometryMeasurements`、`SupportMeasurements` 和transition/support context。物理数组不能broadcast：位置为[N,2,5,3]/[N,2,3] m、弧长[N,2] m、接触面积[N,2,5] m²、独立per-digit力[N,2,5] N。每个source有精确bool mask；valid=true却NA、单位错误、shape不符、实际算术溢出立即拒绝。NA/遮挡/来源不足保留NA与false mask，贡献为0不等于物理measurement为0。两手各自先对有效渠道平均再等权，单手来源不能冒充双手几何/承载证据。所有因子都受same_episode、transition_valid与原有phase_active控制，不引入成功判据。

R1的四项`geometry_terms(before,after,context,config)`均为`coefficient × (Phi_after - Phi_before)`，不乘dt，不给不变状态重复正奖励：

| 因子 | 物理输入与势能 |
|---|---|
| geometry_pad_distance | 实际pad和对应目标region点XYZ欧氏距离，Phi=-双手有效pad距离均值/pad_distance_scale_m |
| geometry_coordination | 实际两手grasp相对位移与目标两手相对位移的误差范数，Phi=-error/coordination_scale_m |
| geometry_arc_position | 实际与目标线缆弧长位置的双手绝对误差均值，Phi=-error/arc_scale_m |
| geometry_contact_area | 实际有效接触区域面积/目标区域面积，逐pad clip到[0,1]后双手均值作为Phi；已测量的无接触可以是valid面积0，未确认来源须mask |

R2的`support_terms`保留原signed upward support deficit在composer中一次，另外实现五类因素（过载分normal/tangential两条trace）：

| 因子 | raw/归一化与dt |
|---|---|
| hand_support_balance | 显式geometry-derived expected share[N,2] fraction，合法且sum=1；raw=Σ手\|实际手向上支撑N - share×实际总向上支撑N\|，除实际物重N，cost×dt；使用已测总支撑比较份额，避免再惩罚总deficit |
| opposing_pad_balance | 每手显式实际对向digit IDs，独立两pad法向载荷差N；按两者载荷和归一化、双手均值，cost×dt；不将整手力广播到指 |
| contact_recovery | 已确认required pad的接触比例改善，只有实际break recovery active-before、same recovery epoch和same episode才计势能差；无dt，正常hold/重置/换epoch不重复奖励 |
| normal_overload / tangential_overload | 逐pad实际过限载荷N的正部，除逐pad实际合格limit N，双手有效渠道均值，cost×dt；法向力不充当向上support |
| force_target_delta | 仅实际executed force target N的绝对变化，必要变化成本为0；不必要变化除实际force-target scale N，双手均值；离散delta不乘dt，单手不必要jitter也计成本；requested target或SDK joint target rad不能替代 |

左右承载**没有50/50默认**。`SupportContext.expected_hand_support_share`缺失/NA时该项raw和normalized保持NA、mask=false；shape必须[N,2]、unit=fraction、sum=1、每份额[0,1]。独立 `expected_support_share_from_geometry` 使用实际World两grasp、COM与gravity(m/s²)，在明确的quasistatic vertical point-support模型下用水平支撑线lever arms导出份额。重合支撑点、COM不在线段/横向模型容差内、来源不足均返回NA，不clamp猜比例。模型适用性还需要独立contact/dynamics/external-support资格；CPU 80/20 fixture证明合法非对称份额不被罚，不证明native载荷来源。

候选系数与尺度在独立 `privileged_teacher_shaping_candidates_v1.json`；`load_shaping_config`解析，enabled默认false。系数只作为未调参候选，不授予物理合理性或效果。`compose_native_reward`新增显式`shaping_measurements`/`shaping_config`参数；R0拒绝附加typed shaping，R1/R2先执行原来源gate，再计算新因子。检测generic/typed双owner、native同owner、support deficit重复；每个contribution/mask严格匹配native batch，防止[N=1]奖励广播到多个env。每项均有raw、normalized、coefficient、coefficient/raw unit、dt scaling、phase、source及原source mask、per-env与total贡献。

纯函数不要求先完整成功，所以可以CPU验证几何/恢复/负载算术。**现有formal `_qualification(...,'geometry',...)`仍调用完整native episode验证（包括force资格），独立geometry专用资格verifier尚未实现；不是仅仅等数据。** 实际pad/目标区域/弧长/有效contact area、COM/share模型适用性、per-digit向上/法向/切向力、recovery epoch、必要目标变化与executed force target的native measurement factory和逐项source/current-episode/tick绑定尚未接线。当前24D controller并不因新增接口而获得力目标；缺失力目标须NA，不能把joint target换算猜成N。正式Teacher CLI仍只允许R0。新因子代码完整不等于R1/R2原生训练/验收完整，Phase1整体继续pending。

TDD记录：缺模块RED；单手不必要target变化漏罚、有限输入算术溢出、缺config loader的回归先RED再修复；review补native batch防broadcast与移除把载荷`dex_balance`错当grasp协调的alias；新增actual geometry 80/20、缺share不fallback、非法份额与几何不适用回归先RED再GREEN。最终纯算术新26例全部通过（26 passed in4.13s，exit0）；完整关联回归命令和最终结果在收尾报告记录。所有来源均为明确CPU fixture，组合算术测试显式mock gate，不计native资格，未运行Isaac/GPU/native。

snapshot02最终关联命令（源随后冻结）：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest tests/test_dh116_privileged_shaping.py tests/test_dh116_privileged_teacher.py tests/test_squashed_gaussian.py tests/test_privileged_teacher_reward_audit.py tests/test_surface_reward_ablation.py tests/test_surface_success.py tests/test_dh116_privileged_cli.py -q -p no:cacheprovider
```

实际结果：**148 passed in 13.87s，exit0**。本轮文件为新增`tabletop_privileged_shaping.py`、`test_dh116_privileged_shaping.py`、`privileged_teacher_shaping_candidates_v1.json`，修改`tabletop_privileged_reward.py`的显式可选调用与两份disabled R1/R2配置说明。旧R0数学、旧controller、原Task/success没有修改。本次仍无native分辨率测试、Isaac执行、GPU训练或来源资格提升。
