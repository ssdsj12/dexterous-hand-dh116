# Phase 1 Privileged Teacher 实现与验证记录

当前状态：Phase 0 已由主任务封存为 `497e290`；五个独立模块、四份配置及 CPU 测试已实施。本文只报告软件验证，没有宣称 native 训练、完整原生成功或 force qualification。Phase 1 的正式原生验收仍须由独立 CLI/native callback 运行并提供证据。

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

R1当前实现qualified geometry potential差分（coefficient1、无dt），R2在此基础上仅实现qualified signed support deficit（-1.5points/s×clip((W-S)/W,0,1)×physical_dt），且检测已有native support deficit重叠时报错。**R2 的contact recovery、bilateral balance、overload、force-target change等扩展尚未实现；尚无这些来源的合格native证据，不能把当前subset称作完整R2。** 四份config默认仅R0 enabled；R1/R2 enabled=false，不自动启用或迁移成功标准。

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
