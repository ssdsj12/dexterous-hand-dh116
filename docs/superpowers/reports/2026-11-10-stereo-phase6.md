# Phase 6：独立双目 Student asymmetric PPO 与部署包

此阶段实现软件训练/保存/部署路径；CPU fixture 不能证明 Isaac/PhysX 原生成功。没有执行 native、GPU训练、长训、checkpoint迁移、push或成功池扩充。旧3844视觉分支和Phase0所审371D/24D任务源码保持原样。

## 输入隔离与优化

`tabletop_stereo_student_ppo.py` 的 `StereoAsymmetricPolicy(network_config)` 包含真实 `StereoStudentActor` 与独立371→256→256→1 critic。actor的latent_mean/deterministic只接收exact `StereoHistoryBatch`或其精确sensor dictionary；未知Teacher、critic、GT和success keys立即拒绝。critic `.value(privileged371)` 从不同参数与调用路径计算，改变critic truth不改变actor action，critic loss不产生actor gradient。

`history_to_tree(history)` 持有输入副本，按depth_valid/point_valid/history_valid只将被遮罩的NA depth/confidence/points中性化，使PPO存储有限；不修改原始sensor，不补GT，不把缺失值标有效。bool mask始终bool，history_valid保留。RGB、测量本体状态、时间等仍由原sensor validator严格校验。

训练构造接口为 `make_student_trainer(config,num_envs,device='cpu',rollout_steps=None)`。config使用`network`、`ppo`、`rollout_steps`及默认关闭的`auxiliary_bc`。PPO复用Phase1同一saved-latent tanh Gaussian、owned pending/token、on-policy版本校验、GAE/terminal/bootstrap、all-rows minibatch与noise projection数学；不复制第二套PPO。Student spatial/temporal dropout必须0，避免likelihood重算噪声。log_std直接是actor的24D参数。

`attach_auxiliary_bc(trainer,bc_trainer,metadata=...)` 要求clean boundary、独立optimizer且不含critic参数、native来源/数据SHA/原始qualification evidence。调用时再次检查buffer/pending，绝不插入PPO storage。默认config权重0/disabled；来源不能证明时拒绝，不自动启用DAgger或Teacher标签。

## 新checkpoint与actor-only export

训练checkpoint version=`dh116-stereo-rgb-student-checkpoint-v1`，显式保存完整policy、Adam、PPO实际config、counter、policy_version、RNG、history_reset、network及common Student contract。common contract验证当前371D state schema SHA、canonical action SHA、calibration、输入/输出语义和完整config；额外保存actor输入contract、相关实现source SHA及policy tensor SHA。拒绝旧visual3844、网络/输入keys/version/calibration/SHA不一致、dirty PPO boundary及隐式history恢复。

```python
save_student_checkpoint(path,trainer,contract,network_config=config['network'],history_reset='clear')
load_student_checkpoint(path,trainer,expected_contract,history_reset='clear')
export_actor(checkpoint_path,actor_output_path)
actor = load_deployment_actor(actor_output_path,expected_calibration_sha256=calibration_sha)
action = actor.deterministic(sensor_history)
```

所有输出使用exclusive path，不覆盖旧checkpoint。载入在改变live model前验证policy topology、Adam moments、noise、RNG与计数。历史重置只允许clear；优化resume不等于复现物理world，也不产生无恢复原生成功资格。

部署package version=`dh116-stereo-rgb-deployment-actor-v1`，只含actor.state_dict、network config、calibration SHA、preprocessor说明与独立actor input contract。没有critic weights、Teacher、optimizer或env state。deployment loader只构造纯 `StereoStudentActor`，顶层不导入PrivilegedTeacher/PPO；训练用PPO与训练checkpoint helper采用lazy import。estimator中保留的预测heads是模型输出，不能被解释为模拟器GT输入。

## 验证记录

先执行RED：新测试首例因缺少Phase6模块失败，exit1。实现两模块后15 passed / 1 failed，唯一失败为尚未创建default config；随后补配置。当前测试覆盖真实shared ResNet encoder反传、critic反事实隔离与gradient隔离、mask bool与NA、禁止truth keys、保存/载入/actor-only export逐值重放、独占路径、legacy/input/calibration/network/actor tensor SHA拒绝、全新Python进程证明部署无Teacher/PPO依赖及dirty auxiliary拒绝。

继续增加训练后Adam/RNG恢复、坏moment原子拒绝、dirty checkpoint与完整experiment contract拒绝；声明的backbone/resolution现在是实际约束，不允许配置说PointNet而实际运行ResNet，声明分辨率与RGB history不符也拒绝。该约束先有RED回归（未拒绝错误backbone），修复后通过。训练checkpoint还绑定实际PPO math source SHA与完整Adam hyperparameters。

最终CPU命令（既有Python，不安装包、不导入Isaac，GPU关闭）：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest tests/test_dh116_stereo_student_ppo.py tests/test_dh116_privileged_teacher.py tests/test_squashed_gaussian.py tests/test_privileged_teacher_reward_audit.py tests/test_surface_success.py tests/test_surface_reward_ablation.py -q -p no:cacheprovider
```

实际结果：**136 passed in 25.00s，exit0**（Phase6新19例、Phase1新34例、既有分布8例和Phase0/success/profile75例）。实际共享ResNet encoder参数产生非零gradient并更新；部署包在全新Python进程中载入且没有import PrivilegedTeacher/PPO，推理动作与保存前逐值相同。

后续同时收紧Phase1正式成功BC入口和R1/R2资格：公共 `verify_native_episode` / `verify_force_qualification` 必须重读完整native raw evidence，自称native JSON或任意source bytes不能授予资格；不具备证据时默认关闭。Phase6 auxiliary沿用同一原始资格gate及独立BC dataset校验。

Native完整30mm/5s/.5mm成功、正式Student PPO效果、随机初态/抓法覆盖、部署速度和硬件资格仍未取得证据。优化checkpoint不是物理world快照；保存清空history与重新运行物理过程不能被称作原回合无恢复延续。

## Student CLI、640×480与BC warm start补充

已新增 `tabletop_stereo_student_cli.py` 和 `train_dh116_stereo_student.py`、`run_dh116_stereo_student.py`、`export_dh116_stereo_student.py`。训练默认关闭；`--check-config`只读校验，不创建run、不导入Isaac或启动native。实际脚本默认检查返回640×480、image_processing_batch_size=2、native_started=false（exit0）。新默认配置为独立 `stereo_student_ppo_accuracy640_v2.json`，原224配置未覆盖；CLI按所选camera实际分辨率派生network与新contract，不静默resize回224。

训练需同时明确 `--enable-experiment --execute-native` 和全新output。正式模式在启动native前调用公共Teacher与stereo原始资格gate，需要`--teacher-proof --teacher-checkpoint --stereo-proof`并绑定实际checkpoint文件SHA和camera calibration；缺证据就拒绝。`--diagnostic-only`限制为一update、最多2环境、rollout<=8，所有产物明确notqualified。CPU测试替换launcher/runtime时保存evidence_scope=cpu_fixture，不登记native。

实际训练loop使用当前同步sensor history给actor、独立有限371D给critic、native R0 reward一次、实际issued action与owned sample精确一致；复用step协议检查physics/time/episode/reset与previous-issued对齐。`privileged_valid`是Student监督资格mask，172个NA_forstudent不等于critic的numeric physics invalid；回归覆盖全false监督mask仍能用有限原生371D做诊断PPO，未给这些字段补0或伪造有效监督。terminal后SDK读数NA按原mask记录为JSON null。

新增 `save_actor_package(path,actor,contract,network_config=...)` 支持显式BC/DAgger/Student角色，只保存实际StereoStudentActor权重、网络、校准和独立actor contract，不伪装PPO optimizer或授予native资格。`--actor-package`只warm start actor，critic/Adam重新初始化；同拓扑、canonical action、state schema、calibration严格绑定，允许显式改变image chunk资源与说明性network字段，记录非exact resume。warmstart噪声在第一次rollout前按当前PPO envelope投影；回归先观察std=.5违反当前max=.3，再修复并通过。`--checkpoint/--resume`仍是完整严格优化resume，两类输入互斥。

```text
python scripts/train_dh116_stereo_student.py --check-config
python scripts/train_dh116_stereo_student.py --enable-experiment --execute-native --diagnostic-only --num-envs 2 --updates 1 --rollout-steps 8 --output NEW_RUN
python scripts/run_dh116_stereo_student.py --actor-package ACTOR.pt --observation SENSOR_HISTORY_TREE.pt --output NEW_ACTIONS.json
python scripts/export_dh116_stereo_student.py --checkpoint STUDENT.pt --output NEW_ACTOR.pt
```

离线infer纯Actor运行，不导入PrivilegedTeacher/PPO；saved history仅允许精确sensor keys。原生infer是固定actor-package、bounded steps、无Teacher/MPC/critic执行，物理诊断仍不作为Actor输入。优化日志与每次control token/z/action/logp/来源identity写入新run；native或模型异常保留failure.json，不清理证据。

最新完整CPU回归命令为上述6个基础测试文件，再加 `tests/test_dh116_stereo_student_cli.py`、`tests/test_dh116_stereo_training_cli.py`、`tests/test_dh116_stereo_runtime.py`：**153 passed in 35.44s，exit0**。其中CLI11例、Phase6 PPO/export20例、Phase1/既有相关117例、root BC/runtime集成5例。`git diff --check` exit0（仅仓库既有LF/CRLF转换提示）；原物理Task/env/reward/action contract无diff。本次仍未运行native或长训。

## 预处理身份与冻结执行描述补充

新增 `tabletop_stereo_preprocessing.py`。`processing_contract(camera_config)` 按实际 `StereoDepth.from_config(...).to_config()` 记录 matcher、subpixel、disparity、质量阈值和深度范围，并绑定 RGB/坐标约定、真实 warm-colour skeleton/128点算法与阈值及 depth/perception/geometry/preprocessing 四个实现文件SHA。image_processing_batch_size仅影响资源分块，不进入语义SHA；改变matcher参数即使calibration相同也改变processing SHA。此软件合同不授予精度或native资格。

`resolve_config` 总是从实际camera生成`config.preprocessor`。BC/DAgger/Student actor package通过common config把它绑定进actor input contract。没有该声明的typed-history包显式标记`history_input_only_unspecified`，仍可对保存的sensor history离线推理；`require_raw_processing(actor.deployment_contract,actual_camera)`拒绝将这种包用于声称同一raw RGB pipeline，也拒绝same-calibration/changed-matcher。正式PPO warmstart需要完全相同processing；仅短窗口diagnostic允许重绑，并在initialization.json记录来源/目标SHA和非资格scope。

原生Student CLI在首个物理step前调用Bridge的`set_runtime_descriptor`。冻结Actor执行写入teacher_execution_enabled/mpc_enabled/critic_deployed/privileged_actor_inputs全部false、critic_training=false、精确RGB/depth来源、实际actor package字节SHA（同时作为本次model_checkpoint_sha256）、artifact路径SHA与processing SHA。训练明确execution_mode=student_PPO_training、critic_training=true、mutable policy身份，只记录initialization checkpoint而不把更新中的模型声称为冻结Actor。公共promotion gate须重读每条raw证据；descriptor本身不能授予native资格。

新增入口先RED（缺config.preprocessor、缺formal warmstart processing gate），实现后CLI与processing测试**16 passed in 16.48s，exit0**。额外覆盖纯Actor执行descriptor/实际package SHA与same-calibration matcher变化拒绝。脚本默认`--check-config`再次exit0，解析640×480、subpixel=true、imagechunk2、实际processing SHA且native_started=false。完整扩展组首次**165 passed / 1 failed in 42.98s，exit1**，唯一失败是Runtime旧CPU fixture的camera.json为空字典；真实标定是processing合同的必要输入，已交Runtime文件owner修正fixture，未放宽生产校验。最终全组结果在下面追加。

本代理文件最终独立回归：下列8文件 **153 passed in 37.62s，exit0**。集成Runtime旧fixture的修复/重跑由其owner记录；未将尚未复核的集成结果写成通过。

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest tests/test_dh116_stereo_preprocessing.py tests/test_dh116_stereo_student_cli.py tests/test_dh116_stereo_student_ppo.py tests/test_dh116_privileged_teacher.py tests/test_squashed_gaussian.py tests/test_privileged_teacher_reward_audit.py tests/test_surface_success.py tests/test_surface_reward_ablation.py -q -p no:cacheprovider
```

独立review还发现部署包loader返回eval，DAgger随后BC更新没有切train，导致VGA共享encoder虽分块却不启用训练activation checkpointing。已先增加真实loaded Actor回归并观察encoder `(training=False,grad_enabled=True)` 的RED；`StereoBCTrainer.step`现在仅优化期间切train，并用finally逐module恢复原来的完整mode（含异常路径），DAgger继承同一入口。真实BC/DAgger encoder反传更新、nonreentrant checkpoint实际被调用、失败不污染模式均有CPU验证：`test_dh116_stereo_bc_mode.py`、collection、dagger、training_cli四文件 **28 passed in 14.28s，exit0**。这是内存路径控制修复，没有运行VGA GPU测量，也没有授予性能资格。

## 显式训练来源登记与传承

`tabletop_stereo_training_lineage.py` 和 `register_dh116_stereo_training_lineage.py` 提供独立artifact登记。模型、原始JSONL、训练dataset与parent manifest均按实际文件SHA重读；每个episode_start的实际object/palm/joint状态重算physical_initial_condition_sha256，source_episode_groups来自原始header，不能由报告手填列表替代。parent/raw/dataset依赖递归合并，重复身份冲突或SHA变化拒绝。CPU/unknown来源、非实际readback、缺父模型manifest或不明确完整性时complete_source_inventory=false。即使显式请求complete，登记也仅表示已声明来源的本地闭包，不能自动发现未登记训练，更不是native成功或heldout资格。

Actor input contract现在保留纯metadata的training_provenance，拒绝Teacher/critic网络、GT tensor和数值数组；保存检查实际source artifact，部署可独立推理而不需要载入Teacher/PPO/训练dataset。`load_deployment_actor`附实际deployment_package_artifact路径SHA。PPO warmstart承接BC/DAgger输入包与其来源，优化resume恢复trainer.training_provenance再追加本次来源；保存到actor contract而不改变严格优化config SHA。native raw在checkpoint里只保存来源路径，关闭producer后由独立登记CLI计算最终SHA；训练中不伪造未关闭文件的最终hash。export追加实际源训练checkpoint路径SHA。

```text
python scripts/register_dh116_stereo_training_lineage.py --model-checkpoint FINAL_ACTOR.pt --raw-episode-artifact CLOSED_NATIVE.jsonl --parent-manifest INPUT_MODEL_MANIFEST.json --training-artifact DATASET.pt --output NEW_MANIFEST.json
```

只有明确请求`--declare-source-inventory-complete`且可审查来源闭包通过才可能标记inventory complete；默认false。公共heldout/promotion还必须独立验证真实native rollout，登记不代替物理证据。新的来源登记先因模块缺失RED，优化resume/export传承再因字段缺失RED；最终lineage7、StudentCLI14、StudentPPO20、BCmode3共 **44 passed in 34.65s，exit0**。所有positive只是CPU metadata/算术，登记夹具均complete=false/native_qualified=false，没有构造伪native合格positive。注册脚本`--help` exit0，未启动Isaac；这些源与CLI现已冻结供整体CPU snapshot。
