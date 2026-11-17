# Phase 4：同步Teacher采集与独立BC

状态：已实现隔离的collector、版本化dataset与Student BC，并完成CPU真实`env.step`协议测试；没有运行Isaac、GPU、Teacher或Student原生训练，没有取得native成功资格。旧3844D/371D/24D源码与历史checkpoint不变。

## 新增模块与接口

|文件|责任|
|---|---|
|`tabletop_stereo_teacher_collector.py`|PreActionState/StepResult、同一pre-action查询与真实动作执行、identity/reset验证|
|`tabletop_stereo_supervision_dataset.py`|版本、来源/合同绑定、按env拆行、exclusive存取、来源分组holdout、typed history批处理|
|`tabletop_stereo_bc.py`|独立Adam、Teacher detached均值目标、动作/状态/可选辅助loss及梯度范数日志|
|`configs/dh116_training/stereo_bc_v1.json`|候选BC权重与epoch/batch配置，默认depth/temporal权重0|
|`tests/test_dh116_stereo_collection.py`|11项CPU执行/拒收/优化测试|

`PreActionState(sensor_history, privileged[B,371], privileged_valid bool[B,371], identities, extra_labels={}, label_validity={}, state_sources={})`提供`.clone()`，另支持显式`privileged_identities`与`force_qualification`。唯一Actor输入为typed `StereoHistoryBatch`；Teacher只得到privileged tensor。identity的全部公共keys必须相等，image age阈值默认1/60s。depth/point无效值遵守Phase3mask，不能以GT填补。

`StepResult(next_state, reward[B], done bool[B], info, issued_actions[B,24], applied_targets=None)`要求info.execution逐env保存真实物理tick范围和前后时间戳。collector保存pre-state身份，step后reward/terminal挂回该记录，不用post-state图像替代pre-action图像。非terminal后续history必须含真实issued action；terminal后episode/reset身份改变，history清至仅当前帧，previous action为0。

`collect_step(env, student, teacher, teacher_probability=1., schema_only=False, teacher_qualification=None, source_sha256=None, ...)`真正调用一次`env.step()`。正式模式先复用`promotion.require_teacher_qualification(report, Teacher_checkpoint_sha)`重读raw证据，并要求NativeStereoRuntime的`native_runtime_capture_unqualified`路径标记；标记本身不授资格。CPUfixture必须显式schema_only。`TeacherCollector.run(...)`返回bounded step记录列表。

提交前若env有`set_action_owners(list[str])`则明确写入逐env完整action owner，供native trace记录，不根据两个动作是否相等猜所有权，不修改动作/控制。

## 动作与数据层级

每条record分别保留`actor_raw{teacher,student}`（latent mu，可NA）、`teacher_mean`（deterministic tanh(mu)）、`teacher_sample`（默认NA，显式请求才采样）、`teacher_std`（latent）、`student_nominal`、`issued_actions`、`applied_targets`及真实`execution`范围。物理SDK targets保留原单位dict，缺失为None，绝不冒充normalized24或实测运动。

dataset记录RGB历史、RGB计算的depth/valid/confidence、world points、部署proprio与独立privileged标签/masks。既定SensorBatch无disparity字段；若runtime提供计算得到的`stereo_disparity_px`诊断extra_labels或同步raw sidecar，dataset会原样保留，loss不把未识别诊断字段当GT。

`build_dataset(records, source_sha256, contract=None, schema_only=False, qualification=None, purpose='successful_bc')`生成`dh116-stereo-supervision-v1`。公共contract强制canonical action SHA、当前371D state-schema SHA和calibration SHA。source SHA绑定Teacher checkpoint/来源artifact；正式资格gate重读真正文件。每行还必须有实际`physical_initial_condition_sha256`和`source_episode_group`，不允许用seed猜实际物理状态。

`save_dataset()` exclusive create；`load_dataset()`默认拒绝schema-only，并可核dataset文件SHA和完整expected contract。duplicate identity、非单调time/frame/physics、跨reset污染、上一issued不一致、错位privileged身份、动作owner不一致和成功标志篡改全部拒收。`batch_dataset()`重建typed history、detached Teacher目标及逐来源state masks，保留原payload与indices用于BC再次核对，防止用合格pool给无关tensor授信。

正式成功BC pool要求完整first episode从tick0到terminal，无recovery/restore，原生success。除Teacher资格外，还用`qualified_episode_identities()`重新绑定**本次采集的run/env/episode/reset UID**和native raw proof；旧Teacher成功报告不能给新采集episode授权。没有本次完整proof时，只能保存原始candidate，不得转换为成功demonstration。CPU schema pool固定`successful_demonstrations=False/native_qualified=False`。

holdout使用physical初始条件SHA和source-episode连通组；相同初态、同来源episode的所有replay/history副本一起分组，不随机相邻行。

## BC算术与边界

`StereoBCTrainer(actor, config=None, ppo_trainer=None, schema_only=False).step(batch)`为独立Adam，不含log_std，不更新Teacher，不插入PPOstorage，不改saved latent/log_prob。若共享PPO actor，必须位于clean empty rollout boundary。每次step重验dataset并比对history、target、mask与原indices，Teacher targets detached。

动作loss复用已有`masked_group_normalized_loss`：fingers12、world translation6、intrinsic rotation6分别算归一化均方误差，再平等平均有效组；物理scale先乘后除等价于canonical normalized差值，避免米/弧度混权。state loss复用Phase3的371D逐来源mask、SO(3)、force proof gate。

`lambda_action/state/depth/temporal`均显式配置；默认action/state=1、depth/temporal=0，都是未经holdout校准的候选。temporal正权重仅在有qualified velocity和跨帧point correspondence mask时计算当前/上一有效历史geometry差分速度loss；CPU测试有真实StereoActor反传。depth正权重必须有实际depth prediction head及qualified target/mask，缺失明确拒绝，没有GT fallback或伪造零loss。

日志返回各项loss、各自gradient norm、clip前总norm、state coverage及blocked原因。原生force标签仍NA，默认coverage0；没有硬件触觉输入，不能把force proxy当已验证传感器。

## 验证记录

实现前两份新测试一起RED：`16 failed in 1.94s`，exit1，均为明确新模块missing。最终CPU命令及联合结果见Phase5报告；Phase4测试11项通过，包括真实FakeEnv step、同步pre-state、stale/身份拒收、exclusive存取、来源holdout、成功资格拒绝、BC独立Adam/PPO边界、真实StereoActor优化和temporal辅助反传。FakeEnv明确`evidence_scope='cpu_fixture'`，没有native统计或物理成功声明。

尚未验证：同步原生Teacher全回合采集、BC holdout动作/状态误差、独立Student任务成功、depth auxiliary decoder/labels资格。默认正式gate关闭是预期状态。
