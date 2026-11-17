# WP2　143D 状态监督的三种训练模式

基线为 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。本轮复用既有 `TaskStateAdapter`、371D 合同、`StereoStateEstimator` 和原生标签核验入口，没有重建 143D 定义或改变旧 Actor 的默认输入。

## 实际修改

`tabletop_stereo_state_supervision.run_training_mode()` 明确区分三个模式，现有 `scripts/train_dh116_stereo_state.py` 直接调用更新后的入口。

|模式|执行内容|模型与资格|
|---|---|---|
|`schema_test`|重读候选、480Hz 账本、同帧实例参考与 SHA；验证历史输入、371D 标签和掩码结构|无优化、无模型、无资格|
|`diagnostic_fit`|只对独立原生参考与同钟 SDK 读回共同核验的 61、62、63 中心坐标拟合；允许少量真实初态|独立诊断模型，只有训练样本内误差；validation/test 为 NA，不能用于 BC、部署或泛化晋级|
|`formal_training`|保持原有物理初态与源 episode 连通分组；train/validation/test 不重叠；各 Head 分别核验真实来源|有完整三路数据才允许优化；模型训练结束也不自动授予物理成功或部署资格|

三份独立配置为 `stereo_state_schema_test_v1.json`、`stereo_state_diagnostic_fit_v1.json`、`stereo_state_formal_training_v1.json`，通过 `--config` 实际读取。CLI 的 `--mode` 可以显式选择模式。配置不允许写入或覆盖资格字段，旧配置不变。

`_optimize_head()` 为诊断和正式模式共用的掩码优化路径。它记录每个 batch 的原始行索引、loss、梯度范数、视觉 Encoder 梯度和未监督输出行的最大梯度。原有 Geometry→Velocity→Deformation→Contact→Slip→Force Proxy 遍历及来源检查保留。诊断模式本轮只训练 canonical `privileged_state[61:64]`；额外 `geometry[128,3]` Head 没有真实完整标签，未训练。

中心坐标使用原合同的 `(object_center - mean(left_wrist_origin, right_wrist_origin)) / 0.05 m`，方向为 World XYZ。`_metrics()` 同时报告归一化误差和恢复为米的误差。逐样本标签审计保留 identity、有效掩码、归一化值和米值，缺失坐标保留 NA。

还修复了一个字段来源统计问题。`_admit_rigid_labels()` 追加真实几何监督时，原代码覆盖了编码器来源标记，使后续 143D 审计漏计已读取的关节字段。本轮保留 `direct_readback_source_scope`，最新真实读回中 24 个编码器维度各计 8 个样本；新增几何标签依然只有 3 维各 6 个样本，没有扩大其他标签掩码。

## 真实历史样本的离线拟合

这部分是本轮重新执行的 CPU 离线拟合，输入来自上轮已保存的原生采集，不是新增原生实验。最终服务器输出为 `/tmp/dh116-stereo-wp2-modes-20261111-v04`，使用最终 `/tmp/dh116-stereo-wp-final-preflight-v03/snapshot`，快照 manifest SHA 为 `fab9451a7be5decf9e5f484aa62ecfadb8bcd623f8db05d7ce3fae71a6eade6e`。解释器为现有 Python 3.10.12、PyTorch 2.7.0+cu128，`CUDA_VISIBLE_DEVICES` 为空，未启动 Isaac。

原始候选 SHA 为 `4da4f234db2ff77c505d100366bcf30b5c9b2d007f48cfd5b132c1e5a0f8cd70`，原始物理账本 SHA 为 `da81d600f58864b059cde81131353ce6384be5e84f53bc6446a94bb7b00c6929`。每次执行均重读这两个文件及逐帧原生 Wire instance ID/光轴深度参考，并重新核对 Teacher、相机、episode、policy tick 和 physics tick 身份。

数据为 8 行、2 个独立物理初态，二指与三指各 4 帧。每种抓法首帧缺少同钟前一物理样本，只有 policy tick 1–3 的中心坐标能入训。有效标签为 6 行 × 3 维，占整个数据集中心标签槽的 75%。下表是这 6 个训练样本上的统计，不是独立验证集结果。

|中心坐标|拟合前 MAE mm|拟合后 MAE mm|拟合后 RMSE mm|拟合后 P95 mm|
|---|---:|---:|---:|---:|
|X，371D 索引 61|124.533763|5.394501|5.671066|7.519180|
|Y，371D 索引 62|7.967745|1.521719|1.572873|2.012610|
|Z，371D 索引 63|83.287327|4.257552|4.522546|5.963621|

诊断使用原 128 维空间特征、两层时序网络、四帧历史和掌局部点输入。seed=116，lr=0.0003，batch=2，image-processing batch=1，3 epochs，共 9 次优化。平均 batch loss 从 1.056469123 降至 0.003672556，9 次 Encoder 梯度均非零，最小为 0.201975790；未监督 canonical 输出行梯度最大值始终为 0。未知不确定度没有被当作校准误差，校准半径为 NA。

`diagnostic_fit/state_estimator.pt` SHA 为 `4bc8c94a1fe8095e599e122fd99ad77277aab2bbdda29df184ecaa5126befa49`。保存格式 `dh116-stereo-state-fit-v2` 绑定模式、模型结构、371D 合同 SHA、候选 SHA 和账本 SHA。`generalization_qualified`、`actor_deployment_qualified`、`successful_demonstration_pool` 均为 false。实际重新创建模型并严格加载全部 state_dict 后，所有张量与保存值一致；与相同 seed 初始模型比较，canonical 输出只有 61–63 行发生改变，其他输出行与额外 Head 权重保持逐位一致。共享 Encoder 和时序网络参与训练，不能把未训练 Head 的数值输出视为合格估计。

最终 v04 已包含本轮新增的 `point_provenance` 数据通路、Encoder 实现、cold candidate 窄准入及初态跨链绑定。该诊断继续使用默认 `point_source_encoding=False`，旧候选缺失的来源编码保持 unknown，不重新标注为材料节点。v01、v02、v03 和最终 v04 的模型 SHA 完全一致，核验了兼容默认计算路径；v04 严格加载校验与逐维输出比较也重新执行，`fit_verification.json` 与 v03 逐字节相同，SHA 为 `fb6798d4455f70410291bbb7a7dd7f21f2e9d378af8fcde1cce591a970af5ad4`。全部已执行模块的字节 SHA 在私有 `v04/source_manifest.json` 中；最终监督实现 SHA 为 `e9082164de2d82450582524fda3df801862052c0b1eabd60b1e59574bb030fb7`。

最终三种模式执行结果如下。

|模式|退出码|实际耗时|结果|
|---|---:|---:|---|
|schema_test|0|6.327 s|schema_validated，无优化|
|diagnostic_fit|0|51.829 s|diagnostic_fit_completed，仅上述三个中心坐标|
|formal_training|2|5.679 s|blocked_insufficient_independent_groups，无模型|

确切命令、完整逐 batch 梯度、逐标签 identity、输入与模型 SHA、读回比较保留在私有 `artifacts/stereo-wp2-20261111/v04/`。各次远端独立输出和原候选均保留，没有覆盖旧模型。

```text
python scripts/train_dh116_stereo_state.py --config configs/dh116_training/stereo_state_diagnostic_fit_v1.json --candidate <original/stereo_teacher_candidates.pt> --candidate-sha256 4da4f234db2ff77c505d100366bcf30b5c9b2d007f48cfd5b132c1e5a0f8cd70 --raw <original/native/native_physics.jsonl> --raw-sha256 da81d600f58864b059cde81131353ce6384be5e84f53bc6446a94bb7b00c6929 --output <new/diagnostic_fit> --device cpu
```

## 测试与剩余门槛

测试先复现 6 项缺失模式能力，随后分别复现配置准入和编码器来源漏计问题，再修改实现。专门模式/监督测试 21 项通过。包含三种 Actor、TaskStateAdapter、Estimator、历史采集 CLI 的 78 项回归通过；日志保留在私有 `artifacts/stereo-wp2-20261111/`。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_training_cli.py
```

Geometry 除中心 XYZ 外仍无当前合格标签。Velocity 缺连续帧共享的已验证轨迹，Deformation 缺真实柔性求解器状态，Contact/Slip 缺合格逐接触和固定锚测量，Force Proxy 缺资格化反作用力来源。以上 Head 均保持关闭。

下一次采集需复用已有 `tabletop_privileged_evaluation.prepare_cold_trial_batches()`、冷重置 buffer 和实测 readback 身份路径，采集左右 RGB、同钟 Teacher 371D 标签、实际 24D 动作以及原生实例参考。默认 reset jitter 为零，增加 env 数或更换 seed 不能代替实测独立初态。每种抓法至少 3 个确实不同的初态只能满足最小三路拆分，不能建立泛化或不确定度资格；还需要更多独立组与完整回合。该原生采集须与 WP1 对照结果结合后按用户要求另行申请，本报告不将未采集数据计入结果。

## 冷初态采集代码与待授权清单

随后实际接入 `tabletop_stereo_training_cli.collect_cold_candidate_reference()` 和 `--cold-trial-manifest`，复用原有 `_install_reset_trials()`、`_readback_reset_trials()`、reset buffer 暂存与恢复。先在影子 buffer 核验全部初态与原限位，再逐批原生 reset，实际读回 SHA 和 episode UID 必须在整个采集 cohort 内唯一，源分组使用原 manifest trial ID。每批清空相机历史后复用原 `collect_candidate_reference()`，执行 Teacher deterministic mean；任一环境首次 terminal 后结束本批，禁止将自动重置后的回合冒充该初态延续。

`stereo_candidate_cold8_v1.json` 是独立的有界配置，CLI 实际约束 2 环境、每种抓法 4 初态、每初态最多 4 control steps、Roll20 surface96 相机。manifest 数量、相机标定和 Teacher 文件 SHA 在 CPU 预检，原物理合同在实际执行前再核对。原正式评估的 100 初态/抓法门槛未修改，小 cohort 明确为 candidate-only 数据，不能获得成功专家资格。

还修复了多次冷 reset 的参考文件冲突。原先 `state_reference_<policy_tick>` 会与上个 batch 的 post-step 图像重名，因为 reset 本身不推进 policy tick。`tabletop_stereo_runtime.state_reference_stem()` 追加本批 episode UID 的摘要，使不同 reset epoch 的同 tick 参考分别保存，旧文件不覆盖，物理与渲染时钟不改写。

CPU 先复现缺少 cold collector、reset 同 tick 文件冲突、超预算及标定误配未拒绝，再修改实现。冷采集、原有 cold evaluation、采集 CLI 和 Teacher 原生身份路径回归共 **49 passed / 16.02s**，私有日志为 `cold_final_v02.log`。这些 fixture 只验证 buffer/身份逻辑，不计入原生次数。

后续完整预检又发现并修复了无 terminal 的有界冷 reset 数据无法读回、异常部分 RGB/Teacher 记录丢失、Teacher 文件 SHA 缺少加载前后双检，以及 reset readback 与 raw 初态 header 缺少跨链比较。正式 BC/DAgger 的 terminal 和成功门槛保持原样。最终针对 cold admission、collector、state supervision 和三训练模式共 **43 passed / 10.44s**，另由独立 reviewer 复核原点/四元数/错槽等 5 项通过；详见独立 cold8 预检计划。此前统计是各次实际执行的局部回归，不能相加冒充新原生次数。

已在 CPU 上从本轮真实 Teacher `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`、其真实 `native_task_contract.json` 和原始左右初始化文件，用原生成器固定取前 4 个初态/抓法。保留完整 200 个 authored 原始 manifest 与来源 sidecar；本次申请仅使用其中固定的 8 个，没有执行这 200 个物理实验，也没有按策略表现挑选初态。

- 8 初态 manifest 文件为 `/tmp/dh116-wp2-cold8-plan-20261111-v01/candidate_cold8.json`，SHA `e2e1280e02cfe193678b2474a179e4e7207c4239d75641b7e2efc2ad5c969e83`。
- manifest semantic SHA 为 `afbb1984ab20c296921e251a78846eb9cab665582b35a6328eec73eeea5fed78`。
- 相机标定 SHA 为 `02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`。
- 实际 base contract 文件 SHA 为 `71d1d95d8e7630d941af119fad08ff2724a0ef7d40393d513f746fbf900e6179`。

**用户本轮未批准新采集，新增原生采集次数为 0。** 目前只验证 authored 初态，不声称 8 个实际读回初态已通过。最多 32 个候选样本，实际个数取决于 terminal 和来源核验。已补充独立的 [cold8 逐字段预检和执行计划](2026-11-11-stereo-cold8-preflight-plan.md)，清楚区分 CPU 已证与原生待证；必须待用户明确批准后，由主任务统一冻结源码、排查单 Isaac 实例并执行，输出必须使用全新目录。

```text
scripts/python.sh scripts/collect_dh116_stereo_teacher.py --config configs/dh116_training/stereo_candidate_cold8_v1.json --teacher-config configs/dh116_training/privileged_teacher_ppo_guarded_v2.json --teacher-checkpoint /tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000001.pt --camera-config configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json --native-task-config configs/dh116_training/stereo_native_task_v1.json --candidate-only --state-reference --cold-trial-manifest /tmp/dh116-wp2-cold8-plan-20261111-v01/candidate_cold8.json --cold-minimum-per-style 4 --num-envs 2 --steps 4 --seed 116 --output <exclusive-new-output> --enable-experiment --execute-native
```
