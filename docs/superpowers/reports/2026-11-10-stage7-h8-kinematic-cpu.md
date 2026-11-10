# Stage 7：H8 完整17D运动学外推 CPU实现与验证

本单元只发布独立库及测试，**未Native接入**。不替换默认Teacher、不改变3844D Actor/371D Critic/24D Tanh及原物理/材料/执行器/30mm/5s/0.5mm标准，PR保持Draft；不宣称MPC完成、新成功或B3/PPO许可。

| 本次代码 | SHA256 |
| --- | --- |
| [tabletop_bimanual_mpc_dynamics.py](../../../src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_dynamics.py) | `6096983ceaf3575269741bd075c2f3c16eb9199b0313a875e607978413a24ad1` |
| [test_bimanual_mpc_dynamics.py](../../../tests/test_bimanual_mpc_dynamics.py) | `c6db850345ae5210f791013fa8ce14c4dd6db11e6fc85559b6d19470cebf27ff` |

## 实现与数学边界

`canonical_candidate_digest`复用既有objective的HORIZON/CONTROL_PERIOD_S与候选摘要；严格接受8×24有限范围动作。`Full17VelocityForecast`显式携带64×2×17区间速度、valid mask、逐DOF单位、来源clock/descriptor/source及候选绑定。`rollout_kinematic_h8`计算每一行的`q_next=q+qd/480`，保留64个poststep和第8/16/.../64步控制端点，分别复用两手完整mounted FK。

当前SDK selected row的q/qd与未来假设分开；每手17行包含被动关节，按实际m/rad布局解释，不进行perfect-mimic代填，也不把wrist generalized qd冒成world twist。缺失/无效速度、错绑定、terminal/reset来源被拒绝，未选inactive batch的NaN padding不被选作状态。future只携带相对时间offset，不伪造未来实际SDK clock。

`measured_constant_velocity_forecast`仅重复当前已测full17 qd，明确candidate-independent。更换候选即使改变摘要，也不会令该常速轨迹变成动作响应。外部candidate-conditioned标签仍只是声明，不是合格物理响应。没有目标/驱动响应、RNEA、物体/接触预测、约束求解或动作提交；几何坐标/根姿态保持仍为显式假设，所有原生校准/动力学/规划/执行/硬件/成功资格为false。

测试包括64区间独立手算、左右不同轴闭式FK、混DOF与非零被动行、CV候选独立性、无效/缺失/错源拒绝与不可变性。初始模块缺失及current source/receipt、CV重标等反例实际先RED再GREEN；独审getter before/after clock缺失/错值/重复等8个反例也先RED，再定向8 passed，并完成同四文件154 passed回归。这些是CPU算术与来源边界，后述source15才是本报告所绑定的完整冻结证书。

## 已有synthetic CPU耗时观察

同SHA H8核心已有独立PRIVATE探针：一环境、双手各17 DOF/23 links的64子步纯调用，Torch intra/inter-op均2线程；imports与fixture建立在计时外，5次warmup后10次steady的P50为**134.39ms**。探针输入与23个source文件前后SHA保持，记录SHA为`55613e97e5f04ad929ec75ff2f468632e8248fd8ec60ca3085fc011382343869`。这是共享CPU合成fixture的分布观察，不是最坏耗时上界，不是Native闭环测时，也不提供16.67ms/60Hz资格；后续需要独立优化与原生时限验证。本次只读已有记录，没有重跑探针。

## 后续门槛

先有独立原生current FK/J校准与合格的候选动作响应/驱动模型，再补物体/接触、完整64区间碰撞/支撑/限位约束与solver。未知必需项保持NA/UNCERTIFIED，不能由常速或较小误差自动取得规划准入。此单元没有宣告完整H8 MPC完成。

## 真实冻结CPU证据

证书对应独立immutable `stage7_command_constraints_cpu15`，55个显式test文件，**1360 passed、3 CUDA-unavailable skips、238 subtests，pytest 258.05s、完整入口wall 261.466s，exit0，Native0**。本文件的39项全部通过，是上述整组的子集；H8与contact两份报告共用这一个证书，不能相加为两次独立完整验证。其它尚未发布/接线的算术模块也在这份组合中，不归成本提交新增实现。

CUDA被显式关闭；3个skip分别是native_sensor_batch的一项CUDA owner测试和common-pause两项真实CUDA mask测试。CPU证书不赋予CUDA/Native资格。源副本和父源原字节前后保持，四原controller保留；该清单不自动覆盖当前工作树后来单独进行的descriptor ABI修复。

| 闭合证据 | SHA256 |
| --- | --- |
| manifest | `9ebbb531c95878e84abd6e95bdeb98005c1c79c5837f2320783e00be77de0984` |
| validation | `716978bdc72565868f2c9b5b8f7d2bd0ec2b3fe69d977db78ddba21805174664` |
| command | `4f87a5f84b619ef73dade52bc11fa4f4f4475533a55cf58c320b6e08443e4059` |
| log | `345ae58f464174a9c9decb8b2e56d76ad0c20ec7ea2875fb100d0c23949dab4b` |
| junit | `58c58d807bc5d7b002e3dd309239a8435532306ef2f5db00ee1a43de4261c55c` |

实际source manifest：`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_command_constraints_cpu15/source_manifest.json`。记录的完整命令与显式环境如下，cwd为`/tmp/dh116_stage7_command_constraints_cpu15`：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash /tmp/dh116_stage7_command_constraints_cpu15/scripts/python.sh -B -m pytest -q -p pytest_mock -p no:cacheprovider --junitxml=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_command_constraints_cpu15/cpu_validation/pytest.xml \
  tests/test_bimanual_mpc_state.py \
  tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py \
  tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py \
  tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py \
  tests/test_bimanual_mpc_scene_measurements.py \
  tests/test_bimanual_mpc_contact_provenance.py \
  tests/test_bimanual_mpc_aux_capture.py \
  tests/test_bimanual_mpc_aux_integration.py \
  tests/test_bimanual_mpc_articulation_measurements.py \
  tests/test_bimanual_mpc_contact_delivery_probe.py \
  tests/test_bimanual_mpc_model_capture.py \
  tests/test_bimanual_mpc_objective.py \
  tests/test_bimanual_mpc_model_integration.py \
  tests/test_bimanual_mpc_experimental_options.py \
  tests/test_bimanual_mpc_delivery_integration.py \
  tests/test_bimanual_mpc_filtered_contact_provenance.py \
  tests/test_bimanual_mpc_filtered_contact_measurements.py \
  tests/test_bimanual_mpc_filtered_contact_integration.py \
  tests/test_bimanual_mpc_contact_epochs.py \
  tests/test_bimanual_mpc_filtered_contact_values.py \
  tests/test_bimanual_mpc_contact_epoch_integration.py \
  tests/test_bimanual_mpc_model_descriptor.py \
  tests/test_bimanual_mpc_mounted_kinematics.py \
  tests/test_bimanual_mpc_model_descriptor_integration.py \
  tests/test_bimanual_mpc_dynamics.py \
  tests/test_bimanual_contact_runtime.py \
  tests/test_tripod_contact_recovery.py \
  tests/test_tripod_recovery_runtime.py \
  tests/test_bimanual_mpc_command_bridge.py \
  tests/test_bimanual_mpc_constraints.py \
  tests/test_bimanual_load_transfer.py \
  tests/test_vision_command_shadow.py \
  tests/test_stage6_bilateral_recovery.py \
  tests/test_stage6_common_pause_cuda_readback.py \
  tests/test_stage6_tripod_retired_transport.py \
  tests/test_stage6_tripod_entry.py \
  tests/test_tripod_stage_interrupt.py \
  tests/test_tripod_pressure_hold.py \
  tests/test_tripod_execution_guard.py \
  tests/test_tripod_candidate_planner.py \
  tests/test_tripod_nonlinear_geometry.py \
  tests/test_tripod_native_lifecycle.py \
  tests/test_vision_teacher_pressure_mpc.py \
  tests/test_vision_teacher_constrained_mpc.py \
  tests/test_geometric_constrained_mpc.py \
  tests/test_stage6_common_tracking_pause.py \
  tests/test_stage6_common_pause_runtime.py \
  tests/test_stage6_tracking_calibration.py \
  tests/test_full_episode_tracking_experiment.py \
  tests/test_teacher_pressure_reference.py \
  tests/test_stage6_tracking_batch.py \
  tests/test_bimanual_tracking_guard.py
```

原CPU结果行：

```text
1360 passed, 3 skipped, 238 subtests passed in 258.05s (0:04:18)
```

## 原生状态与限制

既有B运行使用原Stage6恢复路径，其运行或结果不算本新wrapper/H8的Native验证。D1曾因USD property-stack ABI导致元数据NA，该结果不构成FK/J通过；ABI修复与D2应按独立source/闭合证据报告，本文不替代其结果。

未来动作响应、驱动/物体/接触模型、完整约束与solver、真实支撑和60Hz期限均未由本报告取得资格。没有新专家池或Student成功，也不开放B3/PPO、默认Teacher替换或正式100不同初态配对验收。
