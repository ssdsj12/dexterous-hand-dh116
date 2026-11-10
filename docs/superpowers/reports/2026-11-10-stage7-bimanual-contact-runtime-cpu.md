# Stage 7：双手接触状态与原预算委托 CPU实现与验证

本单元只发布独立库及测试，**未Native接入**。不替换默认Teacher、不改变3844D Actor/371D Critic/24D Tanh及原物理/材料/执行器/30mm/5s/0.5mm标准，PR保持Draft；不宣称MPC完成、新成功或B3/PPO许可。

| 本次代码 | SHA256 |
| --- | --- |
| [tabletop_bimanual_contact_runtime.py](../../../src/go2_pvcnn/control/dh116/tabletop_bimanual_contact_runtime.py) | `2088f16f13217f2515fe12d9dbb9b40213c615f7466559cb3c4e862212b04dca` |
| [test_bimanual_contact_runtime.py](../../../tests/test_bimanual_contact_runtime.py) | `52f8622bb9a15c47a74f9303514bb8dfefca0096995d2e07af6a52bada31b944` |

## 状态、来源与执行记账边界

`BimanualContactRuntime`以N×2×5状态复用原`TripodContactRecovery`。左右分别显式选择pinch的thumb/index或tripod的thumb/index/middle，输出NO_CONTACT、CONTACT_CANDIDATE、VALID_CONTACT、STABLE、SLIPPING、LOST、REACQUIRING。接触measurement未知时state为NA、valid=false，不把未知写成LOST；几何间隙≤1mm且cos≥0.8只产生候选，不产生真实contact事件。

`SlipThreshold`若配置，必须有明确m/s数值、来源与source SHA；None不推0。slip/opposition未知各自保持NA；双侧opposition需要实际确认来源，不能由几何候选代替。恢复期间另一手已知接触丢失可记协调失败诊断，但不据此推出因果。

原1mm、1s、0.5s无进展预算、20ms pressure/recovery稳定及reserve→原SDK提交返回→commit边界均委托既有ledger，不新增预算或pressure servo。接触/恢复flicker不补预算，20ms STABLE不能冒充双侧持续0.2s、原30mm/5s/0.5mm成功或完整恢复准入。本wrapper不提交SDK目标，不增加Actor输入。

标签和事件绑定来源SHA与run/env/episode UID/reset/policy/World/SDK物理步和时间。所有校验先于ledger mutation；selected真实reset只清该环境，选中批次必须共享实际全局帧/时间/policy，未选兄弟不要求同tick。首terminal保持，后续closed/no-op不覆盖它。新wrapper未接入Native，全部原生/硬件/规划/成功资格为false。

测试实际经历缺模块RED→GREEN、CONTACT_CANDIDATE正式命名及来源/跨episode/未知/terminal/预算等边界；独审selected-reset全局时钟反例先RED再修复，拒绝原子性和兄弟预算保持被直接断言。最终27项在下述source15真实冻结CPU证书中全部通过，同次旧ledger37与旧恢复Runtime15也通过，属于CPU兼容证据。

## 后续门槛

实际控制接线需沿原提交顺序使用旧ledger授权，单实例原生核真实源/时钟/双手接触、恢复期间对侧丢失和budget记录，再验证双侧0.2s与任务门槛。未知measurement/slip/opposition不补造；不能用状态标签或既有B的结果替代新wrapper验证。

## 真实冻结CPU证据

证书对应独立immutable `stage7_command_constraints_cpu15`，55个显式test文件，**1360 passed、3 CUDA-unavailable skips、238 subtests，pytest 258.05s、完整入口wall 261.466s，exit0，Native0**。本文件的27项全部通过，是上述整组的子集；H8与contact两份报告共用这一个证书，不能相加为两次独立完整验证。其它尚未发布/接线的算术模块也在这份组合中，不归成本提交新增实现。

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
