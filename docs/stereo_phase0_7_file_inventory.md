# Stereo 运行链路最新文件清单

本轮相对已审查 HEAD `9bfee426363527190011dbca164c104a00cbc999`，五目录共 **40** 项变动；相对原 `11-10` base `6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b` 的累计五目录 **199** 项。README/CHANGELOG 另计；artifacts、原始日志、图像、模型不列入发布范围。

本轮1030/1030通过（Windows/server；分别1/2跳过）。只有短时未训练Actor执行与有界Teacher更新，没有物理成功资格。旧528保护文件、原控制与成功标准保留；所有历史统计保留在下文。

|相对审查HEAD|文件|
|---|---|
|added|[configs/dh116_training/stereo_native_deformable_rod_v1.json](../configs/dh116_training/stereo_native_deformable_rod_v1.json)|
|modified|[configs/dh116_training/stereo_native_task_v1.json](../configs/dh116_training/stereo_native_task_v1.json)|
|added|[configs/dh116_training/stereo_student_estimated_task_state_palms640_v3.json](../configs/dh116_training/stereo_student_estimated_task_state_palms640_v3.json)|
|added|[configs/dh116_training/stereo_student_hybrid_palms640_v3.json](../configs/dh116_training/stereo_student_hybrid_palms640_v3.json)|
|added|[configs/dh116_training/stereo_student_latent_palms640_v3.json](../configs/dh116_training/stereo_student_latent_palms640_v3.json)|
|modified|[docs/contracts/stereo_phase0_7_readiness.json](../docs/contracts/stereo_phase0_7_readiness.json)|
|added|[docs/contracts/stereo_runtime_repair_validation_summary.json](../docs/contracts/stereo_runtime_repair_validation_summary.json)|
|modified|[docs/stereo_phase0_7_file_inventory.md](../docs/stereo_phase0_7_file_inventory.md)|
|added|[docs/superpowers/plans/2026-11-11-stereo-runtime-repair.md](../docs/superpowers/plans/2026-11-11-stereo-runtime-repair.md)|
|added|[docs/superpowers/reports/2026-11-11-stereo-runtime-repair.md](../docs/superpowers/reports/2026-11-11-stereo-runtime-repair.md)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_privileged_ppo.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_ppo.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_checkpoint.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_checkpoint.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_inference.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_inference.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_matching_costs.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_matching_costs.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_native.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_native.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_observation.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_observation.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_perception.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_perception.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_preprocessing.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_preprocessing.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_pyramid.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_pyramid.py)|
|added|[src/go2_pvcnn/control/dh116/tabletop_stereo_rod_state.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_rod_state.py)|
|added|[src/go2_pvcnn/control/dh116/tabletop_stereo_rollout_storage.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_rollout_storage.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_runtime.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_runtime.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py)|
|modified|[src/go2_pvcnn/control/dh116/tabletop_stereo_student_ppo.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_student_ppo.py)|
|added|[src/go2_pvcnn/control/dh116/tabletop_stereo_task_state_adapter.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_task_state_adapter.py)|
|added|[src/go2_pvcnn/control/dh116/tabletop_stereo_wire_tracking.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_wire_tracking.py)|
|added|[tests/test_dh116_stereo_runtime_actor_modes.py](../tests/test_dh116_stereo_runtime_actor_modes.py)|
|added|[tests/test_dh116_stereo_runtime_integration.py](../tests/test_dh116_stereo_runtime_integration.py)|
|added|[tests/test_dh116_stereo_runtime_palms.py](../tests/test_dh116_stereo_runtime_palms.py)|
|added|[tests/test_dh116_stereo_runtime_pyramid_modes.py](../tests/test_dh116_stereo_runtime_pyramid_modes.py)|
|added|[tests/test_dh116_stereo_runtime_rod_identity.py](../tests/test_dh116_stereo_runtime_rod_identity.py)|
|added|[tests/test_dh116_stereo_runtime_rollout_storage.py](../tests/test_dh116_stereo_runtime_rollout_storage.py)|
|added|[tests/test_dh116_stereo_runtime_task_state.py](../tests/test_dh116_stereo_runtime_task_state.py)|
|added|[tests/test_dh116_stereo_runtime_teacher_identity.py](../tests/test_dh116_stereo_runtime_teacher_identity.py)|
|added|[tests/test_dh116_stereo_runtime_wire_tracking.py](../tests/test_dh116_stereo_runtime_wire_tracking.py)|
|modified|[tests/test_dh116_stereo_student_ppo.py](../tests/test_dh116_stereo_student_ppo.py)|

## 历史清单（保留当时状态）

# Stereo Phase0～7 文件清单与验收状态

> 最新J2-B五目录总数为179项；当前CPU Windows885/1skip、server886/1skip。前文141/153/166项为各阶段历史快照，完整最新逐文件清单见readiness与文末增量。

本清单按原base `6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b` 对比当前已提交、未提交与未跟踪文件，范围为 `src/`、`scripts/`、`configs/`、`tests/`、`docs/`；排除 `artifacts/`。根目录 README/CHANGELOG 由本次发布单独维护，不计入五目录统计。新增文件即使在HEAD后继续修改，相对原base仍记为新增。

实现、CPU验证、原生采集、物理资格是四个不同状态。当前未开始Teacher/Student训练，未选定最终resolution，未完成新Teacher或Student物理任务验收。机器账本见 [readiness JSON](contracts/stereo_phase0_7_readiness.json)，[集成报告](superpowers/reports/2026-11-10-stereo-phase0-7-integration.md)和[原生结果](superpowers/reports/2026-11-10-stereo-native-resolution-final.md)保留详细命令、实测指标、失败及限制。

|状态轴|本次实际结果|
|---|---|
|implemented|Phase0～7独立软件路径已落地；native measurement factories、独立geometry gate和depth辅助decoder仍有明确缺口，不能称整个抓取工程完成。|
|CPU|Windows 564 passed / 1 skipped，75.44s；服务器565 passed / 1 skipped，67.68s；均exit0，无Isaac imports。两平台计数不能相加。|
|native_capture|10次独立相机配置采集，全部measurements_verified=True；每次2环境、4计量帧、2预热帧、480Hz/60Hz，执行零24D动作。|
|physical_qualified|10次accuracy qualification均False；selected_resolution=None。Actor延迟、完整训练吞吐及Teacher/Student每抓法100独立初态验收未测。未测不写成0/100。|

原528份生产源码/脚本/配置raw SHA不变；`11-10`仍为原base，旧224配置未覆盖。历史Checkpoint、控制器、物理参数与30mm/5s/0.5mm成功判据不改。force/support资格、R1/R2 native工厂与独立geometry资格、真实成功示范、正式BC/DAgger/PPO及原生Student/部署验收均保持待完成。

## 相对base汇总

|类别|新增|修改|删除|合计|
|---|---:|---:|---:|---:|
|src|43|0|0|43|
|scripts|16|0|0|16|
|configs|26|0|0|26|
|tests|31|0|0|31|
|docs|25|0|0|25|
|总计|141|0|0|141|

## src：生产新增模块

- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_checkpoint.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_checkpoint.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_dataset.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_dataset.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_evaluation.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_evaluation.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_ppo.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_ppo.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_reward.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_reward.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_shaping.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_shaping.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_privileged_teacher.py](../src/go2_pvcnn/control/dh116/tabletop_privileged_teacher.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_camera_cli.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_camera_cli.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_checkpoint.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_checkpoint.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_contract.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_contract.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_dagger.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_dagger.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_dagger_dataset.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_dagger_dataset.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_dagger_trainer.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_dagger_trainer.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_evaluation.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_evaluation.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_framing.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_framing.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_geometry.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_geometry.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_inference.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_inference.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_losses.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_losses.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_native.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_native.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_observation.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_observation.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_perception.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_perception.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_preprocessing.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_preprocessing.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_promotion.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_promotion.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_reference.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_reference.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_resolution.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_resolution.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_resolution_benchmark.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_resolution_benchmark.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_rig_experiments.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_rig_experiments.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_runtime.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_runtime.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_student_ppo.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_student_ppo.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_teacher_collector.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_teacher_collector.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_temporal.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_temporal.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_training_lineage.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_training_lineage.py)
- 新增 [src/go2_pvcnn/control/dh116/tabletop_stereo_validation.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_validation.py)

## scripts：独立脚本入口

- 新增 [scripts/admit_dh116_stereo_candidates.py](../scripts/admit_dh116_stereo_candidates.py)
- 新增 [scripts/benchmark_dh116_stereo_resolution.py](../scripts/benchmark_dh116_stereo_resolution.py)
- 新增 [scripts/collect_dh116_privileged_expert.py](../scripts/collect_dh116_privileged_expert.py)
- 新增 [scripts/collect_dh116_stereo_teacher.py](../scripts/collect_dh116_stereo_teacher.py)
- 新增 [scripts/evaluate_dh116_privileged_teacher.py](../scripts/evaluate_dh116_privileged_teacher.py)
- 新增 [scripts/evaluate_dh116_stereo_depth.py](../scripts/evaluate_dh116_stereo_depth.py)
- 新增 [scripts/evaluate_dh116_stereo_trials.py](../scripts/evaluate_dh116_stereo_trials.py)
- 新增 [scripts/export_dh116_stereo_student.py](../scripts/export_dh116_stereo_student.py)
- 新增 [scripts/infer_dh116_stereo_rgb.py](../scripts/infer_dh116_stereo_rgb.py)
- 新增 [scripts/probe_dh116_stereo_camera.py](../scripts/probe_dh116_stereo_camera.py)
- 新增 [scripts/register_dh116_stereo_training_lineage.py](../scripts/register_dh116_stereo_training_lineage.py)
- 新增 [scripts/run_dh116_stereo_dagger.py](../scripts/run_dh116_stereo_dagger.py)
- 新增 [scripts/run_dh116_stereo_student.py](../scripts/run_dh116_stereo_student.py)
- 新增 [scripts/train_dh116_privileged_teacher.py](../scripts/train_dh116_privileged_teacher.py)
- 新增 [scripts/train_dh116_stereo_bc.py](../scripts/train_dh116_stereo_bc.py)
- 新增 [scripts/train_dh116_stereo_student.py](../scripts/train_dh116_stereo_student.py)

## configs：版本化配置与独立reset副本

- 新增 [configs/dh116_training/privileged_teacher_ppo_v1.json](../configs/dh116_training/privileged_teacher_ppo_v1.json)
- 新增 [configs/dh116_training/privileged_teacher_reward_r0.json](../configs/dh116_training/privileged_teacher_reward_r0.json)
- 新增 [configs/dh116_training/privileged_teacher_reward_r1.json](../configs/dh116_training/privileged_teacher_reward_r1.json)
- 新增 [configs/dh116_training/privileged_teacher_reward_r2.json](../configs/dh116_training/privileged_teacher_reward_r2.json)
- 新增 [configs/dh116_training/privileged_teacher_shaping_candidates_v1.json](../configs/dh116_training/privileged_teacher_shaping_candidates_v1.json)
- 新增 [configs/dh116_training/stereo_bc_v1.json](../configs/dh116_training/stereo_bc_v1.json)
- 新增 [configs/dh116_training/stereo_camera_320x240_v2.json](../configs/dh116_training/stereo_camera_320x240_v2.json)
- 新增 [configs/dh116_training/stereo_camera_640x360_v2.json](../configs/dh116_training/stereo_camera_640x360_v2.json)
- 新增 [configs/dh116_training/stereo_camera_640x480_v2.json](../configs/dh116_training/stereo_camera_640x480_v2.json)
- 新增 [configs/dh116_training/stereo_camera_front_224x96_v3.json](../configs/dh116_training/stereo_camera_front_224x96_v3.json)
- 新增 [configs/dh116_training/stereo_camera_front_320x240_v3.json](../configs/dh116_training/stereo_camera_front_320x240_v3.json)
- 新增 [configs/dh116_training/stereo_camera_front_640x360_v3.json](../configs/dh116_training/stereo_camera_front_640x360_v3.json)
- 新增 [configs/dh116_training/stereo_camera_front_640x480_v3.json](../configs/dh116_training/stereo_camera_front_640x480_v3.json)
- 新增 [configs/dh116_training/stereo_camera_front_roll20_640x480_block5_experiment_v1.json](../configs/dh116_training/stereo_camera_front_roll20_640x480_block5_experiment_v1.json)
- 新增 [configs/dh116_training/stereo_camera_front_roll20_640x480_block9_experiment_v1.json](../configs/dh116_training/stereo_camera_front_roll20_640x480_block9_experiment_v1.json)
- 新增 [configs/dh116_training/stereo_camera_v1.json](../configs/dh116_training/stereo_camera_v1.json)
- 新增 [configs/dh116_training/stereo_dagger_v1.json](../configs/dh116_training/stereo_dagger_v1.json)
- 新增 [configs/dh116_training/stereo_native_task_v1.json](../configs/dh116_training/stereo_native_task_v1.json)
- 新增 [configs/dh116_training/stereo_network_v1.json](../configs/dh116_training/stereo_network_v1.json)
- 新增 [configs/dh116_training/stereo_resolution_experiment_v1.json](../configs/dh116_training/stereo_resolution_experiment_v1.json)
- 新增 [configs/dh116_training/stereo_state_estimator_v1.json](../configs/dh116_training/stereo_state_estimator_v1.json)
- 新增 [configs/dh116_training/stereo_student_ppo_accuracy640_v2.json](../configs/dh116_training/stereo_student_ppo_accuracy640_v2.json)
- 新增 [configs/dh116_training/stereo_student_ppo_v1.json](../configs/dh116_training/stereo_student_ppo_v1.json)
- 新增 [configs/dh116_training/stereo_tripod_reset_20261105_v1.json](../configs/dh116_training/stereo_tripod_reset_20261105_v1.json)
- 新增 [configs/dh116_training/stereo_validation_v1.json](../configs/dh116_training/stereo_validation_v1.json)
- 新增 [configs/stereo_trial_evaluation_v1.json](../configs/stereo_trial_evaluation_v1.json)

## tests：CPU审计与回归

- 新增 [tests/phase0_state_contract_fixture.py](../tests/phase0_state_contract_fixture.py)
- 新增 [tests/test_dh116_privileged_cli.py](../tests/test_dh116_privileged_cli.py)
- 新增 [tests/test_dh116_privileged_shaping.py](../tests/test_dh116_privileged_shaping.py)
- 新增 [tests/test_dh116_privileged_teacher.py](../tests/test_dh116_privileged_teacher.py)
- 新增 [tests/test_dh116_stereo_bc_mode.py](../tests/test_dh116_stereo_bc_mode.py)
- 新增 [tests/test_dh116_stereo_camera_cli.py](../tests/test_dh116_stereo_camera_cli.py)
- 新增 [tests/test_dh116_stereo_collection.py](../tests/test_dh116_stereo_collection.py)
- 新增 [tests/test_dh116_stereo_dagger.py](../tests/test_dh116_stereo_dagger.py)
- 新增 [tests/test_dh116_stereo_estimator.py](../tests/test_dh116_stereo_estimator.py)
- 新增 [tests/test_dh116_stereo_evaluation.py](../tests/test_dh116_stereo_evaluation.py)
- 新增 [tests/test_dh116_stereo_framing.py](../tests/test_dh116_stereo_framing.py)
- 新增 [tests/test_dh116_stereo_geometry.py](../tests/test_dh116_stereo_geometry.py)
- 新增 [tests/test_dh116_stereo_inference.py](../tests/test_dh116_stereo_inference.py)
- 新增 [tests/test_dh116_stereo_native_bootstrap.py](../tests/test_dh116_stereo_native_bootstrap.py)
- 新增 [tests/test_dh116_stereo_opencv.py](../tests/test_dh116_stereo_opencv.py)
- 新增 [tests/test_dh116_stereo_perception.py](../tests/test_dh116_stereo_perception.py)
- 新增 [tests/test_dh116_stereo_preprocessing.py](../tests/test_dh116_stereo_preprocessing.py)
- 新增 [tests/test_dh116_stereo_resolution.py](../tests/test_dh116_stereo_resolution.py)
- 新增 [tests/test_dh116_stereo_resolution_benchmark.py](../tests/test_dh116_stereo_resolution_benchmark.py)
- 新增 [tests/test_dh116_stereo_rig_experiments.py](../tests/test_dh116_stereo_rig_experiments.py)
- 新增 [tests/test_dh116_stereo_runtime.py](../tests/test_dh116_stereo_runtime.py)
- 新增 [tests/test_dh116_stereo_runtime_contract.py](../tests/test_dh116_stereo_runtime_contract.py)
- 新增 [tests/test_dh116_stereo_student_cli.py](../tests/test_dh116_stereo_student_cli.py)
- 新增 [tests/test_dh116_stereo_student_ppo.py](../tests/test_dh116_stereo_student_ppo.py)
- 新增 [tests/test_dh116_stereo_training_cli.py](../tests/test_dh116_stereo_training_cli.py)
- 新增 [tests/test_dh116_stereo_training_lineage.py](../tests/test_dh116_stereo_training_lineage.py)
- 新增 [tests/test_dh116_stereo_trials.py](../tests/test_dh116_stereo_trials.py)
- 新增 [tests/test_privileged_teacher_reward_audit.py](../tests/test_privileged_teacher_reward_audit.py)
- 新增 [tests/test_stereo_action_contract_audit.py](../tests/test_stereo_action_contract_audit.py)
- 新增 [tests/test_stereo_phase0_source_freeze.py](../tests/test_stereo_phase0_source_freeze.py)
- 新增 [tests/test_stereo_privileged_state_contract_audit.py](../tests/test_stereo_privileged_state_contract_audit.py)

## docs：合同、计划、报告与本账本

- 新增 [docs/contracts/privileged_teacher_reward_contract.json](../docs/contracts/privileged_teacher_reward_contract.json)
- 新增 [docs/contracts/stereo_action_contract.json](../docs/contracts/stereo_action_contract.json)
- 新增 [docs/contracts/stereo_native_resolution_results.json](../docs/contracts/stereo_native_resolution_results.json)
- 新增 [docs/contracts/stereo_phase0_7_readiness.json](../docs/contracts/stereo_phase0_7_readiness.json)
- 新增 [docs/contracts/stereo_phase0_source_manifest.json](../docs/contracts/stereo_phase0_source_manifest.json)
- 新增 [docs/contracts/stereo_privileged_state_contract.json](../docs/contracts/stereo_privileged_state_contract.json)
- 新增 [docs/privileged_teacher_phase1_report.md](../docs/privileged_teacher_phase1_report.md)
- 新增 [docs/privileged_teacher_reward_audit.md](../docs/privileged_teacher_reward_audit.md)
- 新增 [docs/stereo_action_contract.md](../docs/stereo_action_contract.md)
- 新增 [docs/stereo_phase0_7_file_inventory.md](../docs/stereo_phase0_7_file_inventory.md)
- 新增 [docs/stereo_privileged_state_contract.md](../docs/stereo_privileged_state_contract.md)
- 新增 [docs/superpowers/plans/2026-11-10-stereo-privileged-dagger-implementation.md](../docs/superpowers/plans/2026-11-10-stereo-privileged-dagger-implementation.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-bootstrap-snapshot02.md](../docs/superpowers/reports/2026-11-10-stereo-bootstrap-snapshot02.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-cold-trials.md](../docs/superpowers/reports/2026-11-10-stereo-cold-trials.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-native-resolution-final.md](../docs/superpowers/reports/2026-11-10-stereo-native-resolution-final.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase0-7-integration.md](../docs/superpowers/reports/2026-11-10-stereo-phase0-7-integration.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase0-audit.md](../docs/superpowers/reports/2026-11-10-stereo-phase0-audit.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase1-integration.md](../docs/superpowers/reports/2026-11-10-stereo-phase1-integration.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase2.md](../docs/superpowers/reports/2026-11-10-stereo-phase2.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase3.md](../docs/superpowers/reports/2026-11-10-stereo-phase3.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase4.md](../docs/superpowers/reports/2026-11-10-stereo-phase4.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase5.md](../docs/superpowers/reports/2026-11-10-stereo-phase5.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase6.md](../docs/superpowers/reports/2026-11-10-stereo-phase6.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-phase7.md](../docs/superpowers/reports/2026-11-10-stereo-phase7.md)
- 新增 [docs/superpowers/reports/2026-11-10-stereo-resolution-v2.md](../docs/superpowers/reports/2026-11-10-stereo-resolution-v2.md)

## 清单复核方法

```text
git diff --name-status --no-renames 6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b -- src scripts configs tests docs
git ls-files --others --exclude-standard -- src scripts configs tests docs
```

本清单列明包括自身和readiness JSON。processed native指标与artifact地址可公开；原始RGB、深度、日志、模型、SDK源码快照及私有归档不进入此清单或Git index。上传或Draft PR不授予模型、深度或物理任务资格。

## J0/J1 增量（2026-11-10）

前文141项是`7c80878`阶段快照；本轮当前完整五目录清单为153项，机器readiness的file_inventory已同步。以下为J0/J1新增文件，原Phase0～7模块复用。

- [docs/contracts/stereo_j1_error_diagnostics_summary.json](../docs/contracts/stereo_j1_error_diagnostics_summary.json)
- [docs/superpowers/plans/2026-11-10-stereo-j0-j1-diagnostics.md](../docs/superpowers/plans/2026-11-10-stereo-j0-j1-diagnostics.md)
- [docs/superpowers/plans/2026-11-10-stereo-j2-evidence-driven-perception.md](../docs/superpowers/plans/2026-11-10-stereo-j2-evidence-driven-perception.md)
- [docs/superpowers/reports/2026-11-10-stereo-j0-latest-source-audit.md](../docs/superpowers/reports/2026-11-10-stereo-j0-latest-source-audit.md)
- [docs/superpowers/reports/2026-11-10-stereo-j1-error-diagnostics.md](../docs/superpowers/reports/2026-11-10-stereo-j1-error-diagnostics.md)
- [docs/superpowers/reports/2026-11-10-stereo-j1-matcher-rootcause.md](../docs/superpowers/reports/2026-11-10-stereo-j1-matcher-rootcause.md)
- [scripts/analyze_dh116_stereo_errors.py](../scripts/analyze_dh116_stereo_errors.py)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_error_cli.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_error_cli.py)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_error_diagnostics.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_error_diagnostics.py)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_error_geometry.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_error_geometry.py)
- [tests/test_dh116_stereo_error_diagnostics.py](../tests/test_dh116_stereo_error_diagnostics.py)
- [tests/test_dh116_stereo_error_geometry.py](../tests/test_dh116_stereo_error_geometry.py)

修改已有文档：README.md、CHANGELOG.md、本清单与readiness JSON；不修改原matcher、控制器或物理参数。原生新增0，准确率/物理/部署资格继续false；本轮不push。

## J2-A 增量（2026-11-11）

最新五目录清单为166项，readiness包含当前逐文件清单。新增文件：

- [configs/dh116_training/stereo_matcher_five_cost_j2a_failure_v1.json](../configs/dh116_training/stereo_matcher_five_cost_j2a_failure_v1.json)
- [configs/dh116_training/stereo_matcher_reliable_sad_j2a_v1.json](../configs/dh116_training/stereo_matcher_reliable_sad_j2a_v1.json)
- [docs/contracts/stereo_j2a_validation_summary.json](../docs/contracts/stereo_j2a_validation_summary.json)
- [docs/superpowers/plans/2026-11-11-stereo-j2a-matcher-contract.md](../docs/superpowers/plans/2026-11-11-stereo-j2a-matcher-contract.md)
- [docs/superpowers/reports/2026-11-11-stereo-j2a-matcher-contract.md](../docs/superpowers/reports/2026-11-11-stereo-j2a-matcher-contract.md)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_matcher_candidates.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_matcher_candidates.py)
- [tests/stereo_j2a_analytic_fixtures.py](../tests/stereo_j2a_analytic_fixtures.py)
- [tests/stereo_j2a_legacy_oracle.py](../tests/stereo_j2a_legacy_oracle.py)
- [tests/test_dh116_stereo_j2a_analytic.py](../tests/test_dh116_stereo_j2a_analytic.py)
- [tests/test_dh116_stereo_j2a_contract.py](../tests/test_dh116_stereo_j2a_contract.py)
- [tests/test_dh116_stereo_j2a_integration.py](../tests/test_dh116_stereo_j2a_integration.py)
- [tests/test_dh116_stereo_j2a_reliability.py](../tests/test_dh116_stereo_j2a_reliability.py)
- [tests/test_dh116_stereo_j2a_safety.py](../tests/test_dh116_stereo_j2a_safety.py)

修改已有depth/诊断/preprocessing/source绑定和一个旧测试的五Tensor shape枚举；另更新README、CHANGELOG、readiness与本清单。原528保护文件不变；不训练、不新native、不push。软件通过与原生精度/物理资格严格分开。

## J2-B 增量（2026-11-11）

- [configs/dh116_training/stereo_matcher_census_j2b_v1.json](../configs/dh116_training/stereo_matcher_census_j2b_v1.json)
- [configs/dh116_training/stereo_matcher_coarse_to_fine_j2b_v1.json](../configs/dh116_training/stereo_matcher_coarse_to_fine_j2b_v1.json)
- [configs/dh116_training/stereo_matcher_gradient_j2b_v1.json](../configs/dh116_training/stereo_matcher_gradient_j2b_v1.json)
- [configs/dh116_training/stereo_matcher_zncc_j2b_v1.json](../configs/dh116_training/stereo_matcher_zncc_j2b_v1.json)
- [docs/contracts/stereo_j2b_validation_summary.json](../docs/contracts/stereo_j2b_validation_summary.json)
- [docs/superpowers/plans/2026-11-11-stereo-j2b-matching-candidates.md](../docs/superpowers/plans/2026-11-11-stereo-j2b-matching-candidates.md)
- [docs/superpowers/reports/2026-11-11-stereo-j2b-matching-candidates.md](../docs/superpowers/reports/2026-11-11-stereo-j2b-matching-candidates.md)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_matching_costs.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_matching_costs.py)
- [src/go2_pvcnn/control/dh116/tabletop_stereo_pyramid.py](../src/go2_pvcnn/control/dh116/tabletop_stereo_pyramid.py)
- [tests/test_dh116_stereo_j2b_analytic.py](../tests/test_dh116_stereo_j2b_analytic.py)
- [tests/test_dh116_stereo_j2b_costs.py](../tests/test_dh116_stereo_j2b_costs.py)
- [tests/test_dh116_stereo_j2b_integration.py](../tests/test_dh116_stereo_j2b_integration.py)
- [tests/test_dh116_stereo_j2b_pyramid.py](../tests/test_dh116_stereo_j2b_pyramid.py)

另新增matching_costs/pyramid两个生产模块，修改depth/preprocessing/error producer接线及来源绑定；README/CHANGELOG/readiness同步更新。528保护文件不变，无新native/训练/push；软件与解析资格不等于物理或部署资格。
