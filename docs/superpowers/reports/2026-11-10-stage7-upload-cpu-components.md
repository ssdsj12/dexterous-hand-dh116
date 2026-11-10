# Stage 7 上传：CPU组件及验证边界

本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。

本次上传没有启动或修改原生实验/训练，不评估或预判既有B等实验结果，也没有新增Teacher/Student成功。A只提供默认关闭的只读观测接线，尚未在Native启用验证；B/C/D未接新控制器。H8几何、真实物理响应、接触/承托及完整约束仍未校准。CPU数学/夹具通过不等于Native验证、真实抓取成功、实时性能或专家许可。

CPU组合包含此前模块及本次全部候选；本报告只按下方逐次追加的Published scope累计发布范围，不把组合中尚未随该次提交发布的模块提前算作已发布能力。PRIVATE parity、源码ZIP、日志和artifacts不上传。

## 一次性Linux CPU主证据

主证据冻结源清单共1530文件，前后SHA map一致；wall-clock 311.899s。validation SHA `82349d1a7afb546954c87134d0acd9eaf4b107c5b872fe7a4825a422c63e4470`。

| 本地闭合收据（不上传原件） | SHA256 |
| --- | --- |
| `command.json` | `621486dce7224e124b665f6d5e962ca04e6c37ad8733bfb693180559e0fe6485` |
| `pytest.log` | `1bc22c5133d36ed378e946ab4232b3d81386e47659ca32a6f5e641bb2bb2c8d5` |
| `pytest.xml` | `0de4083609fd4be5160459ec5cb59c482b4e5040902169b31002c520f8f42a55` |
| `source_manifest.json` | `351a8b513f7f6308dcf8b2e0e062edabeec302d125a127ba8022029e70b06857` |
| `remote_validation.json` | `65b3df7134b6a26de89564128fbad9ce978a71b82c355a2e21f9f1861264bbb2` |

实际argv/cwd/显式环境来自闭合command.json；本Linux组合不包含PRIVATE parity测试，不以skip排除本机28项失败。

```json
{
  "command": [
    "bash",
    "/tmp/dh116_stage7_upload_cpu17/scripts/python.sh",
    "-B",
    "-m",
    "pytest",
    "-q",
    "-p",
    "pytest_mock",
    "-p",
    "no:cacheprovider",
    "--junitxml=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_upload_cpu17/cpu_validation/pytest.xml",
    "tests/test_bimanual_mpc_state.py",
    "tests/test_bimanual_mpc_measurements.py",
    "tests/test_bimanual_mpc_capture.py",
    "tests/test_tripod_native_evidence.py",
    "tests/test_tripod_raw_contact_capture.py",
    "tests/test_native_sensor_batch.py",
    "tests/test_vision_teacher_collection.py",
    "tests/test_bimanual_mpc_scene_measurements.py",
    "tests/test_bimanual_mpc_contact_provenance.py",
    "tests/test_bimanual_mpc_aux_capture.py",
    "tests/test_bimanual_mpc_aux_integration.py",
    "tests/test_bimanual_mpc_articulation_measurements.py",
    "tests/test_bimanual_mpc_contact_delivery_probe.py",
    "tests/test_bimanual_mpc_model_capture.py",
    "tests/test_bimanual_mpc_objective.py",
    "tests/test_bimanual_mpc_model_integration.py",
    "tests/test_bimanual_mpc_experimental_options.py",
    "tests/test_bimanual_mpc_delivery_integration.py",
    "tests/test_bimanual_mpc_filtered_contact_provenance.py",
    "tests/test_bimanual_mpc_filtered_contact_measurements.py",
    "tests/test_bimanual_mpc_filtered_contact_integration.py",
    "tests/test_bimanual_mpc_contact_epochs.py",
    "tests/test_bimanual_mpc_filtered_contact_values.py",
    "tests/test_bimanual_mpc_contact_epoch_integration.py",
    "tests/test_bimanual_mpc_model_descriptor.py",
    "tests/test_bimanual_mpc_mounted_kinematics.py",
    "tests/test_bimanual_mpc_model_descriptor_integration.py",
    "tests/test_bimanual_mpc_dynamics.py",
    "tests/test_bimanual_contact_runtime.py",
    "tests/test_tripod_contact_recovery.py",
    "tests/test_tripod_recovery_runtime.py",
    "tests/test_bimanual_mpc_command_bridge.py",
    "tests/test_bimanual_mpc_constraints.py",
    "tests/test_bimanual_load_transfer.py",
    "tests/test_vision_command_shadow.py",
    "tests/test_stage6_bilateral_recovery.py",
    "tests/test_stage6_common_pause_cuda_readback.py",
    "tests/test_stage6_tripod_retired_transport.py",
    "tests/test_stage6_tripod_entry.py",
    "tests/test_tripod_stage_interrupt.py",
    "tests/test_tripod_pressure_hold.py",
    "tests/test_tripod_execution_guard.py",
    "tests/test_tripod_candidate_planner.py",
    "tests/test_tripod_nonlinear_geometry.py",
    "tests/test_tripod_native_lifecycle.py",
    "tests/test_vision_teacher_pressure_mpc.py",
    "tests/test_vision_teacher_constrained_mpc.py",
    "tests/test_geometric_constrained_mpc.py",
    "tests/test_stage6_common_tracking_pause.py",
    "tests/test_stage6_common_pause_runtime.py",
    "tests/test_stage6_tracking_calibration.py",
    "tests/test_full_episode_tracking_experiment.py",
    "tests/test_teacher_pressure_reference.py",
    "tests/test_stage6_tracking_batch.py",
    "tests/test_bimanual_tracking_guard.py",
    "tests/test_bimanual_mpc_passive_pair_values.py",
    "tests/test_bimanual_mpc_passive_pair_integration.py",
    "tests/test_bimanual_mpc_passive_pair_aux_integration.py",
    "tests/test_bimanual_mpc_batch_kinematics.py",
    "tests/test_bimanual_mpc_local_response.py"
  ],
  "cwd": "/tmp/dh116_stage7_upload_cpu17",
  "environment": {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "DH116_PYTHON": "/mnt/mydisk/xk/envs/dh116/bin/python",
    "DH116_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2"
  },
  "cpu_only": true,
  "native_jobs_executed": 0
}
```

## 独立保留的本机失败与审查证据

本机完整61文件尝试：28 failed, 1445 passed, 3 skipped, 5 warnings, 246 subtests passed in 254.24s (0:04:14)；exit1，wall-clock 255.807s，886个.py文件前后未变。validation SHA `8e137a854b301a9a9d79c3c098d5fc195e4f1d48d6f992ef6e2e417ad1a0abba`，原log SHA `b3c6dc6d71c1bdcd44f9708488613b3eb21998249fd02df91e584bbe6a66eef0`。已记录的失败涉及缺少tensordict/pxr及Windows路径/fcntl平台要求；保留全部失败，不通过新增skip隐藏它们。

独立审查组合：53 passed、71 subtests，0.86s；包含PRIVATE旧SOURCE parity的2个合成case。它分别核对原Scene完整raw、Aux packet/terminal和summary键集合；只作为默认关闭兼容性证据，不冒充本机完整61文件通过，也不计入Linux60文件主验证。

本机失败仍是未通过记录，不能仅抽取其passed数当作全组通过。Linux主验证与这些组合口径不同，计数不相加；PRIVATE parity、所用旧SOURCE ZIP和日志不上传。它们只检查默认关闭的字节/键集合与原源码一致，不是Native反力或任务成功证据。

## Published scope A — 三路被动接触观测侧车（默认关闭）

复用现有Wire→桌面/两端支架视图，新增默认关闭的只读采集接线；保留实际路径、每getter时钟/来源和独立normal/friction原值及NA。不创建World/view，不改变原控制或support字段。

正常量先own再读摩擦；所有行的count/start、容量、重叠仍严格检查，仅按真实Wire行选择活跃非reset载荷。保留首terminal、退休代及失败partial，不以空报告、未知摩擦或numeric ID缺失填造反力/资格。

本次提交限定10个公共code/tests和4个文档文件，共14文件；下列代码SHA取自同一次闭合CPU source map。新增Native验证0，不增加成功次数或规划/执行资格。

| 公共code/tests | SHA256 |
| --- | --- |
| `src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_passive_pair_values.py` | `1ed83e023dcb909305038eff4bcfbd332e74cd4b71ebd60a9b59c6ae9d96825c` |
| `src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_scene_measurements.py` | `f3c30558e0b21b61681b75a73837ed4912bec52f175ea8e755a89281c45edd85` |
| `src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_aux_capture.py` | `52adbf9b3179392059800aba3f5516e786f1f22a4be7a336168b6e0cf287dcf5` |
| `scripts/train_dh116_tabletop_gpu_ppo.py` | `4353281ea7ebdb076af472a4e57832789809128fedf245d27c3e99d4d488d644` |
| `src/go2_pvcnn/control/dh116/tabletop_vision_teacher_collection.py` | `75033f127188dff41e3a5d570f8fbfdb33ed8ba26207bc98e4d785199440a733` |
| `tests/test_bimanual_mpc_passive_pair_values.py` | `4a98a831f9e45205407e8a68ca7352e42887b936b42513bf755ec69e2d6dee12` |
| `tests/test_bimanual_mpc_passive_pair_integration.py` | `becf9e0557ac2a81544488596a9572d71c0d5f6e4a884d8d22e7d1daac831f21` |
| `tests/test_bimanual_mpc_passive_pair_aux_integration.py` | `c4bd380fcacb83cdfbda7660f1816e9fbb965ce4bfe21caf06d0897568940968` |
| `tests/test_bimanual_mpc_experimental_options.py` | `2dabb4e56fad22716f7cc598ea4f4d5ddc7c88ce28824eba34be32463b96be47` |
| `tests/test_bimanual_mpc_aux_integration.py` | `5699aab8300db1f7202023a7d870d4f56e8abfc9cb97acb1c9607e85aacd7318` |
