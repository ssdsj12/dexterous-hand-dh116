# Cold-trial独立评估接线附录

状态：已实现与CPU验证；native execution=0，不授Teacher、Student、force或support资格。解决的是固定checkpoint在相同原始physics下评估100+/grasp新冷初态的接线，未改checkpoint合同、原物理控制或成功判据。

## 固定policy与原physics

新增`tabletop_privileged_evaluation.py`、`scripts/evaluate_dh116_stereo_trials.py`、`configs/stereo_trial_evaluation_v1.json`及`tests/test_dh116_stereo_trials.py`。Teacher首先在原base contract上通过既有`privileged_cli._policy_for_evaluation`严格加载一次，之后只切换预先声明的reset seed buffers。没有删掉grasp-init、close-seed、pose或材料等字段来绕过模型身份。

完整manifest由原`validate_manifest`先验证≥100个独立float32 authored状态/抓法，source checkpoint SHA与完整`evaluation_physics_sha256(base_contract)`均匹配；后者保留所有原物理/control/success/init语义。默认N=2，可选2/4/8，100例/抓法分别用100/50/25个平衡小batch，而不是一次建立200个相机环境。

原public reset wrapper对split=validation不允许小于100/style，因此runner不伪造subset.split=test、不放宽旧helper。它在完整manifest验证后，复用wrapper内原有`_install_reset_trials`与`_readback_reset_trials`内核。所有batch还先在仅resetbuffer克隆的staging对象上运行原writer校验，完整coupling、实际native joint limits、zero wrist/root velocity约束都在首次真实reset前检查。

实际每batch顺序为：原writer写声明seed→`bridge.bind_cold_trials`只绑定metadata→`bridge.reset`原物理reset→原actual readback→sensor cache clear→frozen deterministic policy。保存outer manifest SHA、原split、trial indices/IDs、authored state SHA、actual readback SHA与原始全状态。既有2e-5 reset readback容差未改；没有声称恢复settled PhysX solver snapshot。

两style数量必须相等且能被N/2整除；否则执行前明确拒绝，没有padding克隆、丢trial或按结果选样本。完成后只恢复入口时reset buffers/_per_env_reset_seed，不增加物理reset或step，不改变已记录实际状态。

## Teacher/Student hook

```python
evaluate_cold_trials(
    bridge, policy, manifest, new_output,
    max_control_steps=1200, minimum_per_style=100,
    owner='teacher', checkpoint_path=checkpoint,
)
```

Teacher仅调用`policy.deterministic(bridge.critic.clone())`，不更新参数/optimizer。Student必须使用：

```python
evaluate_cold_trials(
    bridge, actor, manifest, new_output,
    owner='student', checkpoint_path=actor_package,
    observation_callback=lambda: environment.snapshot().sensor_history,
    on_batch_reset=environment.clear_sensor_history,  # 调用时传bridge参数，仅清sensor cache
    step_callback=environment.step,
)
```

Student只收到typed `StereoHistoryBatch`，不收到371D。step_callback是必要的，因为直接bridge.step不会清NativeStereoEnvironment的cached RGB/history。新RED→GREEN回归通过真正step callback更新连续两个tick的RGB/history（均值0→0.1），避免只测试type而实际一直用旧帧。typed StepResult的issued dtype/value、reward与done均核对，Teacher仍走原4tuple bridge.step。

global UID/reset/physics/policy clocks始终不复用；trial first-episode身份由bridge metadata hook记录。每slot只采用该cold reset第一次UID的pre-reset 480Hz terminal。slot已结束后明确执行diagnostic_zero直到同batch其余slot结束；后续自动reset episode不会替代原trial。done却没有对应原UID terminal raw evidence时直接报错。超出budget为incomplete，不能变成success。

输出只记录本次观察与unqualified raw source；force/support仍None。policy tensor摘要和原base contract/input files在前后核对，确保模型冻结及输入readonly。

## Manifest冻结与显式policy derivation

`freeze_teacher_cold_manifest(checkpoint, base_contract, primary_initialization, secondary_initialization, output, seed=...)`复用原adaptive/template/generate helper，检查primary文件SHA属于原base初始化，两grasp source_contract一致，保存全部模板/模型来源SHA及完整base contract。生成的状态继续是authored_unverified；实际policy还必须走原严格loader，生成manifest不等于checkpoint可用或原生成功。

Teacher与Student的模型SHA不同，需要显式：

```python
derive_policy_trial_manifest(source_manifest, old_policy_artifact,
                             new_policy_artifact, new_manifest_path)
```

仅新建manifest的policy-source binding，所有trial IDs、physical states与state SHA完全保留。来源sidecar绑定原manifest、两个policy文件SHA、相同物理cohort与physics fingerprint。原manifest与两个模型都readonly；不是checkpoint转换，不静默替换元数据。两角色可据此在相同物理cohort上分别评估。

## CLI与输出

默认`--check-config`或未传`--execute-native`仅校验/打印配置，不创建output、不导入Isaac、不启动训练。Native必须同时显式`--enable-experiment --execute-native`，提供manifest/checkpoint/new output；Teacher与Student输入来源分别处理，原训练源码不改。

每run exclusive目录；每batch独立reset_readback.json、全局control_intervals.jsonl与trial_evaluation.json绑定原始native_physics.jsonl路径。CPU fixture固定schema_only/source_scope=cpu_fixture、native_successes=None/native_qualified=False；没有把CPU终止事件当Native成功。

## CPU结果与K数值回归

初始RED为10个missing module/CLI失败；首GREEN后补齐policy derivation、Teacher manifest冻结、Student连续history推进。最终cold12项通过。另对VGA相机float32 focal/aperture roundtrip单独RED复现：K约1e-5以上的数值残差误触旧统一绝对阈值。

加固后只对K使用原atol1e-5px加`rtol=2*float32.eps`，对应SDK lens字段舍入贡献，并在packet输出intrinsic/translation/rotation残差。T继续原atol1e-5、rtol0；额外2e-5m仍拒，K误差0.01px仍拒。这不是修改任何task/actuator/guard或放宽原生精度晋级阈值。

完整最终命令：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_trials.py tests/test_dh116_stereo_resolution.py tests/test_dh116_stereo_geometry.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py
```

本次115 passed in12.66s，exit0，无warnings。原30个action/control/training源文件逐字节SHA不变；旧224配置仍为`f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`。所有fixture只验证软件/算术/接线；实际Linux Python3.10/Torch2.7与原生短probe由后续独立流程执行，本附录不预报其通过。


本次关键文件SHA256：

|path|SHA256|
|---|---|
|`src/go2_pvcnn/control/dh116/tabletop_privileged_evaluation.py`|`7a8f4990da3ea30472ef2e505ed7cf4fd253a27c92ec4f74a04194b6e52cf826`|
|`scripts/evaluate_dh116_stereo_trials.py`|`a715f4b32213afef97d10c226140344f2d4ac80a2f1c045165c8286199f8d8aa`|
|`tests/test_dh116_stereo_trials.py`|`8a55dc8564e8029bfe420d93c7730429411bebea0748fdcf5801b95f41ffcaf3`|
|`configs/stereo_trial_evaluation_v1.json`|`ff0de9a9dd9cd44ec67d3eaf1be1a37dc935318249bd904422ca864e5d932af7`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py`|`e0db2dfb0e0a9737e0821b015de96888a32377d0ac0c80ad927c01b6511d2664`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py`|`10bd3a95725bc30d100a9f8eafac4938b0b0477e78454753d94ebd36edb7ef1c`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_resolution.py`|`cefac066fc212017716a4aea96369c150e686e3fb3fbdeed815bc645027ff15a`|
|`tests/test_dh116_stereo_resolution.py`|`2ecc1c271fcfba8b1bc0bb35fd610a12eacaf7fe0f8a4fe4e097a0fcfcd1c595`|
