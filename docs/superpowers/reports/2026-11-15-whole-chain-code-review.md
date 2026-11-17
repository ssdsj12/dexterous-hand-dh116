# DH116 Teacher／视觉／Student 执行链审查

日期：2026-11-15 UTC
基线：`74e9404aa207be0311de7f963dbd6bb49499db5f`
范围：当前执行链及未集成原生入口旁路审查；不是对全部运行行为的穷尽证明。未修改生产源码、未运行新 native、未训练模型。

## 基线和验证

Ubuntu integration、Windows relay、GitHub HEAD一致，PR #2 Draft。integration干净。1913个tracked Python文件通过当前Python3.10 AST解析，0语法错误。`git diff --check`通过。528保护文件和两个历史Checkpoint验证通过。

正确启动方式：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q \
 tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py \
 tests/test_dh116_stereo_wire_geometry.py tests/test_dh116_stereo_wire_tail_audit.py \
 tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_depth_auxiliary.py \
 tests/test_dh116_stereo_runtime_wire_tracking.py tests/test_dh116_stereo_estimator.py \
 tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_state_multicohort.py \
 tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_bc_mode.py \
 tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_student_ppo.py \
 tests/test_dh116_stereo_inference.py tests/test_dh116_privileged_teacher.py tests/test_dh116_teacher_wp3.py
```
实际结果：**262 passed，48.54秒，exit0**。私有日志SHA：`1b2e12790875ab576a68153daa114f580ac40ff861f4ec32886854b213ee71be`。

## 确认缺陷

| ID | 位置 | 实际证据 |
|---|---|---|
| R01 P1 | tabletop_stereo_bc.py:63 | latent BC成功；estimated_task_state报1x32与3x128不匹配，hybrid报1x32与35x128；三模式独立推理均成功。DAgger复用该BC入口。 |
| R02 P1 | tabletop_stereo_losses.py:177–191 | 第149分类维target0/logit0给loss0、grad0，部署sigmoid=.5；BCE应.693147。20个分类维有同类数学分叉。独立状态训练入口已用BCE。 |
| R03 P1 | tabletop_privileged_dataset.py:52–118 | 验源SHA与episode资格，未重放join observations/actions；隔离上层测试在固定proof下接受替换张量。 |
| R04 P1 | tabletop_stereo_supervision_dataset.py:60–132 | 固定proof/identity下改RGB和Teacher/issued action仍通过上层数据校验；数据集自身hash不能证明来自raw源。 |
| R05 P1 | tabletop_stereo_student_cli.py:291–301 | 仅检查calibration；collection入口34–47还比depth matcher。根代理内存调用确认Student正式准入没有读所选matcher。 |
| R06 P1 | scripts/dh116_pinch_acquisition_v2.py:444–466 | clear_sim异常后events=[clear,close]，STOP和sim=None跳过。不能把风险清理当正常释放。 |
| R07 P2 | tabletop_stereo_camera.py:66–67,96–97 | Gf真实投影：640x480/fx400/cx350/cy260，光轴实际落290.000002/219.999999；作者与回读错误互相抵消。中心默认不触发。 |
| R08 P1 | scripts/run_dh116_mesh_reset_numeric_pinch_work_packages.py:28 | 重绑定entry后entry.prior.sha不存在；保留原失败receipt和CPU包装器复现均证明封存层错误。 |

R03/R04/R05的隔离测试假定下层原生资格已经合格；它们定位上层绑定缺口，没有证明当前真实flexible资格可被绕过。没有合格来源的数据仍不能进入成功池。

## 可复用部分及尚未完成项

PPO旧likelihood重验、保存latent/action/mu/std、Tanh动作和GAE公式未发现本次新错误。新Action24HistoryOwner可复用实际request来源，不能重写PPO来掩盖入口缺失。

实际Teacher仍有完整链路缺口：
- 生产privileged CLI仍走旧rigid入口；actual no-camera collector工厂固定拒绝构造。
- 当前backend为881701系列，旧adapter仍绑定2fea系列，必须新合同而非改旧receipt。
- 正式flexible R0系数为null，training_reward=None。
- measured66+rod保留其他305维未知，不能自动授完整371D资格。

左右RGB共tiled render product；Reference Depth/ID有独立同帧核查。已审实际CameraView映射为Optical-Z，没有证实径向距离混用。原色基线、分池和隐藏区域保护保留。

CPU调度依赖失效未传递到历史passed CPU包，但原生规划/发布另有递归receipt复验；列状态一致性风险，不声称原生绕闸。timeout当前是资格截止，child.wait可继续等待，不是硬执行时限。

中断前直接Python测试缺pxr，包括未提交Tripod/provider和保留账本新复核，属于启动环境缺USD路径；正确scripts/python.sh会设置该路径。这些失败保留，未把它们算成生产修复或原生通过。旧任务未提交文件保持原样。

## 修正路线

采用用户已选择的“现有执行链分包修复”。设计见：
[设计规范](../specs/2026-11-15-dh116-execution-chain-repair-design.md)。

本包仅审查与设计：`source_edits=0`、`new_native_runs=0`、`optimizer_updates=0`、`teacher_admission=false`、`student_admission=false`。设计仍待用户审阅。下一步是逐包实施计划，不是把这262个历史回归写成上述缺陷已修复。
