# R0 Reset 实测接近基准 — 2026-11-13

基线 `0753bcdefac84b8c5bf1ce5a37836fc4aad8382c`。本包是已有原生数据的 CPU 重放；新增物理步0、PPO更新0，无新 Isaac 实例。

## 实际修复与来源

`MeasuredPinchApproach.capture_reset` 读取原始 Reset 账本中 PhysX `get_link_transforms` 得到的双手23-link位姿、完整17DOF q/qd、实际SDK目标，以及同钟41个杆节点。原Reset owner在读取前调用 `update_articulations_kinematic()` 刷新运动学，并核验没有新增SDK/solver物理步。

Reset并未保存接近距离标量，但保存了计算所需的实际getter数组。上一包“Reset基准缺失”仅对标量成立；本次通过原始记录可重建该基准，不需要借用第一步、不重跑预算、不将 authored FK 当实测。

验证包括：Reset完成状态、SDK/tick/solver时钟、参数包与节点、原seed SHA和实际q/qd/目标/effort、左右独立link路径、23-link/17-DOF类型、单位四元数、所有10个指腹副本、实际rod collider索引、锁定SDF/capture/USD SHA。继续调用相同 sampled-node SDF算法，不改变物理、动作、保护或原奖励公式。原始Reset owner源码未修改。

仅 `--measured-approach --measured-reset-approach` 启用新重放模式。前三种模式保留，16份post-step来源只计算一次并在对照中复用；Reset基准独立绑定。Tripod/Power不能复用此Pinch身份。

## 验证

先失败：`r0-reset-approach-red-6CogJX/pytest.log` 保存缺少 `capture_reset` 的失败。
新增23项测试，覆盖实际来源、18类篡改、原seed、所有权、其他抓法拒绝、端到端回放和缺少来源拒绝。
全量相关回归 **144 passed，0 failed，56.16s**：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_r0_reset_approach.py tests/test_dh116_r0_measured_approach.py tests/test_dh116_native_parameter_r0_replay.py tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_measured_closure_geometry.py tests/test_deformable_r0_replay.py tests/test_dual_hand_reset.py
```

日志及严格JSON：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-reset-approach-regression-0hmEuq/`。
comparison.json SHA：`05c1334fbccacb8300e93abff4f52fd57defb1de726f88d6c2c10d9267f87ec2`。
父receipt SHA：`f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`。
原始physical.jsonl SHA：`1981d68067f6554e63a27b190b9eb55f8663501ebdc9daca4700c73929e4ba2a`。
Native imports为空。重放CLI拒绝覆盖已有文件；原始账本只读。

## 同一原始轨迹结果

Reset位于实际SDK2、rod tick0、timestamp=2/480秒。左接近距离1.833917107mm，右2.160667907mm。
第一步平均差值 **−0.000024214387mm**，保留实际负值，没有填零；本指标是离散节点到指腹SDF的正距离均值，不是Lift、接触或连续最近距离。其误差尚未校准，不能把微小变化当物理显著性。

| 模式 | 控制0有效接近样本 | 控制1有效接近样本 | 控制0剩余阻断 |
|---|---:|---:|---|
| 原R0，实际半径 | 0/8 | 0/8 | approach、hold、slip、instability |
| applicability v2 | 0/8 | 0/8 | approach、hold、instability |
| 加post-step实测几何 | 7/8 | 8/8 | approach、hold、instability |
| 加实际Reset基准 | 8/8 | 8/8 | hold、instability |

控制1接近增量仍为0.010818941519mm，与上一包相同。实际hand–rod pair仍0，table子步/节点pair记录7872。

## 资格与后续

- 总diagnostic reward仍NA；training_reward仍null。这里系数是既有CPU fixture，不是已资格化生产R0权重。
- `r0_native_reset_complete=false`：来源重建不等于实际R0 owner的Reset顺序已经通过原生验证。
- stable没有已定义的测量判据，不能从“未报错”或“无接触”推断稳定。hold/instability继续阻断。
- 继续推进实际对向接触、材料锚/滑移、手桌接触来源、剩余371D字段和生产奖励。Teacher准入0/3。
- 未改528个保护文件、两个历史Checkpoint、原资产/物理/动作合同；PR #2保持Draft。原始数组与实验文件不上传。
