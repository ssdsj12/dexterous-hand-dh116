# 原有效指腹负证据接入柔性 R0（2026-11-14）

## 范围与实际修改

基线176dac170ad457d625971470eec6adfffd964422。新增独立只读适配器，不替换旧控制、旧R0、接触判据或训练配置。

- `tabletop_r0_effective_pad_source.py`：`FrozenEffectivePadSource`校验原生receipt、全部artifact、运行时采集源码、完整账本和原生clock。每步重算原指腹atlas、实际SDK link变换及solver pair，不使用日志派生判定作权威。`EffectivePadR0Diagnostic._measurements`仅替换左右有效接触来源。
- `negative_side_result`：完整来源中全部hand pair都不满足原有效区域必要条件时，输出已知零有效pad与opposed=False；任一候选满足区域条件仍为unknown，不推导深度、滑移或有效承托。
- `replay_dh116_effective_pad_r0.py`：复用原Reset、371D/24D合同、PhysicalR0Diagnostic和ApplicableR0ControlStep，对同一232个物理测量步并排比较legacy与新来源。Approach使用actual reset/SDK/solver节点及锁定SDF。
- 独立opt-in配置、两个回归文件。原R0所有系数保持null；回放沿用明确标为CPU fixture的ordinal诊断权重。

## 原始证据与完整性限制

原生receipt SHA：
`9525c46f257fb6d5d81f585f71b47c571dcff7a05019b583e3e93c7535ed0bd8`。
原始physical.jsonl SHA：
`072fe9cf5212abdada92c09a0cd2a04a2032a12b98f437d903565c22eceadc5d`。
该历史原生执行2初始化+232测量，154.725142s，自然exit0。本工作包新增原生运行0、物理步0、PPO更新0。

原始序列化没有保留raw capacity/overflow计数，不能将任意空pair数组认作完整测量。新来源要求冻结receipt+完整artifact，核对原生`tested_head`中的runtime/validate_pair_capture源码字节与现有源码相同，并要求`pair_capture_status=validated_solver_samples`、contact_valid、同一步solver聚合力与逐pair核算一致。该依据是历史producer当时完成的非overflow校验，不冒称新读回了已丢弃的原始计数。任何修改的exchange、foreignclock、pair缺失或side绑定均被拒绝。未来live来源仍需独立资格。

## 同源回放结果

| 指标 | legacy | 原区域负证据来源 |
|---|---:|---:|
| 完整物理测量步 / 控制窗 |232 /29|232 /29|
| 接触unknown物理步 |84|0|
| required_contact缺失步 |84|0|
| contact_persistence缺失步 |84|0|
| grasp_geometry缺失步 |84|0|
| lift_progress缺失步 |84|0|
| slip贡献未定义步 |84|0|
| hold_progress缺失步 |232|232|
| instability缺失步 |232|232|
| 完整diagnostic reward控制窗 |0/29|0/29|
| training reward控制窗 |0/29|0/29|

左侧raw pair=0，右侧2129，原区域接受=0。原固定锚滑移raw仍为null/valid=false；只是沿用已有“已知尚未建立双侧接触”的适用域，使该项贡献不适用，而非把滑移测成0。候选或历史unknown时不能沿用这个负证据结论。

最终回放78.704350s。result SHA `23fd16e8a6608893a4c192e366c93f298cd48e0c4b88ac4be67c132da48e1a09`；逐步对照SHA `c2af730e34a125b287c54815ece124de80dc9cbb78fd5c405e945c1a5346e814`。
私有证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/effective-r0-negative-20261114/`，未上传原始账本、模型或原生数据。

## 测试与运行命令

先写失败测试，缺失模块导致collection失败；随后实现和验证。
CPU回归 **117 passed**（39.06s）。初始18项测试也通过。Red日志SHA `245783a34bb7d3b0e1414124edd139432965f90c82f6c2b61cc144beb9c1c15e`；117项日志SHA `0424d1dd047925acb2531943fa8b513d629ecaece92f0214604ef9f2e4ddd45f`。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_r0_effective_pad_negative_source.py tests/test_effective_pad_r0_replay.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_r0_reward_domain.py tests/test_dh116_r0_reset_approach.py tests/test_deformable_pad_region.py tests/test_dh116_deformx_pair_capture.py tests/test_deformable_r0_replay.py

DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/replay_dh116_effective_pad_r0.py --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/effective-r0-negative-20261114/replay02
```

CPU fixture、原生历史只读回放、原生新执行明确分开。本轮未启动Isaac。

## 未解决与下一步

Teacher仍blocked：stable没有实测判据来源，导致hold_progress/instability各29个控制窗未定义；原生R0 Reset owner顺序和训练系数尚未资格化。371D原始合同仅66维已测，其余305维保持NaN/false；既有独立measured-common模型入口保留。

原`tabletop_tensor_task.py`的stable混合高度、速度和滑移，而柔性R0又把stable=False用作instability终止，不能直接移植，否则未抓起就会终止。下一包明确区分保持必要条件、物理健康与运动稳定来源，保持30mm/5s/0.5mm、原关节/碰撞保护，不因缺少正成功而阻断可学习的已知负奖励，也不填造未知。

正有效对向接触、材料滑移、完整双手抓取仍未通过。当前修改没有Teacher训练或抓取成功资格，R2、正式BC/DAgger/Student保持关闭。Tripod/Power使用自身可达性证据，不复制Pinch资格。预算与历史原生配方不重跑。
