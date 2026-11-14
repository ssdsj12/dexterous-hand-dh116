# R0 保持必要条件与真实节点速度审计（2026-11-14）

## 修改

在已发布的原有效指腹负证据适配器基础上，新增独立 `NecessaryHoldR0Episode` / `NecessaryHoldR0ControlStep`，复用旧R0与适用域结算，不重写控制、物理、Reset、成功或终止算法。

若完整接触来源已经证明当前不满足双手有效对向接触，则该步不可能增加连续保持时间。因此仅将原本缺失的hold_progress贡献标为“不适用”；原始raw仍null、valid仍false，hold/stable均不伪造成测量值。未知接触、可能双侧接触时仍要求原稳定性与滑移来源。此前已发生接触时，滑移历史门控继续锁存，不被本修改清除。

- `tabletop_r0_hold_necessary_domain.py`：独立诊断合同和适用域。
- `flexible_r0_hold_necessary_domain_v1.json`：默认关闭、只读诊断，原30mm/5s/0.5mm不变。
- `replay_dh116_effective_pad_r0.run`增加显式可选reducer；默认仍旧路径。`main`允许独立诊断入口复用原写入、超时和失败保存。
- `replay_dh116_r0_hold_domain.py`：同一冻结原生记录比较，附480Hz实际solver节点速度和原env/episode/policy/physics/timestamp。
- 3个测试文件覆盖未知接触、已知无接触、接触丢失、原不稳定立即终止、预算、缺失值、节点速度单位/身份。

## 结果

继承原生receipt SHA `9525c46f257fb6d5d81f585f71b47c571dcff7a05019b583e3e93c7535ed0bd8`，
账本SHA `072fe9cf5212abdada92c09a0cd2a04a2032a12b98f437d903565c22eceadc5d`。
本轮原生运行0，物理步0，PPO更新0。不是重新执行已消耗预算。

完整回放232测量步、29控制窗，耗时82.633659s。
- 接触unknown保持0；左raw pair0，右2129，全部原区域拒绝。
- hold_progress贡献未定义步：232→0。原始hold raw仍未测量，不补零。
- 剩余instability：232步 /29控制窗来源未知。
- 完整diagnostic reward仍0/29；training reward仍0/29。
- 已知instability=False/True的原语义保留：stable=False仍在首物理步触发终止，不延后、不忽略。
- 正向原生成功、Teacher准入均未取得。

实际solver节点速度峰值 **8.240307mm/s**：
`env_id=0`，`episode_uid=dual-pinch-0dac87bf45a14e70b8f762f0f5bbc694:one`，
`physics_step=152`，`policy_step=18`，`substep=5`，`rod_tick=1800`，
`timestamp_s=0.31666666666666665`，`node_index=0`。
每个物理步最大节点速度的P50/P90/P95为0.035660/3.624262/3.997734mm/s。
232个实际端点均低于旧刚体代码用于比较的0.03m/s；此比较**不赋予柔性稳定性资格**，配置没有采用这个阈值，stable仍null。这里是480Hz端点读回，不是5760Hz全子步速度历史。

## 测试与证据

先写失败测试：新模块缺失；新增可选reducer参数的测试先因KeyError失败。实现后 **90项CPU回归通过**，另外 **5项实际速度来源测试通过**，总95项。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_r0_hold_necessary_domain.py tests/test_effective_pad_r0_reducer_selection.py tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_control_step.py tests/test_effective_pad_r0_replay.py tests/test_r0_motion_source_audit.py

DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/replay_dh116_r0_hold_domain.py --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-hold-domain-20261114/replay01
```

result SHA：`1626dc3c8e791c92f558d094075ea110be0c402f415bc712301a801393606d83`。
progress SHA：`cf95b27215f7995322f7a067972e7c5c30c8906cf2535b14d007c762105b8540`。
90项日志SHA：`a0f9d53d71627b1c4f2077a66a923e79902d33a5100c59eab786c65ad1ed62dd`。
5项日志SHA：`c0964ffc85325b91914057c5a569780d0149da2f1265407164b39ca6cc0f9f12`。
私有证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-hold-domain-20261114/`。
未上传原生账本、Checkpoint或原始模型。

## 剩余工作

1. 区分solver/关节数值健康、运动稳定和完整保持成功，建立有来源的柔性稳定判据，不能将旧刚体stable表达式直接作为失稳终止。
2. 真实运行接入R0 Reset owner与训练配置，新增原生配方只能使用独立冻结身份和对应预算；本包未执行。
3. 训练系数仍保持原null，诊断权重不冒充训练权重。
4. 正有效接触、固定材料锚滑移、无外部支撑仍需真实资格；已知负接触本身不应成为PPO永久前置成功门槛。
5. Pinch/Tripod/Power独立资格与历史模型保持不变；R2、正式BC/DAgger/Student仍关闭。
