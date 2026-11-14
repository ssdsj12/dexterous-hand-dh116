# 原 flexible R0 的观测缺失与奖励定义域审计

基线 `a4a5af8b2328c1290ef8511adf3146504f73ccbc`。本包只新增纯 CPU 测试与本报告；原 kernel、replay、reducer、task 配置、系数和 30mm / 5s / 0.5mm 成功标准均不改。没有新 native、GPU、Isaac、几何查询或训练更新；integration 未改，未 push。

结论：真实 v4 的 NA 同时包含观测缺失和当前奖励定义域限制。补齐传感来源能让部分不满足条件的项由原公式计算为 0；但首次 approach 差分和没有 bilateral 时的 material slip 仍然被原代码定义为 NA。不能据此宣布物理永久不可用，也不能给未定义项静默补零。

## 已发布真实输入

只读核对已发布 `2026-11-13-dual-v4-r0-native-replay.md`、机器摘要和完整真实重放文件：

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-v4-r0-replay-cpu-gmhrfazr/native-v4-r0.json`

SHA `1489d19c86f7eb19731f7b59c09fd1e6d01e721219fdb8be91bff80941ced5b8`。

其原生 ledger SHA `ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c`，只含历史 2 setup + 8 measured。八行都是 approach：左右完整 pair capture 中没有手 pad 样本，杆—桌面 SDF 接触为 true，实际 joint guard 为 false；radius、stable、approach_distance 及三个外部辅助支持源均没有绑定。新测试没有把 CPU 数据覆盖这些实际未知项，也没有重新执行该原生运行。

## 原公式如何区分未知与已定义的零

准确入口为 `src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_task.py`：`reset` 第308行、`_slip` 第384行、`step` 第416行。phase 在奖励量计算之后才由 holds/bilateral/coverage 派生；名字为 approach/lift/hold 并不代表存在通用的 phase-based zero mask。

| 项 | 原公式与本次原因 | 能否由原公式得到零 |
|---|---|---|
| approach_progress | 第434–441行：没有距离样本则 NA；reset 第328行将 self.approach=None，首个有效样本只有当前值，没有前值 | 相邻两次有效平均距离相同才是0。首样本不是0；中间缺样还会清空下一次差分基准 |
| lift_progress | 第465–474行：eligible_lift 要求 contact_known、support_known 和已知半径形成 min_surface。v4 前两者已知，缺 radius | 有实际半径来源后，当前无bilateral或有桌面支持使best不增加，原式给0；本次radius未知仍必须NA |
| hold_progress | 第475–496行：还要求joint_limit、stable已知，且 not bilateral 或 slip_ok已知。v4 joint已知且bilateral=false，因此此处不需要slip先有效；仍缺radius与stable | 所需输入已知但保持条件不成立时给0。本次不能只因不在hold阶段就绕过未知输入 |
| instability | 第509行：stable未知则NA，否则float(not stable)，没有phase门 | stable=True给0，False给1；无手接触、零动作、joint guard没报错不等于stable=True |
| material_anchor_slip | `_slip` 第387–390行：not bilateral直接返回NA；即使调用者提供字段，原分支仍不建立适用锚 | 空接触不是0滑移。只有适用的双手固定材料锚测量才可能产生真实零漂移 |

已定义的零：完整零pad证据使 required_contact、contact_persistence、grasp_geometry 为0；实际joint guard使joint_limit为0。`PhysicalR0Diagnostic._measurements` 第236–287行刻意只在不足两个原始pad时证明必要对向条件false；至少两个pad的计数仍不能证明有效opposed，不能靠增大接触计数自动取得正向R0资格。

支持量也不能混淆。第452–461行对支持条件做三值OR：存在一个true便已知true，其余未知不妨碍这个结论。v4真实杆—桌面pair使external_support_violation原raw=1；该字段名不表示已经观测到外部人为施力。第462–464行只对external_support/authored_force/attachment的true锁存禁止支持，桌面接触本身不会在approach终止。PhysX hand-table仍是独立NA来源。

`stable` 在当前flexible链中只有带source的输入接口，没有已绑定的原生物理判据。`_sample` 第180–187行拒绝已知值缺少非空source；非空字符串只满足API合同，不验证标签对应真实native测量。replay明确把auxiliary标为caller-supplied labels，不能将测试标签视为资格证据。

## 纯 CPU 参数矩阵

新增 `tests/test_dh116_r0_reward_domain.py` 直接调用未修改的 `FlexibleGraspEpisode.step`，并对三种stable状态调用原 `FlexibleR0ControlStep.push/settle`。样本使用明确的 `cpu_r0_reward_domain_fixture:*` 来源；半径.005仅是已有CPU fixture值，未补写真实v4半径。fixture保持完整空pad、table=true、joint_limit=false，external_support/authored_force/attachment未知。

| CPU radius | CPU stable | lift raw | hold raw | instability raw |
|---|---|---:|---:|---:|
| NA | NA | NA | NA | NA |
| NA | true | NA | NA | 0 |
| NA | false | NA | NA | 1 |
| known | NA | 0 | NA | NA |
| known | true | 0 | 0 | 0 |
| known | false | 0 | 0 | 1 |

六格全部继续验证：phase=approach、无success、无禁止支持锁存；三个接触项和joint项为有效0；桌面支持项为1；材料滑移NA、总奖励NA、training_reward=None、native_qualification=false。每格首approach为NA、下一相同距离为有效0。

三项reducer用例证明stable=false在approach首个物理样本就按原instability终止，而stable=true/unknown均结算完整8样本；unknown不是默认安全。另有两项拒绝已知radius/stable却无source、一项table=false而其他支持未知仍为NA、一项距离序列中丢样后重新建立基准。总计13项新增。没有用仅schema断言替代公式执行。

## PPO准入中不能靠观测自动消失的定义域缺口

`tabletop_deformable_control_step.py:155` 的settle要求窗口内每一项每个样本都有贡献，否则总diagnostic_reward=None；原测试还明确零系数不消除未知raw。新矩阵继续用既有CPU系数1…10只为诊断，断言原task系数全为None且输入task不变，未校准任何生产reward。

已有测试 `test_deformable_r0_replay.py:96` 给完整auxiliary：第二个窗口approach已完整，但仅material_anchor_slip仍阻断总奖励。已有 `test_dh116_deformable_control_step.py:45` 验证首窗口只有7个有效approach差分；第161行验证系数0不能绕过未知。因此：

- 第一窗口缺reset/pre-action距离基准，是当前reset/reward接口定义问题；增加后续物理采样不能补回其第一条差分。
- 无bilateral时slip没有适用固定锚，是原函数明确的定义域分支；即使半径、稳定性、距离来源齐全，原严格reducer仍不能结算完整标量。
- 真正bilateral之后还需共同material reference、连续rest station/uncertainty、固定pad tangential reference及slip/uncertainty；原0.5mm成功判断不能用离散节点编号、空接触或净力替代。
- full371、原生reset/source资格以及生产系数绑定是另外的准入条件，本包没有解决或降级它们。

## 独立后继奖励合同需要明确决定的语义

这些是下一合同的具体决策点，并非本包已经实施或授权的改动：

1. **首个approach基准与采样间断。** 是否在完整reset之后、首action之前取得同episode/rod身份的真实距离基准；首个没有前值的transition应保留NA、划为明确初始化阶段，还是采用另行定义的奖励规则。距离度量本身必须绑定目标点/面、坐标系、左右聚合方式与采样时钟。丢样后能否跨间隔作差也须明确；不能偷用别的episode或重复step建立基准。
2. **滑移适用性与不可观测分开。** 分别规定接触未建立、必要bilateral条件已知false、接触丢失、接触状态未知和已建立固定锚时的奖励语义。若后继合同决定“不适用项贡献为0”，必须保留原物理slip测量为NA以及独立适用性/原因，不能把它写成测得0m。锚建立、保留、接触丢失与重获后的处理必须明确，并保持原成功滑移标准。
3. **stable的物理判据。** 明确被判定对象、所用实测物理量、阈值、时间窗口、失败和丢样规则、来源及clock身份；不能默认等价于joint_limit=false、没有接触或零动作。后继规则还须明确其独立终止作用。
4. **已证条件false时是否仍要求全部输入。** 当前lift/hold严格要求radius/stable等前置已知，即使bilateral已知false。可保持原严格性，也可在独立版本明确安全的条件推导，但不得在旧实现内悄悄短路未知。任何选择都需测试同一输入的NA/0/1区分、失败终止和成功标准不变。
5. **生产系数与总奖励结算。** 单独锁定生产系数来源、物理有效性与适用性规则，决定如何处理初始化或观测缺失的transition。既有1…10 CPU fixture不能成为PPO系数；不能靠权重0、丢弃未知项、重复奖励或改30mm/5s/0.5mm标准取得可训练标量。

本轮没有选择上述后继语义，也没有创建新的reward模块。

## 验证与身份

实际执行 **61 passed / 10.32s**：13新增、48有针对性的旧回归；无失败。包含原grasp task、完整control reducer以及旧完整空pair重放测试。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_r0_reward_domain.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_control_step.py tests/test_deformable_r0_replay.py::test_complete_empty_pairs_never_make_material_slip_zero_or_first_progress_valid
```

私有证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-reward-domain-cpu-n75k3193` 保存准确命令与退出码。tests.log SHA `d0e7dc8d3d6fb6741e1b0bc90c773ba7acdb7bfb954bcb060b5e82369245514f`。

| 文件 | SHA256 |
|---|---|
| 新test_dh116_r0_reward_domain.py | c198d88c76c9d8ce437fe32d3379398e3690ed4ef00d2ece47beb69f829e6fac |
| 原tabletop_deformable_grasp_task.py | 2ddceff33310752a5623813a07aab55a183482c167c8d905a936f63571bd21e5 |
| 原tabletop_deformable_r0_replay.py | 79fdc32b04638672c9e38092294c4b03cad4549e519ce8bc8ea9323a25dfe8d8 |
| 原tabletop_deformable_control_step.py | e97651ecd0d5173cac69f648bb3374e4751b5ebdac31492859714b11ebb67b4f |
| 原flexible_dual_hand_grasp_v1.json | 8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63 |

旧文件已与基线逐字节比较一致，530项保护校验通过。测试只有CPU来源，不能替代真实native参数、stable、材料锚或Teacher资格。

独立physics agent只读审查了13项新增测试，未发现新增阻断；没有代跑测试，也未审查本报告或原生结果。
