# 双手 physical episode 到 R0 的可重放诊断

日期：2026-11-13
基线：`6d8af476d5d19c873ac3a87639a3f43ff9ef89bc`
独立树：`flexible-r0-replay-6d8af47`
范围：CPU 数据接线与磁盘重放；0 native / 0 GPU / 0 optimizer updates。

## 交付与实际边界

新增 `PhysicalR0Diagnostic` 接收原 `DualHandPhysicalEpisode` 的 reset / physical 记录、`pre_action_state()`、原 24D action 与 control 结果。逐物理样本调用现有 `FlexibleR0ControlStep.push`；该 reducer 内部调用原 `FlexibleGraspEpisode.step`，每条只调用一次。结算复用原 `settle`；为保存逐物理 R0 行，只读取 reducer 已产生的私有 `_rows[-1]` 副本，不重复计算 reward。这个依赖已纳入源码 SHA。

提供 `encode_owned/decode_owned` 严格 JSON 编解码和 `replay_diagnostic_bundle`，以及 `scripts/replay_dh116_flexible_r0.py --input ... --output ...`。NaN/Inf 用显式标签保存，不改成 0；无 native handle 可进入 bundle。CLI 从磁盘重读，绑定输入 SHA，拒绝已有输出或 symlink，最终 exclusive-create 回执。重放失败保存独立失败回执并 exit 1；禁止用同名输出覆盖成功证据。

实际可接线次序为：

1. 构造诊断器，明确传入完整原 task、grasp、env、独立 frozen-eval 预算、全部显式诊断系数、backend 文件及 SHA、radius value/source。
2. 把 `diagnostic.record` 作为新 `DualHandPhysicalEpisode` 的 recorder，执行其原 reset。
3. `diagnostic.begin(episode.pre_action_state(), action24)`。
4. `result = episode.control(action24)`；原 owner 的每条真实 physical 记录进入 recorder。
5. `diagnostic.end(result)`；保存 `encode_owned(diagnostic.bundle())`，再用独立 CLI 重放。

本包不构造场景、不驱动 SDK/solver、不执行 native reset，也不替代外层安全投影、reaction 或退出审计。测试使用原 owner / runtime / provider / reducer 代码与明确的 CPU SDK/solver doubles。没有原生轨迹，不能据此宣称实际抓取、接触或 PPO 已通过。

## 同钟与来源检查

- reset、pre-action、action pre/post、raw rod、provider boundary 均绑定 run / env / episode / reset / rod、SDK step / time、solver time / tick、policy step / substep。保留 `clock_valid=True` 要求；SDK 每条 +1、solver 每条 +DT、rod stride 由原 reducer 的显式预算检查。
- 从两侧原 mounted source packet 重建 66/371 已知字段，并核实际 pre-action q/qd；其余 305 维不补齐。核每个 pre/post 的两侧 path、17 个 named DOF、单实例、row index、mounted 状态、object/view ID、有限 q/qd/targets，以及跨物理步前后 state 连续性。
- ACTIVE 六索引和提交的 ACTIVE + 三直线 wrist + 三旋转 wrist 共 12 个索引，从原 joint names 重建。空 targets、错序、错误 shape、非有限值和 backend 不一致均拒绝。
- `RodBoundaryState.capture` 在纯只读 owned clock 上重新构造全部 provider 字段，比较速度、位置、rest lengths、contact、tangent、curvature、deformation 等；没有复制一份替代物理归一化算法。
- guard 调用原 `FlexiblePhysicalExchange._guard_failure` 的纯 packet 分支（该实现不使用 self），再核原 physical row 与 control result 的 reason 相同。这个私有方法及依赖纳入 SHA；未来原方法变化会拒绝旧 bundle。
- 原冻结 task 的 backend SHA 与当前 supplied backend 文件 SHA 不同。只有标明 `explicit_diagnostic_copy_only_not_frozen_task_admission` 的内存 task 副本替换该单一 SHA；原文件、系数 NA、材料参数、reset 合同均未改变。文件 SHA 核验不等于证明某原生进程加载了该文件；后者仍需外层证据。
- R0 的六个 reset callback 只核已有 reset 记录/owner 的明确 status/source。不会执行或重新排序 native owner。实际 joint→kinematics→rod reset 与原 R0 rod→joint 顺序没有被证明等价；`r0_native_reset_complete=False`。

## 为什么目前不能生成全 valid 的 R0 奖励

原 contact 包要求 `opposed` 为 bool。完整 pair 捕获中一个侧只有 0 或 1 个原始 pad 身份，足以排除“两 pad”这个必要条件，因此只形成负向的 `opposed=False`；不是对正向接触的认证。两侧有多个 pad pair 时，载荷、normal、同时出现计数都不能证明有效对向接触，所以原 R0 的整个该侧 contact 包保持 NA，原始 pair diagnostics 独立保存。

`loaded_node_rest_stations_m` 只是离散材料节点标签，没有亚节点材料锚、固定参考、漂移/不确定性来源，绝不转换成 material anchor 或 tangential slip。aggregate forces 也不提供 pad identity。没有 pair capture 时接触信息保持 NA；table SDF pair 独立，只表示该 solver 表面事件，不声称 PhysX hand-table contact，也不推出 external support=false。

原 R0 的首条 approach_progress 没有前值，即使当前距离有效也为 NA。无双侧已证实材料锚接触时 material_anchor_slip 为 NA。原 reducer 要求该窗口每个 reward term 每条都有值，所以这些窗口的 `diagnostic_reward` 和 `training_reward` 均为 None。没有修改原 reducer 的全项有效规则。

可选 auxiliary 只能携带同钟、明确来源的 approach distance / support / authored force / attachment / stable。缺失不填零；它们始终标明 caller-supplied，不能冒充 native source attestation。不能注入 opposed、material slip、节点锚或 contact 包。

原 joint-limit terminal 可在第一条结算；mimic/excess-force 这种原 physical guard 提前结束、原 R0 又没有相应已知 terminal source 的窗口，保存为 `physical_guard_closed_partial_window_without_original_R0_terminal_source`，不把稳定性设为 false，不补足 8 条，不结算。原 R0 若因额外明确诊断源提前 ready，后续超额 row 仍由 reducer 拒绝；本包没有改 owner 的终止传播合同。

## CPU 证据

最终相关回归：**306 passed in 28.28s**，其中 30 新增、276 既有；19 个文件、exit 0。命令与完整 stdout 保存在：

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/final-cpu.json`
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/final-cpu.log`

环境固定：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q`，其后为：

- `tests/test_deformable_r0_replay.py`
- `tests/test_dh116_deformable_control_step.py`
- `tests/test_dh116_deformable_grasp_task.py`
- `tests/test_dh116_deformable_task_budget.py`
- `tests/test_dh116_dual_hand_episode.py`
- `tests/test_dh116_flexible_episode_common.py`
- `tests/test_dh116_flexible_physical_exchange.py`
- `tests/test_dh116_flexible_rod_boundary.py`
- `tests/test_dh116_flexible_pair_contact_diagnostics.py`
- `tests/test_dh116_deformx_pair_capture.py`
- `tests/test_dh116_deformable_bimanual_step.py`
- `tests/test_dh116_dual_hand_clock_handoff.py`
- `tests/test_dh116_teacher_mounted_wrist_source.py`
- `tests/test_dh116_teacher_preaction_readback.py`
- `tests/test_dual_hand_reset.py`
- `tests/test_dual_hand_rod_runtime.py`
- `tests/test_dh116_deformable_rollout_admission.py`
- `tests/test_dh116_deformable_privileged_teacher.py`
- `tests/test_dh116_deformable_privileged_ppo.py`

新增测试覆盖真实 owner recorder 接线、双窗口 exactly-once、完整空/多 pad/table/缺失 pair、全部未知保持、action/reset/clock/backend 失配、重复记录、禁止 slip 注入、guard/instance/target 索引和 provider 衍生字段篡改、joint/mimic 提前终止、严格 JSON 与真实 CLI 子进程、已有回执不覆盖。先见失败再修复：guard/provider/CLI 6 failed +14 passed（已保存 red log）；实例连续性 4 failed（已保存 red log）；索引审查 4 failed +2 passed；CLI 重复输出 1 failed。修复后以上最终 306 全过。

独立磁盘案例位于同证据根目录，`cpu-replay-cases.json` 保存精确命令、exit 与文件 SHA。全部为 CPU doubles，CLI 实际子进程退出 0、`native_imports=[]`、0 新物理步：

| CPU 案例 | fake SDK 最终计数 / solver calls | 原 R0 结算窗口 | 诊断结果 |
|---|---:|---:|---|
| empty：完整空 pair，两窗口，显式 CPU auxiliary | 18 / 16 | 2 | approach valid 7/8、8/8；slip 0/8、0/8；两个 scalar 都为 None |
| multi：每个 solver substep 两侧各两 pad | 10 / 8 | 1 | simultaneous-required count 保留，opposed/slip 为 NA，不能产生 positive R0 contact |
| mimic：第一条原 guard 结束 | 3 / 1 | 0 | 明确 partial blocker，无伪造稳定性、无补步/结算 |

fixture-empty 的 input bundle SHA 为 `c8501ddc736ba905d235f458fdbc6c3c04764a234ceb82df2c2fe79582a1ff64`；输出 receipt SHA 为 `403df8238a515ccb3aeacf055ee4615950bffd06ae968d476634cdca3f31a424`。可用新输出路径重放：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/replay_dh116_flexible_r0.py --input /home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/fixture-empty/owned-bundle.json --output /tmp/unique-r0-replay-receipt.json`

## 审查、文件与剩余条件

wireseg peer 做了只读审查，没有重跑测试或 native。其提出的输出覆盖、active/target 索引空切片缺口已复现、修复，并由其复核关闭；未知量、原 reducer stride / once / partial 边界未发现新问题。此审查不等于原生验证。

本提交只新增 module、CLI、test、本报告四个文件。旧配置/材料/physics/模型/checkpoint 与旧冻结源码未编辑；没有 registry、README、native recipe、发布或原生启动修改。

可以接入明确标记的 physical-data 诊断 recorder，尚不具备训练入口。仍缺来源已证实的物理 opposition、固定材料锚与 slip、不缺项的 approach/support/stability 源、原 R0 native reset 合同、完整 371D 与原生资格。原 task/reward/admission 不因 CPU 重放成功而改变。

## 哈希

- `src/go2_pvcnn/control/dh116/tabletop_deformable_r0_replay.py`: `79fdc32b04638672c9e38092294c4b03cad4549e519ce8bc8ea9323a25dfe8d8`
- `scripts/replay_dh116_flexible_r0.py`: `ca296723d28b534480a01774a19f74b031834ce2b8c5bf3e969fadf8d61b425a`
- `tests/test_deformable_r0_replay.py`: `bd798ff40742459a90e15d8f018aed98c53e295a649334effa8f19a53dd6c608`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/final-cpu.log`: `d8fab0644ca20865d42bff4318b8d25d8ec7f8c2a49182b21c135299298421ab`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/final-cpu.json`: `b316daed5fc0419f8099ea1420078216e0cb847ea236a39936557de1afa6b6c8`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-r0-replay-cpu-okqe5163/cpu-replay-cases.json`: `b90b204a8b34ae698e9dcbd73c0d2b24efe879f9f651ed12d8256726563db144`
