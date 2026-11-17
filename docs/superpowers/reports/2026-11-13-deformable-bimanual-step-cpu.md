# 自由杆双手 24D 单物理步连接层（CPU）

本包新增可调用的 `BimanualStepConnection`，连接双手实际 SDK 测量、原 canonical 24D 目标路由及动作后的 target readback。没有创建 SimulationApp、推进 SDK/DeformX、训练或创建 optimizer。Pinch、Tripod、Power 的 native/Teacher 资格仍为 false；没有新 recipe。

基线：`fb6fe4b563154f72ae9fffe6105bdb604a749bea`。独立 worktree：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/deformable-bimanual-step-fb6fe4b`。只新增源码、CPU 测试和本报告，原控制器、物理参数、配置、checkpoint、registry 均未修改。

## API 与实际调用链

`HandBinding(side, hand, env_paths, mount)` 由调用者拥有真实 articulation/MountView。`env_paths` 是外部 env ID 到预期 authored articulation prim path 的显式映射。连接层读取实际 `root_physx_view.prim_paths` 找到每个 buffer row；左右 articulation 顺序可以不同，但必须完整且唯一。必须是不同 hand 和 PhysX view；事务也检查对象 identity、实际名称/路径及 native DOF 类型没有被替换。11 个原 DH116 DOF 和增加完整 6 轴 carriage 的 17 DOF 分开处理；没有完整 MountView 时，非零腕动作在任何 write 之前拒绝。

每个 SDK 物理子步创建一个事务：

1. `begin(bindings, action24, read_clock=..., palms=..., angles=...)`：在两个一致的 shared-clock 读取之间，复制两手实际 `get_dof_positions/get_dof_velocities`、`get_dof_position_targets`、`get_dof_types` 与路径。target reader 复用现有 `read_joint_target_evidence`；command cache 单独保留，不能补足缺失的 backend。q/qd 直接来自 PhysX，避免未更新的 IsaacLab cache 冒充当前状态。
2. 原 `ACTION_CONTRACT` / `validate_actions` 检查 [N,24] 浮点动作。按原 `finger_pre_guard(policy_owned=True)` → `surface_final_guard` 和 `wrist_requested_targets` 计算 requested targets。动作分组保持原 left/right fingers 各 6、world XYZ translation 各 3、seed intrinsic XYZ rotation 各 3；没有 force action。每步从实际 backend target 作为 finger command baseline，不假装保留旧 TensorTask 的 double command 积分缓存；这是独立桥接策略，report/request 中显式标注。
3. `submit` 在任何 SDK write 前检查两手、时钟、q/qd、backend targets 与 mount anchor 均未改变。腕需要调用者提供原 table guard 投影后的 `projected_root_poses`、`projected_angles` 和来源。连接层只验证有限值、quaternion 规范化与原 carriage 行程/角度限制，不声称做过碰撞检查。requested palm offsets 与实际 projected world poses 分账，不能直接把前者当成安全 world pose。
4. 实际写入复用原 `MountView.move`（含 anchor-local translation）、`hand.set_joint_position_target` 和 `hand.write_data_to_sim`。不修改 gains、effort limits 或 actuator 配置，不添加 force。后一个方法会 flush 调用者已有的 wrench composer；外层必须按既定反作用力时序准备该 composer。本包不能代替这项物理责任。
5. 外层执行唯一一次 SDK step、对应的一次 rod step，并进行原 SDK refresh。连接层不持有/调用 step 或 reset 函数。
6. `finish` 复制 post 快照，要求同一 run/env/episode/reset/rod/policy/substep，SDK physics_step 恰好 +1，rod_tick 恰好 +显式 solver_substeps，solver_time 恰好 +1/480 s，并检查写入的 active/wrist target 与真实 PhysX float32 backend readback 相等（absolute tolerance 1e-7）。target 一致并非实际 q 跟踪完成，更不是抓取成功。

`read_clock()` 返回每 env 一行，字段为：

`run_uuid, env_id, episode_uid, reset_id, rod_uid, physics_step, rod_tick, policy_step, substep, timestamp_s, solver_time_s, clock_valid`。

`clock_valid` 必须严格为 True；solver 或后验校验失败后的失效 clock 不能仅凭完整整数字段通过。`timestamp_s = physics_step / 480`；整个 batch 必须是同一 run 和 SDK tick。solver_time 读取实际 solver clock，允许 reset/setup 所造成的起点偏移，但 pre/post 增量必须相同。policy_step/substep 是本次动作子步标签，pre/post 保持相同；本包不累积八子步或结算 reward/terminal，那仍属于独立 R0 control-step owner。

事务仅储存 owned NumPy/Torch CPU 数据和 scalar identity，不保留 HandBinding、articulation、view 或 read_clock 闭包。成功事务不能重用；write 失败后事务失效，不能假装回滚或重试。调用栈异常仍需外层既有 native teardown/traceback 清理流程负责，本包没有新增 app 生命周期路径。

## 真实 API 源码核验（只读，无 SDK import/launch）

已读本机 installed 源码，记录 SHA：

| 源码 | SHA256 | 核验点 |
|---|---|---|
| `/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` | `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309` | 1079: `set_joint_position_target` 仅写 internal buffer；218/258: `write_data_to_sim` 调 actuator model 后对 implicit actuator 写 backend target，并 flush 已有 wrench |
| 同目录 `articulation_data.py` | `d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a` | 470–482: root link pose 从实际 `get_root_transforms` 读取并 xyzw→wxyz |
| `/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py` | `bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b` | 622: actual prim_paths；677: types 0 rotation / 1 translation；1270: root transforms；1379/1408: actual q/qd；1645: backend target buffer |

MountView 使用的 IsaacLab anchor cache 还须与本次实际 backend root transform 一致（1e-6），否则拒绝；没有用缓存自行宣称实时性。所有 native getter 输出复制后才保存。

复用源码 SHA：`tabletop_command_kernels.py acc8687ee965fb3f4dc11859083ca9a5bc84139a604101a219ac01d518a5ccd8`；`tabletop_surface_contract.py f1a28e5dca4e44a49d16244d92c2c849023f8030943cbec62853dcaeebabdb00`；`tabletop_palm_mount.py 4eb43511df607c5fee1503d9b389bf49ed52102dd7304c19e4bca7028f60e307`；`tabletop_joint_target_readback.py 3037241c9468a8c7d9bc57cdedd1f612b9942c1b8d15a7cc33fda626682026b8`。完整 sources manifest 见下方私有 evidence。

## CPU 验证

最终命令：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
bash scripts/python.sh -m pytest -q \
 tests/test_dh116_deformable_bimanual_step.py \
 tests/test_joint_target_readback.py \
 tests/test_dh116_teacher_preaction_readback.py \
 tests/test_dh116_palm_mount.py tests/test_surface_contract.py \
 tests/test_free_wrist_grasp.py tests/test_free_wrist_mount.py \
 tests/test_bimanual_physical_action_adapter.py tests/test_deformable_grasp_runtime.py
```

实际结果：**174 passed in 6.02s**，其中新包 25 项，原相关范围 149 项。测试使用标明 CPU-only 的 SDK API doubles，验证真实生产 API 调用形状与路由，不冒充原生物理结果。

覆盖不同左右实例行顺序、反序 17DOF 名称、所有 24D 分组、原 MountView anchor-local translation、11DOF 腕拒绝、错误/缺失关节与 DOF 类型、缺失真实 backend 但 cache 存在、cache 掩盖 target mismatch、同路径替换 hand、SDK/rod/solver/reset/policy clock 错位、所有手的校验先于第一 write、write 后失败不可重试、返回数组 owned 且不保留 SDK owner、实际 q/qd 不受 stale IsaacLab cache 影响，以及 stale mount anchor 拒绝。三项 clock_valid=False/None/1 负例先得到 3 failed，加入严格有效时钟门后通过。新增两项 stale-cache 测试曾先得到 **2 failed**，修复实时 source 绑定后全过；此前一次失败仅为测试 regex 大小写不匹配，已修复。

私有日志目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/deformable-bimanual-step-cpu-9yg0o2n4`。

- `final-v2-cpu.log` SHA256：`9e35e08749392faf021636604fb61a2e82a882c9dcf0594402d5833472b07b68`。
- `final-v2-manifest.json` 记录完整命令、环境、源码及 installed SDK SHA。
- 新源码 SHA256：`91d1ac2cf70da43d309dbdf83e53008ba480b5774e3be0a7a93a2d6ddb98bc8f`。
- 新测试 SHA256：`d8b8fc9dcf892628a2902c444370a96b2b46e64ba0482fa0b3ae4b998902bdbc`。

独立 wireseg/R0 agent 已只读审查本文件对应实现、MountView 和 command kernels，未发现 clock/R0 边界阻断；没有代替本次 CPU 执行或原生验证。read_clock 的真实来源、palms/angles 的当前控制状态来源及原 table projection 仍由外层 owner 负责。后续必须显式把 action token/policy/substep 与 R0 reducer 对接，不能将本包 packet 直接当作 R0 settlement。

## 尚未完成的物理依赖

原 `FreeRodRuntime` 仍是单手五个 distal mesh + table，旧 native stereo launcher 仍拒绝 `deformable_rod`。另一个独立工作包正在提供双手十个 mesh 的组合层；该层须与本包共用外层时钟，并在新批准预算下检验真实 SDK/DeformX 路径、反作用力和自然退出。本次没有运行这些步骤，也没有修改旧 Pinch recipe。

原 371D preaction adapter 严格接受 11 个手指 DOF，而本包 17DOF packet 保留全部实际 q/qd/type/name，不伪装成该 adapter 的已准入来源，也不填补其 337 个未知 common channels。尚未接入完整 actor/Teacher input、R0 native settlement、material anchors/contact normals、三 style 的真实安全接近与 grasp qualification。原 pose/force/material NA 继续保留。
