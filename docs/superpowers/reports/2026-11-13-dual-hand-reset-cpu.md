# 双手原地 reset 事务：独立 CPU 准备

基线：`8a9b26759c92e880ad726075f6637aa1bc8feec0`。本包只增加独立 reset 模块、测试及本报告；没有修改旧 physics、任务配置、控制器、recipe、registry 或 README。没有启动 Isaac、DeformX、GPU、optimizer 或训练。

当前结果：新 20 项 CPU 测试通过；剔除已独立复现的旧 fixture 失败文件后，相关 14 个测试文件共 **312 passed in 16.24s**。这验证调用、映射和失效处理，不能证明原生 reset 已成功，也不授予初始几何、接触、R0 或 Teacher 资格。

## API 与范围

`prepare_reset_state(hand_roots, origin, root_xy, initial_offset_m, joint_positions, joint_velocities, source_description)` 要求显式输入全部物理 seed；返回可序列化内容及 SHA256。`hand_roots` 是左右手 WXYZ pose，q/qd 为 canonical 11DOF。复用原 `TensorWorld` 的 seed 算术：

- `authored_root_seed_w = hand_roots + environment origin + per-hand root_xy`。
- `reset_root_pose_w` 再加显式 `initial_offset_m`，二者分开保存。
- 11 个 q/qd 保留调用方值，不推断 follower 测量；新增支架 6DOF 的 q/qd 为零。
- 主动关节验证原 `active_limits()`，完整 11DOF 验证原 source limits；不改限位、材料或动力学。
- SHA 证明输入内容未变，不证明调用方 `source_description` 来自已合格源，不证明姿态安全。

`DualHandResetTransaction(seed, episode_uid=...).run(sim, rod, bindings, owner_resets=...)` 消费一个现有 `DualHandRodRuntime` 和 canonical left/right `HandBinding`，严格单环境。UID 必须由上层按运行/episode 生成；已见 UID 拒绝。一个事务只能调用一次；成功返回 owned 数组/标量，不保存 hand/view/rope/composer/native handle。

`owner_resets` 必须显式提供 `controller/filter/contact/camera_history` 四个 callable；每个接收独立 context 副本，返回 JSON-safe `{owner, status, source, episode_uid, ...}`。允许 `cleared/not_present/unavailable`，不以缺失字段默认为零。这里验证回执身份、类型和来源标签，**不证明 owner 内部语义**。控制器 accepted snapshot 仍由原 owner 负责；`accepted_controller_snapshot=False`。

## 真实调用顺序与读回

在任何写入前核验 seed SHA、两手实例/actual articulation paths、17 个真实 joint names/types、23 个 body names/paths、10 个 rod pad body index/path 及现有 MountView 绑定。允许安装后 SDK 的 articulation path 是 hand container 或其 `wrist_fixed`，但必须与 pad container 对应。

1. 拷贝实际 SDK step/time/dt 与 rod clock，读取每只手的 native q/qd、root transform/velocity、position/velocity targets、stiffness、effort limits、explicit DOF actuation 和 composer scratch。原始 pre-state 的非有限值保留并标记；reset 可用于恢复 q/qd，不能把原状态宣称有效。
2. 每手按原 `TensorWorld.reset` 调用 `hand.reset`、`write_joint_state_to_sim`、`MountView.reset`、`write_root_pose_to_sim`、`write_root_velocity_to_sim`、主动位置 target、主动 stiffness=10，并同步 `actuators["active"].stiffness`。调用原 `write_contact_effort` 用现有 rod candidate 的主动 effort limit 恢复零 explicit effort；六个 wrist/被动 gain 与 effort limits 保持读回相等。
3. 显式清零所有 17DOF velocity/effort target。对两个 composer 做完整 `reset(None)`，向 permanent composer 提交完整 23-body 零 wrench，经原 `hand.write_data_to_sim` flush 后再完整 reset 两个 composer。安装源码的 `reset(env_ids)` 不关闭 active；本包严格单环境，因而使用完整 reset。
4. `sim.physics_sim_view.update_articulations_kinematic()`，然后 `hand.update(DT)` 刷新 ArticulationData 缓存时间。这里不调用 SDK step。随后用 `get_link_transforms` 取得新真实 pad poses（XYZ + XYZW），不是源 USD、旧缓存或 seed 推算 pose。
5. 原 `DualHandRodRuntime.reset` 在**同一** rope 上执行一次，`reset_id + 1`、新 episode UID、`rod_tick=0`。SDK step 与 solver time 保持不变；10 个 mesh pose 由刚读回的真实 pad pose teleport。原始 `rod_reset` 行完整返回，不再归一化 provider。
6. 执行明确的 owner callbacks，再复读真实 SDK buffers、composer 与时钟；验证 q/qd/root/target/gain/limit、两手身份、同一 rope、reset identity，以及 composer 零且 inactive。

零力证明边界：

| 项目 | 证据/限制 |
| --- | --- |
| 手 composer scratch | 拷贝实际 `composed_force_as_torch/composed_torque_as_torch` 并验证全零、inactive |
| explicit DOF actuation | 实际 `get_dof_actuation_forces` 全零；不是总 PD 力矩或外部 body wrench |
| 外部 body 零提交 | 复用安装 `Articulation.write_data_to_sim -> apply_forces_and_torques_at_position` 的全零 API 调用路径；CPU 用 double 验证 flush |
| native external-body-force readback | 未找到已确认 getter，返回 `status="unavailable_no_getter", values=None, zero_write_observed_at_backend=False` |
| rod contact/reset scratch | 冻结 `FreeRodRuntime.reset` 明确可能保留 force scratch；不猜零、不作为新接触测量。原行 `contact_valid=False, distributed_contact=None, named_wrenches=None` 保留 |

任一 mutation 后异常将事务标为 failed，并令 rod `_needs_reset=True, _base.last=None`。不伪造 rollback，不恢复计数，不允许重试该事务；仍须由外层 native lifetime owner 清理异常 traceback 及 SDK 引用。本包不创建、关闭或释放原生 owner。

`r0_reset_complete` 恒为 false：现有 R0 hook 顺序是 rod→joint→controller→filter→contact→camera_history，而物理 reset 需要先 joint/kinematics 再以新 pad pose reset rod。这里提供一个真实物理事务，**未以 no-op hooks 冒充 R0 链**。上层应在此成功后显式对接 provider/R0 owner。本包不推定 rod 初始 pose 无穿透，所有 `initial_pose_qualified/native_qualification/teacher_admission` 恒 false。

## 安装源码依据

源文件 SHA 全记录于 private `final-cpu-manifest.json`，实际读取但未 import/运行安装的 Isaac 模块：

| 文件 | 关键位置 | SHA256 |
| --- | --- | --- |
| `src/go2_pvcnn/control/dh116/tabletop_tensor_world.py` | constructor 114，reset 565，write_contact_effort 32 | `dd2a17bc397f7ec14254c3373ff1ef22aa4515b156fd25db50deba3052f564f4` |
| `/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` | reset 208，write_data_to_sim 218，root pose/velocity 399/485，joint state 561 | `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309` |
| `/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/utils/wrench_composer.py` | active 71，scratch getter 100/112，set 221，reset 320 | `095ab1fcf374e4bce96b61430f810c83f3519b7225476ce6e0a6ceb403a5f6a8` |
| `/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py` | stiffness 807，max forces 859，link/root transforms 1185/1270，q/qd 1379/1408，targets 1645/1674，actuation 1703 | `bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b` |

## CPU 验证与已知基线失败

新 20 项覆盖：真实 runtime composition 中的同 rope reset/count、双手真实名字顺序重排、authored seed/offset 区分、实际 post-refresh pad routing、所有 owner 显式要求、缺 getter/错误 pad/root/mount/seed 拒绝、七类 mutation 故障（hand、kinematic、rod reset、owner、target readback、SDK tick、solver time）、新 UID 递增与旧 UID 拒绝、接触未知状态保留、非有限 pre-q 恢复、owned receipt 不保留 native-shaped handles。

最初完整 15 文件运行：**316 passed / 4 failed**。四项均来自原 `tests/test_dh116_root_pose_writes.py`：三个旧 fixture 未设置 `TensorWorld.device`；一个旧 reset fixture 未设置 `roots`。单独运行该旧文件亦 **4 passed / 4 failed**。该测试与 `tabletop_tensor_world.py` 都核对为 base `8a9b267` 同字节 SHA；本包未改这两个文件。失败日志保留，不将排除后的通过数写成全套通过。

最终有效相关范围：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q` 后接：

`tests/test_dual_hand_reset.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_deformable_bimanual_step.py tests/test_dh116_flexible_physical_exchange.py tests/test_dh116_deformable_control_step.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_deformable_grasp_runtime.py tests/test_joint_target_readback.py tests/test_dh116_palm_mount.py tests/test_free_wrist_mount.py tests/test_dh116_tensor_world.py tests/test_dh116_contact_effort.py tests/test_dh116_flexible_solver_clock.py`

独立 import probe 退出 0，`ropesim / warp / isaaclab.sim / omni.kit.app / omni.physx` 的 loaded modules 为空。没有原生启动、物理 step 或训练 step。

Private evidence：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-hand-reset-cpu-2fq1_oaj`。

| 文件 | SHA256 |
| --- | --- |
| `related-cpu.log`（完整范围 4 个旧失败） | `e25d4cab3069452f6e79355031dcc0edebcddc18b49cbd2f6cfce57608f6b926` |
| `baseline-root-pose.log`（独立复现） | `077e2f47d828653cae197083a3fd0b82db6b394484c1ebb29ff7b383b07bfd52` |
| `baseline-root-pose-manifest.json` | `d48e13fa5a184d5dd5aa41f72688847200142d40081396bdcca1eda31447236f` |
| `final-cpu.log`（312 passed） | `73799970ef64a50372e1d93106be16ebcc779ad36a21ad3ee7283c683a403f70` |
| `import-boundary.log` | `b2391aca4a4af382a24eebca1f363cb6ddc11da9fb2c3423fb0ceb320bc241c3` |

代码 SHA：`tabletop_dual_hand_reset.py = da35b878d6ef4d5236649e2cadd97ae1a66972ce3f0646419d69c0a6d0c6bce2`；
测试 SHA：`test_dual_hand_reset.py = da8edd7676af069f8b0813d5de4a4c3287d2a391c0b90c04c78fe9ce55eaeae9`。

独立只读短审：camera agent 检查全量 reset 源，未发现新增阻断；确认 composer 清零/flush 顺序、native body-force NA 边界、owned 状态与 mutation failure poisoning、实际 SDK/solver clock 检查。该审查未编辑源码、未重跑本包 CPU、未运行 native。
