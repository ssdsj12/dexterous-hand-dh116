# Mounted wrist：原 371D 的 [153,185) 实测来源扩展

基线 94051e118215c31889888a9b074c82d5d996d040。只新增独立来源模块、CPU 测试和本报告；未修改冻结 TensorWorld/free_wrist、旧 source validator、Teacher、PPO、配置或 checkpoint。新增物理步 0、optimizer updates 0；未启动 Isaac、solver、GPU 或策略。

首版 dh116-teacher-mounted-preaction-source-v1（现由文末路径修正 v2 取代）仅用于具备原 6 轴 MountView 的实际 17DOF hand 接口。旧 validator 不改，仍拒绝该新版本。新 validate_mounted_observation() 从保存的 source packets 重建数值与 mask；require_mounted_teacher_common() 在存在任何未知字段时拒绝导出。当前最多 66/371 字段可由 CPU 已验证的适配器提供，其余 **305 项仍未知**，不授予 Teacher/native 资格。

## 精确原语义

| 字段 | 左手半开索引 | 右手半开索引 | 来源与原归一化 |
| --- | --- | --- | --- |
| pose | [153,160) | [160,167) | 实测 base_link actor-frame world pose；xyz 减 authored root_seed 后除 0.05，raw WXYZ 原样保留 |
| velocity | [167,173) | [173,179) | 原 body_lin_vel_w/body_ang_vel_w；world COM 线速度除 0.05，角速度不缩放 |
| mount angles | [179,182) | [182,185) | 实测 joint_pos 中 wrist_roll/pitch/yaw，按名称排序并除 pi/2 |

TensorWorld.refresh() 在 mounted 情形用 body_link_pose_w[:,mount.base_id] 作为 wrist_pose；measure() 直接使用 body_lin_vel_w/body_ang_vel_w。实际安装的 IsaacLab 中，这两个 velocity 属性是 body_com_* 的别名，来自 PhysX get_link_velocities()。因此没有改用 body_link_vel_w、没有把 COM 速度换算到 link origin，也没有把 mount/rail anchor 或 body COM pose 当 palm pose。

本机已只读核对 articulation_data.py：
/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py
SHA256 d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a。
body_link_pose_w 在 583 行起以 get_link_transforms() 读取并从 XYZW 排列为 WXYZ；body_com_vel_w 在 643 行起读取 get_link_velocities()；body_lin_vel_w/body_ang_vel_w 在 1071/1076 行是 COM 别名。此为来源语义检查，不是新的原生采集结果。

capture_authored_reset_seed() 从原 TensorWorld 的 layout.hand_roots、environment origins、seed.root_xy 重建并逐值核对 root_seed。明确排除 initial_offset/root_offset；也不使用 episode_reset_root_pose_m、第一次 measured palm 或 rail anchor。seed 绑定 run/env/episode/reset/rod，跨 reset/episode 不能复用。

## 读回与 validator

capture_mounted_hand() 检查：

- 完整、唯一的 17 个 joint names，原 MountView 实例/hand/base_id，以及三个线性与三个角度 joint indices。
- 实际 articulation prim_paths、body_names 与对应 base_link 的完整 link path；data row 由显式映射选择，不从 env_id 猜索引。
- SDK integer DOF types：wrist_x/y/z 为 translation，finger 与 intrinsic wrist angles 为 rotation。
- data.joint_pos/vel 与 SDK get_dof_positions/velocities 逐值相等。
- data.body_link_pose_w 与 SDK link transform 的 XYZW→WXYZ 结果逐值相等；原 COM velocity cache 与 SDK link velocities 逐值相等。
- 两次 shared clock 读取之间 identity 不变，clock_valid 必须为 true，substep 属于 [0,8)，SDK timestamp 与 480Hz tick 一致，并保留 solver_time_s。
- 缺数据、固定基座 11DOF、错名称/单位/path、陈旧 cache、非有限值、错误 quaternion、来源变化或归一化溢出均拒绝，不以零、target 或其他 frame 补齐。

已有 11-finger 来源适配逻辑通过明确命名投影复用。单手为 17 个已有字段 + 16 wrist 字段 = 33；双手为 66。所有未知值继续为 NaN、valid=false、source=null。新 validator 重建 source packets 后比较 mask、数值、来源、资格及执行计数，拒绝补零、篡改或伪造全 true mask。packets 绑定新模块及相关原 helper 的实际源码 SHA。

reset owner 在写 joint state 后必须完成 physics_sim_view.update_articulations_kinematic 及原 SDK/Articulation 刷新；未刷新的 link transforms 可能与旧 cache 一起保持陈旧，二者相等本身不能证明 reset 刷新。clock 回调的真实性、SDK 刷新和同步仍由调用方负责；两次回调与缓存/SDK parity 不能独立证明真实原生执行。新版本没有自动接入现有 rollout admission CLI，旧入口也不会悄悄接纳它。后续整合仍需独立审查，且剩余 305 字段尚未实现。

## CPU 验证与可重放 fixture

先写失败测试确认缺模块。原 observe() parity 测试起初误用了 PyTorch 默认 float32，而 TensorWorld.t()/measure() 使用 float64；修正 fixture 到原实际 dtype 后，保持严格逐值比较，没有放宽容差。归一化溢出与执行计数篡改也先观察失败，再补拒绝逻辑。

新增 **21 项 CPU 测试，相关总计 106 passed / 5.14s，退出码 0**。原 tabletop_free_wrist_grasp.observe() 的前 32 个字段与新 [153,185) 输出逐值相同，包括负号 WXYZ、COM velocity 和 authored seed offset 区别。其它覆盖固定基座拒绝、命名/DOF/path、SDK/cache 陈旧、时钟变化、seed 复用、单手未知、owned copies、mask/value/计数重建以及有限输入导致归一化溢出。

~~~sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_mounted_wrist_source.py \
  tests/test_dh116_teacher_preaction_readback.py \
  tests/test_dh116_deformable_rollout_admission.py \
  tests/test_dh116_deformable_bimanual_step.py \
  tests/test_joint_target_readback.py
~~~

私有证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/mounted-wrist-source-20261113

- snapshot-fixture.json：695608d39fa38df4b545f3eef1a22b88fc9c518b978488676b990c353d741b37
- replay-proof.json：73585595e9fbf049331c2b7b52ecd9ec2699efc54133eebe78ce152a6255097e

文件明确标记 CPU_API_DOUBLES_not_native_capture。将保存的 source packets 重新输入适配器后，66 个归一化值和全部 mask 完全一致，未知数仍为 305。原生成功池和训练状态不变。

独立 physics agent 已只读审查完整模块，未见新的 clock/原观察语义阻断；未将该审查计作新原生证据，亦未重复测试。

## Articulation root 与 hand container 身份修正（独立后续提交）

基于 4684d89ef38cdad4a91cfa3dfc7fba527489aa5e。只改本模块、本测试、本报告，未改其它包。安装版 articulation.py 的 _initialize_impl()（1506–1547 行）查找实际 ArticulationRootAPI prim 后，以其路径创建 articulation view；本仓库 tabletop_palm_mount.py（37–39 行）把该 API 放在 hand container 下的 wrist_fixed。base_link 位于同一个 hand container 下，并非 wrist_fixed 的子节点。因此首版将 view prim_path 与 hand container 当成同一身份，会拒绝此有效布局。该结论来自只读源码与 CPU 复现，尚未测量实际 native backend 返回路径。

capture_mounted_hand() 现在要求两个显式参数：expected_articulation_path 与 expected_hand_container_path。前者由 scene owner 建立实际预期 view 身份，并逐字核对 SDK prim_paths；后者独立核对 base_link 及该 row 的所有唯一 link_paths。支持显式核验的 container path 或其 descendant path，但不猜 backend 规范化规则、不自动剥离或添加 wrist_fixed。路径必须绝对、无空组件或 dot traversal，不能以 leftHandExtra 冒充 leftHand。正确 mount fixture 为 articulation=/World/envs/env_8/leftHand/wrist_fixed，body=/World/envs/env_8/leftHand/base_link。

packet 版本提升为 dh116-teacher-mounted-preaction-source-v2，保存 articulation_path、hand_container_path 和 body_path；重放也验证三者关系。旧 v1 packet 与旧源码 SHA 不能借用新 validator。以上原始 snapshot-fixture.json / replay-proof.json 保留不变，是首版 CPU 证据，不声称已升级。新增独立 mount-path-v2-fixture.json（同目录）SHA256：
b9f6b0ef03f06f03cc3157fa4c93927316e0d1ff4b6755cf4f92217c80870e07。
它明确是 CPU API doubles，双手各以 wrist_fixed view 与 sibling base_link 成功捕获，66 项有效、305 项未知，actual_physics_steps=0 / optimizer_updates=0。

先用新 mount 路径 fixture 在原实现复现 1 failed（actual SDK environment/body paths differ）。随后增加分离路径、错 env/side、错 container prefix、把 base_link 错放在 wrist_fixed 下、dot traversal 和旧 packet 拒绝测试。当前 mounted 共 **28 passed / 1.86s**；使用上节相同命令的相关回归为 **113 passed / 3.81s，退出码 0**（相对首版增加 7 项）。git diff --check 通过。physics agent 对 v2 diff 独立只读复核，无新增阻断；未把该审查计作测试或 native 运行。

## 下一最小可执行包建议：控制器拥有的 B40/B41 来源（本次未实现）

已逐块核对冻结的 stereo_privileged_state_contract.json 和原 observe()/begin_step()/end_step()。以下均为控制器内存的事实，不可用 measured joint state、SDK target、action decode 推测值或默认零替换：

| block / 半开索引 | 原对象字段 | 关键时序约束 |
| --- | --- | --- |
| B02 [24,36) | task.previous | end_step() 才 copy_(command)；不是 old_command，不能未经完成边界就填 current command |
| B03 [36,48) | task.reference | 原 template/reset seed 与 per-env reset 状态；不是 measured q，也不是 authored root_seed |
| B37 [185,191) | task.rotation_targets | 本地 requested intrinsic angle target；不等同实际 mount angles |
| B38 [191,197) | world.wrist_angles | advance() 经 allowed/live 门后保留的 submitted angle 命令 |
| B39 [197,203) | world.palms | advance() 经 transport 和安全投影后的 offset；不是 palm_targets 或 measured palm |
| B40 [203,215) | task.command | 原 float64 governed finger command，经原 low/high 归一化；SDK float32 target 仅可旁证提交 |
| B41 [215,239) | task.actions | 当前 pre-action 边界之前最近一次 accepted 24D action；不是 previous_actions 或尚未提交的下一个策略动作 |

建议最小下一包只实现 B40+B41，共 36 个字段：独立 owned controller packet，在原 begin_step() 前由同一 shared clock 读取实际 task.command/task.actions；绑定明确 action24 contract、原 active joint order/limits、reset 初始化记录、最近一次 accepted submission 的 run/env/episode/reset/rod/policy 身份以及当前 pre-action clock，拒绝提交中间态、跨 reset、缺失来源及未来 action。不得用任意 caller 字典加 valid=true 授权；重建 validator 需保留原对象来源与实现 SHA。不能为了测试创建 Teacher/optimizer；用 CPU 原 begin_step/observe 与固定 controller doubles 对照，覆盖 owned copy、交错 step、reset、24/26D 混用与当前 command 的 float64/SDK float32 差异。

仅在调用方确实拥有这些已核验 controller 状态与 submission/reset 身份时，它们才可增加 36 项；理论上合并 mounted 的有效上限为 102/371、269 项仍未知，**当前仍为 66/371**。启动时的零 action 只能来自已记录的原 reset 写入，缺少 reset 记录时必须保持未知。B02/B03/B37–39 留给随后具有完整 reset/end-step/transport 生命周期证明的包；B16 还需要 valid consume 的更新事实，不能直接按当前 command-reference 现场重算。root_seed 已作为 mounted 归一化来源绑定，不单独增添观察维度。B70 虽来自 action，也属于完整已完成物理区间的平均，不能把单次 action 当区间均值。刚体 cylinder、rod、接触与 force 来源本次均未扩展。
