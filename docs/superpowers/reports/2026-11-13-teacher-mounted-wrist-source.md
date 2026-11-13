# Mounted wrist：原 371D 的 [153,185) 实测来源扩展

基线 94051e118215c31889888a9b074c82d5d996d040。只新增独立来源模块、CPU 测试和本报告；未修改冻结 TensorWorld/free_wrist、旧 source validator、Teacher、PPO、配置或 checkpoint。新增物理步 0、optimizer updates 0；未启动 Isaac、solver、GPU 或策略。

新版本 dh116-teacher-mounted-preaction-source-v1 仅用于具备原 6 轴 MountView 的实际 17DOF hand 接口。旧 validator 不改，仍拒绝该新版本。新 validate_mounted_observation() 从保存的 source packets 重建数值与 mask；require_mounted_teacher_common() 在存在任何未知字段时拒绝导出。当前最多 66/371 字段可由 CPU 已验证的适配器提供，其余 **305 项仍未知**，不授予 Teacher/native 资格。

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
