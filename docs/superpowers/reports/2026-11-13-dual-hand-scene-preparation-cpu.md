# 双手自由杆场景准备：原 USD、驱动腕、锁定 SDF 与实际 link 几何

本包为后续独立双手自由杆 owner 提供 source-bound 场景准备、真实 USD 离线 authoring 和实际 SDK view 只读几何绑定。原 `tabletop_vec_isaac.spawn_scene(..., external_wire=True)` 已能构建两只原 DH116、原 driven carriage、mimic 和手材料；本包复用这个入口，不复制 GpuWorld，不创建刚性 Wire，不启动 SimulationApp，不执行 SDK/solver step 或训练。

基线 `94051e118215c31889888a9b074c82d5d996d040`；独立 worktree `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/dual-hand-scene-preparation-94051e1`。仅新增模块、CPU CLI、测试及本报告。旧物理源码、USD、候选配置、保护文件、checkpoint、registry 不变。

## 最小可复用部分

- `prepare_scene(project)` 调原任务/候选 loader、`load_geometry`、原 F4 parity lock 和 SDF SHA reader，绑定两份原始 USD、每手 17 个 source collider、每手五个 locked cooked hull / SDF，以及实际读取的 parity、自然退出结果和 schema 文件。项目内 source key 使用相对路径；真实外部 evidence 路径保留。
- `author_source_stage` 在匿名 CPU USD Stage 引用原资产，调用原 `author_mount(rotation=True)`、`install_hand_mimic`、`configure_mimic_compliance` 和材料 binder。创建有限桌面和重力 USD 属性。检查结果为 **34 原 collider、10 个软 pad、12 个腕 joint**。这只是 USD 检查，不是 SDK 初始化；没有应用 native initial joint state 或 SDK actuator override。
- `spawn_parameters` 明确返回原 builder 需要的参数、layout 和两侧 USD 路径。原 builder 将根位置增加 **0.04 m**，准备记录同时保留 source root pose 和 builder initial root pose，未掩盖该差异。原 builder 默认 active effort 为 **0.2 Nm**；准备参数显式使用 `LoadParameters(effort_limit_nm=.15)`，与独立 P0 自由杆候选一致。原 active stiffness/damping 10/.2、mimic frequency 400、damping ratio 1、solver iterations 64/16 和 six-axis wrist 配置继续复用。
- `assemble_prepared_scene` 是留给未来已批准 owner 的 lazy delegate，只调用原 `spawn_scene(..., external_wire=True)`，补显式 table material 并检查 USD。无 launcher/reset/step。**本次没有执行这个 native assembly 函数**；它的调用还需要原生命周期 owner 和新批准预算。
- `validate_simulation_settings` 通过真实 `SimulationContext.get_physics_dt` 和 `SimulationView.get_gravity` 检查初始化后的 480 Hz / [0,0,-9.81]。这是独立 readback 入口，不凭配置声称实际重力通过；CPU 测试只验证调用形状和拒绝规则。
- `load_sdf_arrays` 返回 owned `values/origin/spacing`，直接对应现有 `ropesim.sdf.SdfGrid` 构造输入。没有导入 ropesim、重新 bake 或新增 SDF。

原完整任务中的 gravity、table material、base DOFs 等 NA 保留在 `task_unknown_fields`。另行列出的 `candidate_physics` 来源是原 `flexible_pinch_candidate_v1.json`：P0 simulation coefficients，不是已测材料或已完成的三种 grasp task。手材料使用原 grip_compliant；桌面候选 .5/.4、restitution 0、SDK 默认 average combine；DeformX 的接触律与 PhysX pad compliance 不混同。

## 实际几何 / 原安全投影边界

`bind_actual_geometry(project, prepared, stage, hands, read_clock=...)` 是初始化绑定/读回检查，不应在每个控制步重复哈希全部源文件。它读取真实 `root_physx_view.prim_paths`、`link_paths` 和 `get_link_transforms`；只使用真实 actor-frame pose，把 SDK xyzw 转为原 kernel 的 wxyz，绝不从 USD world pose 猜测动态 link pose。两次共享时钟读取必须相等且 `clock_valid is True`；结果保存 owned arrays，没有持久 native view/hand owner。

结果包含：

- `body_poses_wxyz`：[1,2,23,7]，两手实际 23 个刚体（17 原 hand bodies + 6 carriage bodies）。
- `shapes`：34 个原格式 `(hand_index, actual_body_index, vertices_in_link_frame)`。
- `pad_bindings`：10 个左右独立 actual body index/path + source/cooked/SDF SHA，可接入独立双手杆组合层。
- `clearance_m` / `lowest_shape_index`：直接调用原 `collision_clearance`；此核的 .30 m 桌面与被 pin 的 task/table 匹配。
- `articulation_paths` 分别保存 actual articulation view path、authored articulation root path、hand container path。

必须区分 articulation root 与 hand container。本机 IsaacLab `articulation.py:1506–1547` 查找带 ArticulationRootAPI 的 prim，并将该路径传给 create_articulation_view；原 mount 将该 API 移到 `.../leftHand/wrist_fixed`。SDK 若返回这个 root 路径，不能拼接成 `wrist_fixed/base_link`。本包检查 actual view path 与对应 hand container/root 的关系，逐项核对所有真实 link_paths，并分别记录；CPU 测试覆盖 backend 返回 container 或 wrist_fixed 的两种明确形式。

`get_link_transforms` 的本机 API 文档说明：reset 写 joint state 后若没有 `update_articulations_kinematic`，link transforms 可能仍旧。外层 owner 必须先刷新，这个准备包不会偷偷执行刷新或增加 tick。实际 gravity/dt 检查结果也不会由 geometry packet 自动填成 true。

原 wrist safety projection 由根工作包单独复用 `palm_transport/per_hand_safe_z/compose_rotation`；本包不实现、绕过或替换它。34 个输入点集是 **原 USD convex envelopes**，只有十个 distal pad 有 locked F4 live cooked evidence；两手其余 **24 个非 pad collider** 的实时 cooked shape 仍未知。这里的 source clearance 不能冒充 34 个实时 cooked collider 的全部安全资格。

## CPU USD schema 与实际源码证明

普通 `scripts/python.sh` 的 USD 路径不含 PhysxSchema。为了调用原 mimic helper，本包仅加载本机 `omni.usd.schema.physx` 的 **USD schema metadata library**、Python schema binding 和 generatedSchema/plugInfo；这些文件均进入 preparation fingerprint。依赖库为 USD/Python/C++，不创建 Kit 或 PhysX simulation。

实际路径：`/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.usd.schema.physx`。CPU CLI 显式确认 `isaaclab.sim`、`omni.kit.app`、`omni.physx`、`ropesim`、`warp` 未加载；`pxr.PhysxSchema` 仅用于 USD schema authoring。

本机 PhysX Python API：
`/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py`（SHA `bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b`）：481 为 get_gravity，1185 为 get_link_transforms；root/link paths 通过真实 view property 读取。

本机 IsaacLab：
`/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py`（SHA `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309`）：1506–1547 为 root API 路径解析。原 material cfg 的 hard compliance 默认均为 0（`sim/spawners/materials/physics_materials_cfg.py:67,74`），离线材料 authoring 显式保持相同值。

## 验证

最终命令：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
bash scripts/python.sh -m pytest -q \
 tests/test_dh116_dual_hand_scene_preparation.py \
 tests/test_dh116_deformable_bimanual_step.py \
 tests/test_dual_hand_rod_runtime.py tests/test_dh116_palm_mount.py \
 tests/test_free_wrist_mount.py tests/test_dh116_contact_material.py \
 tests/test_dh116_batch_geometry.py tests/test_dh116_pinch_initial_table_clearance.py
```

最终实际结果：**146 passed in 141.81s**（新增 19 项，原相关范围 127 项）。无 native/GPU/训练。私有目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-hand-scene-cpu-2mia7vau`。

`final-cpu.log` SHA256：`671d68548e4fcc3a10fd2943a904070a518da13ca2e5fcf36e61af70f52691d6`；`final-cpu-manifest.json` SHA256：`452b229b1149325d59c838cb8a0c075c834917063182f48c17199387997788b6`。

CPU CLI：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
bash scripts/python.sh scripts/prepare_dh116_dual_hand_scene.py \
 --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-hand-scene-cpu-2mia7vau/final-preparation.json
```

CPU CLI 正常退出 0，34 collider / 10 软 pad / 12 wrist joints；native simulation modules 列表为空，initial_joint_state_applied=false。

Preparation fingerprint：`f03a84b42307f06d5d2a8fc9245a34db69cef7fe7aba631108d77db115099c79`（仅准备结果，不是 native recipe/批准 SHA）。`final-preparation.json` SHA256：`b6de00de1266fe645416dd47deb9f95a56f00cd66d36d07621c9515fc66ca844`；`final-preparation.log` SHA256：`fc2aaec7210affd3d9bc6aa39bd30f894162b271d0b59e90d895717576d9556a`。

新模块 SHA256：`c79ac1a5a0b9becd5d782708caa634398a249da5729432fd5ce0c1ae78b693fb`；CLI：`4e20338d7dbb716d294d75bc558aba30a4853c2d9d9a8d582fedb3a0eb8e237f`；测试：`71fe1ea01afe060a6f0ea4e396b62b133e0bd8c4b16f9a2e98e2327487c6988f`。

测试覆盖真实两侧 USD / SDF 字节、原 mount/mimic/material authoring、源属性被改动、triangle SHA 不变但 mesh-to-body frame 被改变、桌面尺寸改变、SDF 字节漂移、原 builder 的 4 cm offset 和 effort 默认差异、实际 view/link 行顺序与身份、clock 失效/变动、非有限 pose、owned readback，以及 container 与 articulation root 路径分离。CPU view fixtures 仅验证 native API 边界，不能生成原生接触、抓取、自然退出或 Teacher 资格。

开发期实际失败：原 USD orient 属性为 quatd，最初 float authoring 触发 precision 错误；纯 CPU schema 路径缺失使原 mimic import 失败；hard material 的 codeless compliance 默认没有自动读出，需要显式保持原 SDK 配置的 0。均已在独立源码修复，不修改原 USD/原 helper。

下一步依然需要独立 safe reset/approach 来源、原安全投影与 action/R0 owner 对接、真实两手 solver/contact/反作用力与自然退出检查，以及新 native 预算批准。本包不生成 native recipe，不提升任何 grasp 或 Teacher 资格。
