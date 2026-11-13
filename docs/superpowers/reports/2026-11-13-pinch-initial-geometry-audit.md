# 自由杆 Pinch 起始几何审计与独立姿态候选

基线 `7e98f864fc09aec09e1500a5359d994846c25850`。本包仅新增 CPU 审计、独立姿态 JSON、回归和本报告。未修改旧配方、旧任务、USD、材料、控制器、Checkpoint、注册表或 README；未启动 Isaac / CUDA 求解器，未创建新 native recipe。后续由 root 集成、发布。

## 真实失败首帧与 cooked hull

输入为失败运行 `approved-c5-pinch-20261112-v9l5ajv3/native-runs/flex.f4-free-pinch-left-candidate-native/attempt-crp1pahm/native` 的原始 plan.json、唯一 physics.jsonl 和 result.json，逐文件校验固定 SHA。该行是 SDK tick 3 的测量，pre_state 对应 tick 2；保留原 env / episode / reset / rod / policy / substep 身份。原 joint_names 未保存，因此不猜 SDK 11 关节排序。

五个 pad 读取锁定 F4 `flex-f4-parity-prep-Vu9H0f/gpu` 的 **live GPU PhysX cooked hull**，校验锁文件、原生正常退出、CPU/GPU 拓扑 parity、参考坐标标定、原 USD geometry SHA，以及 mesh/body 基底重合。按原始 pre_state.link_pos/link_quat 变换。表为原 layout 有限盒体，碰撞使用 convex-polyhedron/AABB separating-axis test，包含 edge/face 交叉，不依赖“顶点在盒内”启发式。

| cooked pad | 最低 z − table top (mm) | 有限桌盒相交 |
|---|---:|---|
| finger14_link | 2.400811 | False |
| finger23_link | 0.006333 | False |
| finger33_link | -16.893860 | True |
| finger43_link | -41.096925 | True |
| finger53_link | -63.222302 | True |

严格 cooked 证据为后三指相交；食指仅约 6.3 µm 正间隙。旧 source triangle 的 −0.039 mm 与此不同，不能混用几何类型。原 USD source convex envelope 还显示 base 最低 −62.926 mm、finger11 / finger12 分别约 −12.391 / −12.239 mm，因此只检查 distal pads 不足。

同一原 USD joint frame FK 对五个真实 link 的最大位置误差 `6.15109786651e-08` m，旋转矩阵最大元素差 `8.46855844922e-07`。全部 17 个原 enabled collider 的 source convex envelope 均单独检查；另 12 个非 pad **没有 live cooked 捕获**，不将其源凸包称为真实 PhysX cooked geometry。此审计不证明 mimic_error 的冲量来源或特定关节根因。

完整可复核变换链写入最终 JSON 的 `failed_initial_pose.source_transform_chain`：world/base authored pose、六 active 与十一 canonical expanded 关节、每个原 USD joint 的 parent/child frame 和轴、17 个 source FK world/link 矩阵、每个 collision mesh 的 mesh_to_body/owner/path/geometry SHA。矩阵按 column-vector convention。真实五 pad 的 world/link 矩阵单列在 `actual_pre_state.T_world_link_column_vector`，不将 source root 命令冒充 native root readback。

`failed_initial_pose.joint_readback` 保留完整 pre/post q 与 qd 数组。最大单步 |Δq| 为 `0.480955138803` rad，post |qd| 峰值 `85.4345703125` rad/s；joint_names=null，order_status 明确禁止推断，不将此异常指定给未知排序的关节。

## 独立修正候选

`configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json` 只提供新的左手固定根姿态与显式命名六关节初态；中、无名、小指选择 1.15 rad 收拢。原杆 41 节点、位置、半径 5 mm、桌面、重力、free ends、原材料与驱动限值保持原候选含义。P0 EA/EI 仍是 calibration，未变成测量材料。

闭合仍限制 thumb/index 两通道，各 `0.075 rad/s × 160/480 s = 0.025 rad`，原 active velocity limit 为 0.3 rad/s。expand() 对所有状态执行 coupled / manual / source joint 上下界检查。这里只计算 commanded FK，不证明实际跟踪。

连续杆距离使用有限 segment/convex-solid convex QP，不以离散杆节点或 pad 中心替代。另检查中心线 z 从 0.308 m 到 table_top + radius = 0.305 m 的连续垂直几何包络；不把此包络称为动力学 settling。161 个目标状态之外，以每个源关节的角变化乘下游 lever-arm 上界，计算最近样本之间的位移界。五个真实 cooked pad 和全部 17 source collider 分别应用桌面路径门槛；12 非 pad 对杆包络的距离界另减 2 µm 数值余量。显式 1.9 mm 桌面筛选门槛不是 PhysX contactOffset 的实测值。

最终 CPU 数值：

- 全部 17 source collider 连续桌面间隙下界：`1.991094534` mm。
- 五个 locked cooked pad 连续桌面间隙下界：`2.979065440` mm。
- 12 非 pad 对杆垂直包络的连续 source 几何间隙下界：`0.139786222` mm。
- 端点 thumb/index cooked hull 对 z=0.305 m 杆表面的间隙：`0.079503` / `0.417617` µm。
- 两个最近几何方向的点积：`-0.801774208`。这是 CPU hull 最近方向；effective_opposed_contact=null，不是 solver normal 或有效抓取。
- source 与 cooked 的初始杆包络间隙均非负；geometry_candidate_passed=true，native_recipe_ready=false，grasp_qualified=false。

负回归专门将 cooked 中指移入桌面而保持 source 全部通过：修复前该例错误通过 geometry gate（1 failed），补齐独立 cooked 门槛后通过。失败日志 SHA `a5c80bafafaa91fc20434a026cc4479818448a6546ce8e5e840ebb078ec0a9d9`。

仍需真实同钟 reset / 全 body pose / joint_names / collider / contactOffset 读回，12 非 pad live cooked hull，以及实际动作跟踪、接触、反作用力与自然退出。源 USD 所检查的 base、finger13、thumb、index collision owner 未 authored contactOffset/restOffset；此运行也未记录其 native effective 值。约 0.140 mm 的源近端余量不等于已证明原生安全。原完整 371D、材料锚、有效相向接触、Teacher admission 继续未获资格。

## Tripod / Power 的独立入口

load_geometry(project, side=...)、world_vertices(g, pose7wxyz, active6, cooked=...)、shape_metrics(shapes, layout, radius) 与 settling_gap(vertices, layout, radius) 可读独立样本。audit_style_seed 只消费明确的 canonical 11-joint order、双手各自 root 和 SHA；缺独立 seed 直接 blocked，不回退 Pinch。

Tripod 读取其自身 stereo_tripod_reset_20261105_v1.json（SHA `c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b`），未复用 Pinch：
- left 最低 source collider `finger22_link`：`-5.856137` mm；独立 approach 未提供。
- right 最低 source collider `finger32_link`：`-1.231838` mm；独立 approach 未提供。

Tripod 返回 blocked，CLI exit 2；旧 rigid seed 不晋升自由杆资格。Power 归档候选筛选由 root 独立包处理，本包不发明安全姿态。

## 验证与复现

最终命令（环境 DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1）：

~~~bash
bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_initial_table_clearance.py tests/test_dh116_pinch_scheduler.py tests/test_deformable_grasp_runtime.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_contact_material.py tests/test_dh116_work_packages.py
~~~

实际 **169 passed in 72.27s (0:01:12)，0 failed，0 skipped，exit 0**；其中 13 项新几何测试及 156 项既有相关回归。测试完成后既有 tracked 文件相对基线 diff 为零。

~~~bash
bash scripts/python.sh scripts/audit_dh116_pinch_initial_pose.py --candidate configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json
bash scripts/python.sh scripts/audit_dh116_pinch_initial_pose.py --style tripod --style-seed configs/dh116_training/stereo_tripod_reset_20261105_v1.json
~~~

第一个 CLI exit 0，第二个因独立 seed 不满足门槛 exit 2；均没有启动 Isaac。日志与几何 JSON 留在 Ubuntu 私有目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-initial-geometry-cpu`，不提交原始数据或轨迹。

| 产物 | SHA256 |
|---|---|
| audit script | `15ede98e092a911470bf858f6d37e2a01007f05e3dbdb8ab2178ed95f6712db8` |
| 独立 pose candidate | `a2eefb1a8f487fdba7e567c51aab005211dec3f658d2aa939c0ff7289de0dc44` |
| 新 test | `3a15312453569a624105de20a36b938df800271200fac7c29354407e4bba85c5` |
| final-chain-cpu.log | `65e98a1e906683c2a3ee1ae81edb3b8357de9321304febff242c249fd3d6bed2` |
| final-chain-audit.json | `e4fac4c777c5f8e3ece6baf310026f9f051381e473be21f2037d207ede01a4e4` |
| tripod-audit.json | `1b2627a30d9f28c2cb0b531adbfef88efe77604df668a0a920507a30dcf0eabb` |

仅提交四个新文件：audit script、pose candidate、test、本报告。无 GPU / Isaac / PPO tick，无新原生授权消费。
