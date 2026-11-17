# 柔性逐子步方向与材料站位 source basis

日期：2026-11-13。基线 `1ba8dfd075520bdb7bd81ebbaa8fdce0dc860629`。
独立工作树 `flexible-pair-source-basis-1ba8dfd`。本包只有 CPU 数学、已有原生文件只读重算及测试；0 新 native / GPU / optimizer 步。

## 可执行接口与边界

新增 `tabletop_deformable_pair_source_basis.py`：

- `analyze_pair_capture(raw, mesh_names=..., rest_lengths_m=..., rest_source=..., radius_sample=..., capture_source=...)` 复用原 `validate_pair_capture`，先重开所有 raw 子步/节点/mesh 向量，核独立节点合力和逐 mesh wrench 的步均值，再做方向计算。source 与 mesh labels 是明确的 caller provenance，不自行成为实际 SDK 路径证明。
- `analyze_rod_pair_basis(row, grasp=..., radius_sample=...)` 直接消费当前双手 runtime 行，复用原 `summarize_contact_pairs` 校验固定 11 slots 与 pair/rod 同步边界，保存原 counts。Pinch / Tripod / Power 接口均以实际 runtime 代码和 CPU solver doubles 测试，不改原 gate。
- `material_station_resolution(rest_lengths_m, source=...)` 给出真实输入的 rest station、长度和离散间距；没有隐式材料/滑移值。
- `verify_contact_source(path)` 只读匹配本版本精确 contact.cuh SHA，返回坐标/方向定义及源码位置，不导入或执行扩展。
- `scripts/audit_dh116_flexible_pair_basis.py --input <JSON> --output <新文件>` 可从磁盘独立重算上述 raw 接口，记录输入与实现 SHA；已有输出或 symlink 均拒绝。实际子进程测试确认 `native_imports=[]`。

未修改 R0 replay、旧 task、原成功阈值、材料、物理、配置、registry 或 native recipe。所有输出 `effective_opposed_contact`、`material_anchor_slip`、`approach_distance_m` 保持 None；`native_qualified/teacher_admission=False`。

## 实际源码的方向定义

编译树 header：

`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source/third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh`

SHA256：`8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb`。
旧 F7 v2 扩展文件 SHA：`92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0`。本包只核文件，不重新构建、加载或运行二进制；source 与 loaded-binary 的历史关联来自原校准证据。

- `sdfContactAt` 94–108 行：`n = q * (g / |g|)`，所以 n 是**障碍物 SDF 向外**的世界 normal，不是杆外法向。
- `surfaceContact` 27–45 行：`N = -k*gap`，作用于杆的力为 `N*n` 加 regularized Coulomb 摩擦项。
- `surfaceReadout` 165–206 行：在子步 final iterate 读出，接触点 `p = pos + dx - R*n`，即节点球面点；力为杆侧 `fm`，mesh 反力为 `-fm`，力矩为 `cross(p - mesh_origin, -fm)`。
- patch 记录 `(substep, node_index, mesh_index, p, n, fm, torque)`。这个 node_index 是离散顶点身份，不是亚节点连续材料锚。

因此本包不会直接把 normal 当 rod outward normal，也不会通过 mesh 净 wrench 点积推断接触对。

## 可以复核的数学量

只比较同一个 solver substep、同一侧手、不同原 pad mesh 的接触。table 独立计数，左右手和不同子步不拼接。不同 node 的比较保留各自 station 与差值，不称为同一横截面接触。

对两条真实 pair，`d = p_b - p_a`、`u = d/|d|`：

- 原 normal 点积 `n_a·n_b`。
- 原杆侧 force cosine `(f_a/|f_a|)·(f_b/|f_b|)`。
- 向内 normal 投影 `[n_a·u, -n_b·u]`，以及杆侧力投影 `[f_a·u, -f_b·u]`。normal 由原 validator 在 2e-5 容差内检查为单位向量；这些投影保留原读回数值。
- `geometric_opposition_sign_candidate` 仅表示上述两个方向点积为负、两个向内投影均为正，且两侧法向载荷超过已有的 `1e-5 N` 数值噪声门。没有新增抓取力阈值。它不是有效对向接触、force closure、物理抓取或 R0 的 opposed 来源。
- 重合接触点没有 chord 方向；零力没有 force cosine。相关结果保持 None，不补零方向或伪造 negative。低于数值载荷门的 candidate 也保持 None，原向量仍保存。
- 若明确提供来源可追的 radius，且是同一个 node/substep，还可计算 `|(p_a+R*n_a)-(p_b+R*n_b)|`。这是记录点是否对应同一节点球心的几何残差，不是 live 物理材料锚误差，也没有被设成 admission 阈值。
- 数值方向误差界仍未知。上述向量运算不需要猜材料参数；其解释为有效夹持则仍缺接触几何/误差、摩擦/力学标准、材料锚和实际稳定性证据。

CPU 比较上限默认 65536。先按 mesh 组大小用 `(N² - sum n_mesh²)/2` 的整数公式计算数量，超限即拒绝完整包。只有通过预算后才对不同 mesh 组做 Cartesian product；大量同 pad 行不会触发隐藏的 O(N²) 空比较或截断证据。

## 材料离散界与现有实际证据

令 rest stations 为 `s_0=0`、`s_i=sum(rest[:i])`、最大间距 `h_max`。任意 `s∈[0,L]` 到其最近 station 的距离至多 `h_max/2`，最坏值位于最长 segment 中点。这只是“把连续站位分配到最近离散 station”的条件数学界。

当前 solver 在节点球上读接触，**没有实施或测量这个连续 contact → 最近 station 的映射**。因此这个界不能赋给 capture 的物理 material anchor 或 tangential slip；same node / same station 也不能推出 slip=0。两种物理误差界继续 None。

只读重算了以下旧原件，没有重跑：

| 原件 | 实际可确认的结果 | 不可确认的内容 |
|---|---|---|
| F7 v2 原单 table 校准 | 原 v2 artifact validator 通过；3 个原生 pair，全部 table，独立 CLI exit 0 | 0 个手侧 comparison；opposed/slip 均 NA |
| 失败 F4 v1 的 actual reset readback | 41 节点，rest 总长 0.5 m；最大 gap 0.012500002980232239 m | 唯一 physics 行只有节点 aggregate/mesh wrench，无 pair，不能恢复手侧方向 |
| 最新已批准 Pinch v2 旧输出 | SimulationContext 初始化失败，未产生 physics/pair 测量行 | 没有可用于本包方向计算的新手侧 native 数据 |

F7 输入 rest lengths 来自原校准函数 AST 的 authored nodes，radius 来自其 authored set_contact 参数；明确不是 native getter/readback，不与 F4 的杆混合。F4 的 rest lengths 则来自其实际 reset 记录，episode UID、SDK 2、rod tick 0 与原文件 SHA 一同保存。

F4 最大 gap 对应条件量化界 **0.006250001490116119 m**，是原 R0 `max_slip_m=0.0005` 的 **12.500002980232239 倍**。这是离散分辨率限制，不是测量到 6.25 mm 的滑移，也不能作为真实误差的上界。physical material-anchor / tangential-slip error bound 均保持 None。

approach 也未补值：pair 只有 penetration 时才出现，不能给尚未接触的 pad→目标材料锚距离；原 R0 首条 progress 又缺少前值。接触后的 chord 距离不替代 approach distance。

## 测试与独立审查

最终 Ubuntu：**162 passed in 26.84s**，其中 26 新增、136 既有，9 文件，exit 0。环境为 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1`，执行 `bash scripts/python.sh -m pytest -q`：

- `tests/test_deformable_pair_source_basis.py`
- `tests/test_dh116_flexible_pair_contact_diagnostics.py`
- `tests/test_dh116_deformx_pair_capture.py`
- `tests/test_dual_hand_rod_runtime.py`
- `tests/test_deformable_r0_replay.py`
- `tests/test_dh116_flexible_rod_boundary.py`
- `tests/test_dh116_flexible_physical_exchange.py`
- `tests/test_dh116_deformable_grasp_task.py`
- `tests/test_dh116_deformable_control_step.py`

新增测试覆盖明确向内/反向/平行/零力/重合点、同/异节点、table/跨子步排除、原 raw force reconcile、重复/overflow/错 identity、源 SHA 篡改、非均匀 station 的独立 dense oracle、刚体坐标变换、原双手 runtime 三抓法接线，以及实际 CLI 子进程和输出不覆盖。首次缺模块和缺 wrapper 分别产生 collection error、4 failed 后完成实现。

wireseg 只读数学/NA审查未发现实质数学阻断；其发现输出枚举会绕过比较预算。1000 节点同 pad 的监视器测试先失败（red log 已保存），再改为 mesh-group product，复核关闭。peer 没有运行测试/native；后加的 runtime wrapper 没有计入其完整审查范围。

## 原始证据和剩余门槛

私有证据根：

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz`

含 `final-cpu.log/json`、`red-bound.log`、`f7-native-input.json`、`f7-native-replay.json`、CLI stdout，以及 `actual-source-audit.json`。后者记录全部原文件/源码/扩展 SHA、F4 实际 reset identity、F7 旧原生校验结果、明确的 authored/readback 区别和零新增原生步骤。

本包可作为当前 pair row 的独立数值诊断使用。它不给原 R0 contact/opposed/slip 赋值，不改变全项 valid 规则，不把原生 table 标定推广成实际手抓取，更不解除 Teacher PPO 的 source/reset/reward/native 门槛。

## SHA256

- `src/go2_pvcnn/control/dh116/tabletop_deformable_pair_source_basis.py`: `da6e1c3a55757c6a5884f2d1f6cae656ad6b23cdb43af71c7bfb4b85f1ba23db`
- `scripts/audit_dh116_flexible_pair_basis.py`: `5e14f7ac7e30fbbdf980ba3abf85f4f856e4f77f58bc1e2aadccf482dbc5c05f`
- `tests/test_deformable_pair_source_basis.py`: `d260eb8bb4520e46b4719b9ca2e37cd5f85470cdb8a4065c7dc61d219c7981a6`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz/final-cpu.log`: `7032d25967c976e3fb0e0dadd9b122318fc34d114e4014f6c2694a34d8331eb4`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz/final-cpu.json`: `e45dd6c8abf8eef02d1336e3c42c001edc9e2029f64f98d0d1e5d3ef79e7e747`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz/actual-source-audit.json`: `a92102ffd70272e297ce718b2d60cae487022fb5789b09b4e116b8a849dc392f`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz/f7-native-replay.json`: `b10ba50f0aeeca762246bb137d4fa517d5ab598a6dbbca3fedbdfe49a93fe4c6`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-pair-source-basis-cpu-4uwq54tz/red-bound.log`: `90473b969b66cf79c14a65481cc4d83ad0b4e8620e05acfdfd31e87bf1aa3652`
