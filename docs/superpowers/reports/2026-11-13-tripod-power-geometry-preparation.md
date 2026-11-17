# Tripod / Power 独立碰撞与接近准备

基线 `07db9eb461f8db067beb294c12c077d1ea1b824e`。本包只新增脚本、测试和报告。既有配置、USD、材料、驱动限位、原始证据、旧模型与 Checkpoint 均未修改；没有更新 runtime、注册表或发布脚本。无 Isaac、GPU solver、PPO、姿态 optimizer 或训练 optimizer 执行；复用原审计的 convex-QP 仅测量给定几何的距离，不调整任何姿态。候选生成是预先固定的有限枚举。

## 边界与证据

每侧读取原 DH116 USD 的 17 个 source convex envelopes 和 F4 锁定的 5 个 live cooked distal hull。12 个非 pad 尚无 live cooked 捕获，source envelope 不等同实际 cooked shape。复用 audit_dh116_pinch_initial_pose.py，SHA `15ede98e092a911470bf858f6d37e2a01007f05e3dbdb8ab2178ed95f6712db8`。

场景仅采用已锁定的直杆几何参考：41 节点，x=0.43 m、y∈[−0.25,0.25] m、初始中心 z=0.308 m、半径 5 mm、桌面 top=0.3 m。垂直几何包络覆盖中心 z∈[0.305,0.308] m。该包络与固定 z=0.305 m 的接近距离**不是实际自由杆 settling、变形或运动结果**。原完整柔性任务的材料/状态 NA 均保留。

所有关节状态经过原 expand()/active_limits() 的 source/manual/mimic 上下界检查。每个新接近通道的最大正向变化 0.1 rad，来自原 0.3 rad/s 上限乘 160/480 s；未提高速度、effort、stiffness 或 damping。闭合格点每通道固定 9 个，不是原生 action 或训练 rollout。实际关节跟踪、carriage/arm reach、contactOffset、self contact、Reset/371D 和 R0 仍未验证。

## Tripod：初态净空改善，接近仍阻断

唯一 seed 是自身 `configs/dh116_training/stereo_tripod_reset_20261105_v1.json`，SHA `c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b`；未借用 Pinch 姿态。原 seed 的左 finger22 与右 finger32 source 最低净空分别为 −5.856137 / −1.231838 mm。

每侧固定枚举 225 个候选：world roll −20/0/+20°，pitch −30/−15/0/+15/+30°，thumb/index/middle 同步初态偏移 0/0.05/0.10 rad，world-x 偏移 −4/−2/0/+2/+4 mm。每个候选根据全部 17 source + 5 cooked 的最低点解析平移 root-z，使初态最低桌面净空为 2 mm。每个必要 digit 测量 9 个原限位内闭合角度；距离与几何方向是当前 cooked hull 的结果，不是 SDK contact normal。

左右各检查排名前 3 条、每条 161 状态的完整 FK 路径。按原关节角变化及下游 lever-arm 上界扣除样本间位移余量；source 与 cooked 的桌面门槛分别检查，非 required collider 对杆包络也单独检查。静态选择优先保留初始 source/cooked 均不相交的候选，不用更小端点间隙掩盖坏初态。

粗端点筛选：left `0/225`，right `1/225`。最终两侧静态候选如下：

| side | candidate ID | 初始全部形状桌面净空 (mm) | 初始全部形状杆包络最小间隙 (mm) | 端点 required 最大绝对间隙 (mm) |
|---|---|---:|---:|---:|
| left | tripod-left-r20-p30-b0-x0.002 | 2.000000 | 1.952143 | 0.464932 |
| right | tripod-right-r0-p15-b0-x-0.002 | 2.000000 | 0.766918 | 0.105486 |

双手路径包络的 world-y 分离下界为 `268.044371` mm；因此该两组静态 root 在所检查的 table/rod/bilateral 几何范围内可供后续准备。输出 `bilateral_static_reset_geometry_passed=true` 仅表示这个 CPU 范围；`reset_is_native_verified=false`，不等于真实 Reset 资格。

具体尚缺项：
- left：连续桌面净空下界 1.982492 mm；source finger22_link 对杆包络间隙下界 -3.013530 mm，未通过接近净空门槛。thumb 对 index/middle 的几何方向点积为 {'finger23_link': 0.4498134691379312, 'finger33_link': -0.05031391312889142}。
- right：连续桌面净空下界 1.991246 mm；source finger13_link 对杆包络间隙下界 -0.563260 mm，未通过接近净空门槛。thumb 对 index/middle 的几何方向点积为 {'finger23_link': -0.6657383551066689, 'finger33_link': -0.777142696238648}。

左侧几何方向也未相向；右侧虽通过此粗几何方向筛选，仍没有通过近端杆包络净空检查。因此 `bilateral_geometry_preparation_passed=false`；该有限范围没有可晋升的 Tripod approach。原有两个初态碰桌问题已得到独立静态候选修正，但不能以此启动 Teacher PPO。

所选静态候选的可复核参数（wxyz；canonical active order 为 finger11, finger12, finger21, finger31, finger41, finger51）：

- left root：`[0.5340238162917201, 0.1349008281374014, 0.3872989429225563, -0.41612434118262115, -0.41571427246682197, 0.6208032735975652, 0.5182909142877746]`。
- left active q：`[0.509811095111543, 0.443364014422451, 0.6589761900657238, 0.7043766594952089, 0.7, 0.95]`。

- right root：`[0.5395238333891927, -0.1911292406242388, 0.3968990294601865, 0.4139243892693849, 0.611595998241508, -0.47347581629727525, -0.4800391507831093]`。
- right active q：`[0.14017710294697913, 0.03191588834411812, 0.6305627493793773, 0.6630411117967255, 0.95, 0.19999999999999998]`。

完整候选、9 点距离曲线、161 点路径检查摘要、SHA 和拒绝原因保留在私有 JSON；未生成新 runtime config 或 native recipe。

## Power：独立负档案的有限推进

读取原独立负档案 `e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39`，每侧 16 个原 root；再各构造一个全部 source/cooked 桌面最低净空为 2 mm 的垂直摆放，合计 32 条/侧。此 root 平移是**静态候选构造**，不代表已验证 wrist approach。

对固定 root、固定直杆参考以及原上限内的正向关节闭合，利用 distance 的 1-Lipschitz 性和全部 cooked 点位移上界计算各 required pad 距离下界，另扣 2 µm 数值余量。下界只是不相容性的必要条件筛选；下界≤阈值不能证明可达，真实自由杆可能移动或弯曲，因此也不能把固定参考的拒绝推成物理不可达结论。

- left：0/32 未被所有 required-pad 必要条件同时排除。最低桌面摆放中，最优 archive index 15 的最差 pad 间隙下界为 0.945248 mm。

- right：4/32 未被所有 required-pad 必要条件同时排除。最低桌面摆放中，最优 archive index 13 的最差 pad 间隙下界为 -6.256988 mm。

右侧仅有 4 条未被该必要条件排除：indices 12/13 已有 source/cooked 初始杆包络相交，直接拒绝；另 indices 1/15 各对五个必要 digit 测量 9 个闭合角度（保持该 root 与 thumb spread 固定）：

- index 1：required 最差绝对间隙 7.232210 mm；拒绝项 `['required_cooked_pad_gap_exceeds_0p5mm', 'required_geometric_directions_not_opposed']`。

- index 15：required 最差绝对间隙 11.584331 mm；拒绝项 `['endpoint_table_clearance_below_margin', 'required_cooked_pad_gap_exceeds_0p5mm', 'required_geometric_directions_not_opposed']`。

没有 Power 候选通过该有限接近筛选，也没有将旧 rigid wrap 资格或失败样本移入成功池。原 root 旋转之外的新方向、更复杂腕部接近、自由杆响应均不在此次搜索范围。所有 Power `native_ready / teacher_admitted` 保持 false。

## 验证与复现

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_flexible_style_pose_preparation.py tests/test_dh116_pinch_initial_table_clearance.py tests/test_dh116_flexible_power_pose_preflight.py
~~~

实际 **27 passed，0 failed，0 skipped，43.64 s，exit 0**。包括 7 项新增回归、13 项 exact-cooked/路径审计及 7 项 Power 独立档案验证。开发先观察新模块缺失失败，再实现；新增“初态清晰优先于坏初态的小端点 gap”回归也先观察失败。旧 tracked 文件相对基线 diff 为零。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/prepare_dh116_flexible_style_poses.py --style both --output /absolute/private/output.json
~~~

输出路径必须不存在；固定枚举完成，exit 0，0 physics tick / 0 PPO update。此命令成功表示筛选完成，实际 approach/Teacher 结果仍为 false。

私有证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-power-preparation-20261113-7zhi9kgm`。

| 产物 | SHA256 |
|---|---|
| preparation script | `041f203a229c614a0d4fa03e36ed54b158a270b74d2d4793f404fb6962c0c406` |
| final-cpu.log | `8914f1fa3bf2703712c7f0708776783309baf28fb40bcc616046508c94969127` |
| final-preparation.log | `b431aab5c31052d7d66a0872577684a42d824c50ea294ecfa2c43266b4e0e2a1` |
| final-preparation.json | `e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6` |
| 新测试 | `db321694606a7f3fed63c835a262337571040a53d59a209ca54955f8fddec777` |

下一步仍需在 CPU 候选中通过近端路径净空和正确几何方向；12 非 pad live cooked、实际同钟 Reset / 371D / R0、材料锚、solver normal、反作用力以及自然退出仍是独立门槛。本包没有请求或消费新的 native 授权，由 root 负责集成和发布。
