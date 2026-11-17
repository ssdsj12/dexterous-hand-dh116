# Tripod 两阶段起点与沉降闭合：独立 CPU 拒绝证据

本包没有找到可发布的 Tripod reset/closure 配方。新增价值是将两个阶段的限制分开后，确认左手存在通过原 2.1 mm 桌面门和原 17 source + 5 locked cooked 落杆包络的几何起点；但该起点的 SDF 连续无接触仍未知，而一条由实际节点 SDF 两尺度导数生成的合法闭合方向造成三处近端几何碰撞。右手旧末态自身亦有近端碰撞。没有把这些局部拒绝解释成全局无解。

隔离 worktree 基线为 baa4d5b165e7568fa6574aa6b892e941bc43931b。integration 在只读检查期间由主线程推进，本 worktree 按创建时 HEAD 分叉，之后未修改 integration。只新增本脚本、测试、报告；0 Isaac / GPU / 原生物理步 / PPO 或其他训练更新；没有配置变更、reset proposal、原生 recipe 或发布动作。

## 固定来源和有界设计

保持来自原 Tripod 独立左右 seed 的掌位、thumb-spread、ring/little 不变，不采用新固定底座 Pinch 的 pose 或 q。仅复用其通用 FK 与 contact.cuh 对应 float32 SDF 插值函数。

- 原 Tripod seed SHA：c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b。
- 父结果：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-joint-root-direction-20261113-3ar3y5k7/run01/result.json，SHA 726ca01dc554cb1a2a8a1ee03222c240118b536ae4cd7ad7348a19ca36fa39ba，来自既有 87a4e1b 工作包的独立关节/掌位解。
- 真实 1 mm SDF summary：db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab。十个左右 pad NPZ 的路径、SHA 与原 cooked capture SHA 全部保存在 result.sdf_bindings；原 USD/mesh 来源在 geometry_sources。
- 执行脚本 SHA：ecb6e63a228532e097f88cd36f3323c11ec0ac701e4018160b5af4e0eb60f117。九个直接实现/约束文件的执行时路径与 SHA 在 result.source_sha256。
- 原 0.3 rad/s、160/480 s 闭合时长、原 coupled limits、48 quiet steps 保留。此处 48 仅为离线计划字段，实际观测 quiet steps = 0。
- 2.1 mm 是原六轴策略投影门，1.9 mm 原静态门也单独保留说明；本包使用 2.1 mm，没有套用单手 Pinch 的 1.9 mm。
- 原 coarse 0.5 mm / strict 50 μm 及 opposed dot < -0.2 仍显式计算。100 μm 正 node-SDF 只是一次局部预测目标，不是物理 margin 或新准入门。
- 13 次三 pad SDF 查询（1 baseline + 两尺度三通道中心差分各 6）；仅一次方向、一个末态；65 均匀几何路径样本。没有反复求解或随机/网格搜索。
- 只读前查还检查了原末态/最大回退和三个局部回退量；右手恰在限位的浮点回退被原 expand 拒绝，因此没有绕过限位或修限位常量。最终包不沿此无效分支推进。

## 左手起点：几何通过，SDF 无接触未证明

旧 q 的 thumb/index/middle 各回退 0.030 rad，根姿态完全不变：

~~~json
{
  "independent_style": "tripod",
  "independent_seed_sha256": "c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b",
  "side": "left",
  "base_pose_wxyz": [
    0.5319283577621972,
    0.13489669694253587,
    0.3847989429225563,
    -0.40200419197817483,
    -0.42156529584810926,
    0.6125166379123426,
    0.5343207830966631
  ],
  "active_joint_names": [
    "finger11",
    "finger12",
    "finger21",
    "finger31",
    "finger41",
    "finger51"
  ],
  "initial_q_rad": [
    0.459811095111543,
    0.44832404757387223,
    0.702860623034901,
    0.7562387642089455,
    0.7,
    0.95
  ],
  "endpoint_q_rad": [
    0.459811095111543,
    0.5086194216117065,
    0.7523723443809228,
    0.7965988440762022,
    0.7,
    0.95
  ],
  "active_delta_rad": [
    0.0,
    0.06029537403783425,
    0.049511721346021775,
    0.04036007986725665,
    0.0,
    0.0
  ],
  "active_velocity_rad_s": [
    0.0,
    0.18088612211350275,
    0.14853516403806533,
    0.12108023960176995,
    0.0,
    0.0
  ],
  "quiet_steps": 48,
  "closure_steps": 160,
  "reset_verified": false,
  "native_ready": false,
  "scope": "offline_two_phase_hypothesis_not_reset_recipe"
}
~~~

该起点全部 22 形状的最小桌面间隙为 2.403619676 mm，完整 z=0.305–0.308 m 几何落杆包络的 surface-gap 支撑下界为 +0.471283397 mm。支撑平面下界是真正保守下界，未将 SLSQP 可行距离上界用于安全证明。

SDF 另使用所有潜在接触 cells 的 convex superset；对该 region 到原垂直轴线矩形计算支撑距离，不重复减 rod radius。五 pad 的下界依次为 +0.454702386、-0.002、-0.002、+16.210779463、+14.604290501 mm。index/middle 的 -2 μm 仅来自零支撑距离再扣数值 guard，说明这个粗 superset 无法证明无接触，**不是已证明有接触**。因此 left_quiet_start.static_conditions_passed=false，尚不称其为实际稳定或完整安全起点。

原 registered rod axis 是 0.308 m，半径 0.005 m、桌面 0.300 m，原初态杆底离桌 3 mm。0.305 m 仅为沉降后直杆的条件假设；没有修改构造/reset seed，也没有声称杆已受桌支持或 48 步内完成沉降。连续杆自由变形、实际支持和动态稳定均未观察。

## 一条闭合方向的实际拒绝

两尺度 Jacobian 相对差 0.0001867912，说明该局部导数可复现；不保证有限增量有效。相对旧末态额外闭合：
thumb 0.0302953740、index 0.0195117213、middle 0.0103600799 rad。
从回退起点开始的总速度为 0.1808861221、0.1485351640、0.1210802396 rad/s，均小于 0.3。

65 样本用原 per-link 最大点位移 B 扣 B/128，覆盖每个相邻样本间最近样本距离。整条固定根线性 q 路径的桌面连续下界 2.350836121 mm，保留 0.250836121 mm runtime 余量。然而末态三处非目标形状的 QP **可行距离上界**（已加 2 μm guard）仍为负：

| 原 source 近端 | surface-gap 上界 |
|---|---:|
| finger13_link | -0.319705558 mm |
| finger22_link | -0.070275584 mm |
| finger32_link | -0.831285971 mm |

这些负上界直接证明固定沉降直杆模型中的凸包重叠，独立于较保守的路径支撑下界。不能通过增加闭合量绕过原近端保护。

| 末态 pad | 连续凸包可行 gap | 实际节点 float32 SDF gap |
|---|---:|---:|
| thumb | +0.289636787 mm | +1.107518561 mm |
| index | +0.134526480 mm | +0.337532256 mm |
| middle | -1.095412061 mm | +0.188216567 mm |

节点间距 12.5 mm 与 1 mm 三线性 SDF 的定义不同于连续 capsule/凸包距离。所有三 pad 的 CPU contact predicate node 列表仍为空。middle 的连续重叠没有使实际节点 phi-R 为负；thumb 局部下降方向在这个有限步长反而由 +0.705545302 增至 +1.107518561 mm。这里只说明线性化失效和节点/连续几何差异，没有把唯一原因归到网格误差，也没有 GPU bitwise 或原生接触承诺。

thumb/index、thumb/middle 几何 dot 分别为 +0.281279788、-0.128024855，均不满足原 -0.2；原 coarse 和 strict 均 false。没有保存可授予资格的 candidate 配置，只在不可覆盖的结果中保存被拒绝的 hypothesis。

右手固定旧掌位/旧末态对沉降杆 source finger13 的可行 gap 为 -0.100468186 mm；加 guard 的上界仍为 -0.098468186 mm。两阶段时间拆分本身不能解除此末态重叠。本包未向右手复制左手方向。

## 验证及证据

先写测试得到缺失模块的预期 collection failure；新增最后一个身份负例最初错误期望 missing-file 抛 ValueError（实际 FileNotFoundError），改为真实 tampered 文件验证 SHA 拒绝，未修改被复用 loader。最终 **19 新测试 + 61 相关测试 = 80 passed，2.89 s**。两条既有 SciPy OptimizeWarning 说明 threads=2 被转交 HiGHS；没有测试失败。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_tripod_two_phase_sdf.py tests/test_dh116_tripod_sdf_constrained_direction.py tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_power_alternative_sdf_interval.py
~~~

实际一次完整执行：脚本计时 4.304906347 s；从 wrapper 子进程开始的外层 7.084840225 s，外层 timeout=60 s，退出码 0；内层 50 s SIGALRM 软限覆盖 run，导入启动在内层计时之外。超时测试证明 partial 被保存且资格保持 false。真实本次为 complete。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh scripts/audit_dh116_tripod_two_phase_sdf.py --output <new-exclusive-directory> --wall-seconds 50
~~~

证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-two-phase-sdf-20261113-4078wfvx

| 文件 | SHA256 |
|---|---|
| run01/result.json | 68b244ea5adee6dada5394b5e46e7e96270b12d3f9d6d1cfb24e8d2a9fad51a4 |
| outer.json | 20429f0971bc287e50a3afee8a0884eeea2e5fc65792c790a3f3641d3e2139c1 |
| offline-verify.py | 9baa0a16a8badda956ffc77bf399e7eab21158be5a54b14e2aded1169f039966 |
| offline-proof.json | 90bf9ccf48cdd1b4ba59e9520150c0ab2c59224d2cd2c2652b59a76ffbf973e5 |

离线证明核对九实现 SHA、十 NPZ SHA、父结果，重放 13 记录的方向计算及 65 记录的路径下界并检查负上界代数。它是存储记录和来源的复核，**不是独立重做 FK/QP/SDF**；新增这些几何查询均为 0，原结果不改。

下一项受约束假设应同时改变三指相对杆节点的轴向站位和 proximal/distal 几何关系，而不是继续同一根姿态下的闭合量。该左起点还需对固定节点沿 3 mm 垂直区间的原三线性 SDF 分段极值做更紧证明，以区分粗 region bound 未证明和实际可能接触。两者均未在本轮执行。双手配对、臂可达、自碰撞、12 个非 pad live cooked 和真实动态仍未知；不授予 reset、R0、Teacher 或 native 资格。
