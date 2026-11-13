# Tripod：联合 palm root / 四 active 初态的一次有界方向验证

基线 9e3f24b20076010ce13ecc70ad8d050c2278bab0。本包只新增独立脚本、CPU 测试和报告；不修改原 USD、配置、关节限制、runtime/R0、旧证据或模型。实际 90 次 residual geometry 查询（45/侧）、2 次 CPU 线性最小二乘；0 native physics、0 GPU、0 policy/training optimizer 更新。没有生成 reset proposal，也不授予环境/native/Teacher 就绪。

## 结论

**已确认两侧有限差分都能提出可观测下降的具体联合方向，但一次方向验证没有得到合格 Tripod 候选。** 原固定网格没有重跑。两侧都从自己的原 Tripod lineage 出发，使用同一根冻结直杆，非镜像，非 Pinch seed。

| side | 两步长 Jacobian 相对差 | 基线 objective | 最佳 objective | 下降 | 最佳回溯 |
|---|---:|---:|---:|---:|---:|
| left | 0.001582820 | 10.754007 | 1.640229 | 84.7477% | 0.5 |
| right | 0.000075810 | 0.164980 | 0.045976 | 72.1326% | 0.5 |

这里 objective 是本包明确列出的几何残差平方和的一半，不是 R0 reward，也不是接触成功率。最小二乘预测与实际查询分开保存。左预测 objective=1.220102，右=0.000000101；右的全步实际反而升至 0.357727，说明不能把线性模型预测当真实几何通过。4 档实际回溯才选择了半步。

最佳候选仍被拒绝：

| side | thumb gap 区间 (mm) | index | middle | thumb→index / middle dot |
|---|---:|---:|---:|---|
| left | [0.669124,0.673763] | [0.870756,0.878370] | [−0.448905,−0.444603] | −0.050064 / −0.293628 |
| right | [0.145486,0.156365] | [0.030522,0.048221] | [−0.039487,0.010200] | −0.873846 / −0.883547 |

左 coarse 0.5 mm 和相向条件未全通过。右 coarse 通过，但 thumb 明确高于 strict 0.05 mm 门。几何方向只来自 convex 距离的轴到 hull 方向，不是 solver contact normal，也未证明载荷/摩擦/有效接触。

最佳全部 source/cooked 的端点桌面净空：left 4.282472 mm、right 2.417740 mm；对既有 2.1 mm runtime floor 的余量分别 +2.182472 / +0.317740 mm。原 1.9 mm 静态门、2.1 mm runtime 门和 0.5/0.05 mm coarse/strict 分开保留，没有放宽。

非 required rod-envelope 支撑下界仍有负值：左 finger22 −1.138111 mm、finger13 −0.239999 mm；右 finger13 −0.272916 mm。**负下界表示尚不能证明净空，不等于证明该新候选实际相交。** 本轮未为新候选追加这些 shape 的 QP 上界查询；已有端点/方向失败已足够拒绝。没有完整连续路径、自碰撞、arm reach 或真实 reset 验证，不能把端点桌面正余量提升为这些资格。

## 实现与复用边界

已先读项目现有 TripodCandidatePlanner / TripodNonlinearGeometry、HandPointFK、hand_clearance 和 arm_ik。前者明确固定腕、右手单控制窗口，不能直接承担本任务的联合 root / 双侧独立候选问题；arm_ik 属于 Panda/双点接触模型。本包复用原 USD/FK、完整 source/cooked hull 提取、原 convex 距离查询和已审查的 Power 支撑平面函数，只新增这一次方向验证的适配/预算。

父报告 SHA e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6，独立 Tripod seed SHA c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b。父候选为：

- left tripod-left-r20-p30-b0-x0.002。
- right tripod-right-r0-p15-b0-x-0.002。

局部中心采用各自父候选的已测三 digit 闭合端点 q=parent.initial_q+parent.active_delta；这是 CPU 几何中心，未声称该状态来自真实 SDK 或已装入 reset。ring/little 保持该侧原初态。父报告/helper/current geometry source map 由原 load_parents 严格核验，原 SHA/失败 lineage 不改。

每侧参数 x 为 10 维：root xyz、world rotvec xyz、finger11（thumb spread）、finger12（thumb）、finger21（index）、finger31（middle）。归一化盒 x∈[−1,1]，尺度分别为平移 5 mm/轴、转角 0.1 rad/轴、四 joint 0.1 rad/轴；这只是局部数值信赖区域，不是增加 runtime 允许范围。原 active_limits/expand 的 source/manual/mimic 限制同时收紧 q 边界。原完整手部姿态、arm/carriage reach 和 mounted 动态限制没有在此补造。

一次查询固定读取 17 source + 5 locked cooked：

- 必需三个 cooked pad 到 z=0.305 m 杆轴的 feasible QP 距离仅用于拟合残差；每个 pad 同时取解析支撑平面下界，减 radius 和 2 µm。QP 上端加 2 µm，strict/coarse 验收要求整个 gap 区间位于阈值内。没有用 SLSQP 收敛代替距离下界保证。
- 非 required source/cooked 对 z∈[0.305,0.308] 垂向杆包络，复用 XZ 支撑方向。由于 n_y=0，两个对角点与完整四角的最大投影完全相同；测试覆盖包络内部穿过 hull 的反例。下界再扣 2 µm。
- 17+5 个 shape 的最低桌面顶点、方向 dot 和原 active limits 独立保留。残差由 3 个 gap→+25 µm 目标、2 个方向 hinge、22 个 table hinge 和 16 个非 required rod hinge 构成，共 43 项。
- 缺失几何方向在测量中保留 None，门控拒绝；仅优化损失给予明确无效方向惩罚，不伪造正常方向。

每侧只做一个局部模型：基线 1 次，两种步长 h=0.01/0.005 的中央差分各 20 次。接受 Jacobian Frobenius 相对变化≤0.2 后，至多调用一次有界 lsq_linear（最多 40 iter，包含 0.05I 正则）；方向必须预测下降，才检查 [1,0.5,0.25,0.125] 四个回溯尺度。最多 45/侧、90 总，所有 evaluation attempts/完整记录写 ledger；没有无限求解或迭代到成功。

实际左 linear solver 12 次迭代、右 0 次内部迭代（初始无约束解已足够），但各调用 1 次求解器，不能报告“0 optimizer”掩盖 CPU IK 求解；明确仅训练 optimizer 更新数为 0。

## 保存的具体方向与未知

相对各自 parent endpoint，最佳半步：

- left root Δxyz=[−2.095459,−0.004131,−2.500000] mm；world rotvec=[0.0451062,−0.00337282,−0.0138244] rad；四 joint Δ=[−0.05,−0.0150400,−0.0136156,+0.00686210] rad。
- right root Δxyz=[−0.277314,约0,−0.313248] mm；world rotvec=[0.00468373,−0.00184727,−0.00154075] rad；四 joint Δ=[+0.00224121,−0.00688672,−0.00386337,−0.00163782] rad。

完整 root WXYZ、canonical q、上下界、J、方向、4 个真实回溯结果、所有拒绝项保存在 result.json/diagnostic-candidates.json。后者明确是 IK direction diagnostic，不是 static reset proposal。左右仅在各自全部 scoped endpoint 过门时才会计算正的 world-y 配对条件；本轮两侧未过，配对 false、连续路径 false。

原杆 axis z=0.308 m，radius=5 mm，桌面 top=0.300 m，注册初态底部仍高 3 mm。endpoint 以已使用的 settled axis z=0.305 m 为几何目标，非 required 包络覆盖两者；未改 rod seed，也未声称 R0/source alignment 或真实支撑已实现。12 non-pad live cooked、自碰撞、full 371D、材料/载荷和自然退出仍是其它门槛。下一步若需要，应由本方向的 Jacobian/阻断分量提出新问题；本包不自动再迭代、扩大预算或运行 native。

## 验证和证据

新增 **15** 项，相关回归 **59 passed，0 failed，0 skipped，4.96 s，exit 0**。先观察模块缺失失败；随后 unknown direction/seed 重标记两项负例先失败后修复。覆盖变量盒/原关节限制、2 尺度稳定性、非线性实际回溯、45 次上限、unknown/strict/runtime 门、垂向包络内部反例、缺侧配对拒绝、partial 已完成与attempted计数、独占输出。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_tripod_proximal_screen.py tests/test_dh116_power_closure_constraints.py tests/test_dh116_flexible_style_pose_preparation.py
~~~

实际几何只运行一次：90/90 查询，脚本 4.192562 s、外层 6.883929 s、exit 0。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh scripts/solve_dh116_tripod_joint_root_direction.py --wall-seconds 50 --output /absolute/new/run01
~~~

50 s 软截止从输出创建后的 setup 开始，模块 import/parser 和收尾保存不在该软计时中；60 s 外层 TERM 加 5 s 清理宽限覆盖进程，不声称到限必然正常成功。所有查询 CPU 2 线程，原始结果不覆盖。证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-joint-root-direction-20261113-3ar3y5k7。

| 证据 | SHA256 |
|---|---|
| executed / final source | cb01431d253eddd720263462465f2ae4eab0e7f32bf2e37fe02d8701f1775359 |
| run01/result.json | 726ca01dc554cb1a2a8a1ee03222c240118b536ae4cd7ad7348a19ca36fa39ba |
| run01/progress.jsonl | 7289a831dd026c160df4968d6f0c321ea7e2571c1539831a26b5a23cf623a575 |
| outer.json | d8a8381bfd104d777153d8e1763fa3703bd6e5fc412196f01c8205d1abaf9410 |
| diagnostic-candidates.json | 2eb8c4ab6a27383f81eb00c61f095a849d275cdba90012160f41b9a32a43b72d |
| offline-proof.json | d53198db67ccd92fd41c888c7b8b011d211190ad70b5bd69dd60bff15387650d |
| offline-verify-v2.py | a8058c14ac8730cbd1caac5843160982822db7c21d0769261f0182f7a1eea036 |

离线 proof 0 新几何查询：核所有执行来源 SHA、90 参数重建/ledger精确相等、4 个 Jacobian 从既有行重算、8 次回溯objective、最佳候选及外层退出。首次独立 verifier 将 list 直接 abs 导致 TypeError，在任何 proof/候选输出前失败，脚本保留；v2 只修成 NumPy array 后完成，无原几何重跑。根代理负责集成发布，本工作树不 push。
