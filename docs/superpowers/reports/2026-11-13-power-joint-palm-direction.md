# Power：独立关节与掌位的一次有界方向验证

基线 87a4e1bd2da60f0f94e0c91b0730d5c792b6442d。本包仅新增独立脚本、CPU 测试和报告，不修改原 USD、配置、关节/物理阈值、原证据、runtime 或 R0。实际 90 次 residual geometry 查询（45/侧）、2 次 CPU 线性最小二乘；0 native physics、0 GPU、0 policy/training optimizer 更新。没有生成 reset proposal。

## 结论

**左右真实 Power seed 均给出可复现的局部下降方向，但最佳方向移动较大并违反硬几何门，两侧全部拒绝。** 本轮没有重扫旧网格，也未继续第二轮求解。这里的 objective 是几何残差平方和的一半，不是 R0 reward、实际接触成功率或环境就绪。

| side | 两步长 Jacobian 相对差 | 基线 objective | 最佳 objective | 下降 | root 位移范数 |
|---|---:|---:|---:|---:|---:|
| left | 0.050723301 | 211.877885 | 68.205515 | 67.8090% | 16.786606 mm |
| right | 0.048338667 | 100.187074 | 17.936167 | 82.0973% | 16.030257 mm |

两侧均选择实际回溯全步。线性预测 objective 分别 25.181374 / 4.799556，显著低于真实查询的结果，因此预测不能代替几何验收。左求解器 9 iter、右 11 iter；仅各调用一次，训练 optimizer 更新仍为 0。

左右起点来自自己真实 archive，而非镜像或借用 Tripod/Pinch。原 q 的全部旧 source/cooked 初始 rod-gap 记录非负，本轮解析非 required 包络下界也分别为 +4.527300 / +5.280320 mm，没有带着已知近端初始相交开始。两侧起点最低桌面净空均为 2.0 mm，满足原 1.9 mm 静态门，但对既有 2.1 mm runtime floor 仍差 0.1 mm；从未把起点称为 runtime 合格。

最佳候选的五个 required cooked pad gap 区间如下。上界来自可行 QP 距离加 2 µm，下界来自支撑平面减 2 µm，单位 mm：

| side | thumb | index | middle | ring | little |
|---|---:|---:|---:|---:|---:|
| left | [8.075008,8.081934] | [3.655681,3.659721] | [−2.392036,−2.373143] | [0.918814,0.987293] | [5.272263,5.276329] |
| right | [2.808863,2.812982] | [1.082352,1.093661] | [−0.976122,−0.968326] | [−0.169194,−0.144899] | [3.063250,3.067275] |

middle 的区间上界两侧均负，右 ring 也负，可确认本静态 hull/capsule 几何已重叠。与此区别，非 required source 包络的负下界只能表示净空未获证明：右 finger22 −2.250150 mm、finger42 −0.609341 mm、finger52 −0.267361 mm，不据此声称这些 shape 已发生实际相交。左非 required 最小下界 +1.073390 mm。

| side | 最低 source/cooked 桌面净空 | 对 1.9 mm 静态门余量 | 对 2.1 mm runtime 门余量 |
|---|---:|---:|---:|
| left | −1.141950 mm | −3.041950 mm | −3.241950 mm |
| right | +0.107475 mm | −1.792525 mm | −1.992525 mm |

左真实 transformed hull 顶点已经低于桌面，右虽仍在桌面之上却远低于两门。thumb 对 index/middle/ring/little 的几何方向 dot，左为 [−0.455350,−0.582739,−0.306928,−0.073374]，右为 [+0.043466,−0.908717,−0.701689,−0.008010]；原 <−0.2 的相向条件也未全部满足。这里仅是轴到 hull 的几何方向，不是 solver contact normal 或载荷/摩擦证明。

原 coarse 0.5 mm、strict 0.05 mm、1.9/2.1 mm 门全部保留。objective 下降仍可能使桌面/局部穿入变差，因为它是有限权重拟合；独立硬门在选择之后如实拒绝，没有以 objective 最小值授予资格。

## 来源与方法

父报告 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-power-preparation-20261113-7zhi9kgm/final-preparation.json，SHA e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6。Power 原 archive SHA e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39。

固定选择 left archive15 / right archive1，父模式 minimum_table_clearance_root，父标识 power-constraint-left-15 / power-constraint-right-1。中心用父原 active q，不用旧 closure endpoint、不增加其闭合 delta。原 Power load_selection 核父报告/helper/geometry source map，本包再核独立 style/side/index/archive SHA、原初始无负 gap、桌面门、原 joint limits 和 WXYZ。既有 archive12/13 的已知初始相交未被选入；先前固定根失败不是全局无解证明。

直接复用已集成 Tripod 的 local_direction、EvaluationBudget、envelope_support、endpoint_gates/pair_gates，并复用 Power 支撑平面。没有改这些实现。新增 Power 适配强制全部五 pad 区间、四个 thumb-partner 方向，不能退化成三 pad Tripod 门。

每侧 10 维：root xyz 平移 3、world-Y palm rotation 1、全部 6 个 active q。尺度为平移 15 mm/轴、转角 0.1 rad、每 joint 0.1 rad，归一化盒 x∈[−1,1]；q 同时受原 active_limits/expand 的 source/manual/mimic 限制。这是本次静态数值信赖区域，不增加任何 runtime 允许范围。15 mm 尺度参考上轮固定根 closure 的必要平移量级；该旧下界不适用于本轮同时改变 joint/rotation 的问题，不能当本轮必要性证明。

每次读取原 17 source + 5 locked cooked。五 required pad 的 QP 距离只作拟合估计及上界，解析支撑平面才提供下界。非 required 的 12 source shape 对完整 z=0.305..0.308 m 杆包络用 XZ 支撑方向，扣 2 µm；全部五 cooked 均为 required。桌面使用全部 22 个 shape 的最低顶点。残差为 5 gap（目标 +25 µm）、4 direction hinge、22 table hinge、12 非 required envelope hinge，共 43 项。未知方向保持 None 并使硬门拒绝，只有拟合项使用明确的无效惩罚。

每侧基线 1 + 两尺度 h=0.01/0.005 中央差分各 20 + 四档回溯 [1,0.5,0.25,0.125]，上限 45/侧、90 总。Jacobian Frobenius 相对变化≤0.2 才进入一次 lsq_linear，最多 40 iter、0.05I 正则。所有 attempts、完整 queries、两 Jacobian、线性预测和真实回溯结果保存；没有无界循环，也不沿最佳方向继续迭代。

## 具体方向与拒绝边界

相对各侧原 q/root，最佳全步：

- left root Δxyz=[+15.000000,+0.000067,−7.535923] mm，world-Y rotation +0.060694374 rad；六 joint Δ=[+0.051409808,−0.100000000,−0.100000000,−0.004990139,−0.040068185,−0.062651698] rad。
- right root Δxyz=[+7.663145,+1.505338,−13.999261] mm，world-Y rotation +0.100000000 rad；六 joint Δ=[+0.100000000,−0.016931439,+0.100000000,+0.056619351,+0.078761639,+0.100000000] rad。

active 顺序由原 ACTIVE 保存。左 root-X 和 thumb/index 的负向 joint delta、右 rotation 和 thumb-spread/index/little 的正向 delta 触及信赖盒边界；位移范数大于 15 mm 是每轴盒的结果，不是偷偷越过单轴上限。两侧均不能称为已找到小幅、安全修正。完整 root WXYZ/q 保存为 diagnostic-candidates.json，仅为被拒方向，非 reset proposal。

本轮仅是固定 reference rod 的端点诊断；未做连续路径、自碰撞、arm reach、mount dynamics、实际 reset 或双手接触求解。原 rod axis z=0.308 m、radius 5 mm、table top=0.300 m，注册初态底部仍有 3 mm 空隙。z=0.305 m 只作 settled 几何目标；没有改变原注册 seed，也没有声明原 R0/task reset 已获得桌面支撑。12 non-pad 的 live cooked、full 371D、接触/材料和原生生命周期仍未补齐。

本轮结论局限于两个明确初态和一次局部方向，不能推出 Power 全局无解。下一步若另获授权，应首先处理有限权重方向越过 table/required 接触边界的问题，例如保持可行性的约束步；本包不自动增加候选、改门或运行 native。

## 验证与可重放证据

新增 17 项 CPU 测试；相关回归 **60 passed，0 failed，0 skipped，5.64 s，exit 0**。先观察新增模块缺失失败，再实现。覆盖原 seed/SHA/index/初始相交拒绝、全部六 active 与原 limits、原 q 不混入 closure delta、五 pad/four direction 不得删减、unknown 拒绝、直接复用两尺度 45 次预算、父报告篡改、缺侧配对、独占输出和 timeout partial 的 completed/attempted 区分。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_power_joint_palm_direction.py tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_power_closure_constraints.py tests/test_dh116_flexible_style_pose_preparation.py
~~~

实际几何只运行一次：90/90，script 4.526791 s，outer 7.540647 s，exit 0。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh scripts/solve_dh116_power_joint_palm_direction.py --wall-seconds 50 --output /absolute/new/run01
~~~

50 s 软截止从输出目录创建后的 setup 开始，模块 import/parser 和收尾保存不在软计时中；60 s 外层 TERM 加 5 s 清理宽限覆盖进程，不把被外层终止称为成功。此次正常完成未触发截止。证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/power-joint-palm-direction-20261113-9dqvcyh8。

| 文件 | SHA256 |
|---|---|
| executed / final script | 24474e5c07860137f2123272b37d4f139a0bed7acd428a7d6601b508d2026ffb |
| run01/result.json | 9f27c11856b12ebdb8ec47f1be552255fd4fb4d2e1664516a2373e5173af9ef7 |
| run01/progress.jsonl | 8d3e545f9a39399c97c1cc3da7d6febbab6ad3674a3bfac9ba0780d3dd848bb1 |
| outer.json | 6720ddd0497ecd5ebcd685af29159e3c7ca962bae84c5d5f895afec705ac0f0d |
| diagnostic-candidates.json | 2078a8731ae5db3085e879d55f8c5de8ff834b14ff4def1b7d28630d2e046908 |
| offline-proof.json | 40732b7dd081f5b84cbe20beeef26f15c65cc72db0e1e3414732ffec75e99d4e |
| offline-verify.py | f02ca69f612325ba89182290b6d2002fc095820eea16d698fedcba85f980db8c |

独立 offline proof 没有新几何查询：核当前全部执行来源 SHA、90 参数重建、ledger 精确相等、4 个 Jacobian、8 次回溯 objective、原 q 中心、5-pad 结构、最佳方向和 outer exit 0。原结果不可覆盖。根代理负责集成发布，本工作树不 push。
