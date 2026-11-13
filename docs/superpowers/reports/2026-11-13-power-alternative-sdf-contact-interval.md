# Power：未重复原姿态的安全起点与 SDF 闭合区间

基线 9a8d0ac59d6ede0e7dcd825003a335c76be07e0e。只新增独立 CPU 脚本、测试及报告，integration 和原 USD、SDF、材料、关节限制、动作时间、原配置及旧证据不改。0 native、0 GPU、0 训练更新；没有 reset proposal、Teacher 或原生资格。

## 与 7bec943 的差异和结果

7bec943 已验证 left archive15 / right archive1 的掌位与关节方向，结果被桌面和目标 pad 重叠拒绝。本轮不重复这两个 seed、90 residual 或旧五根九角检查。只读既有完整初态记录，在未做 closure、原 source/cooked 不相交的候选中，按五 pad 初始最大 gap 最小排序，选择 **left archive6 / right archive6**。它们各自保留原 Power 四元数、xy 和六 active q，非镜像、非 Pinch/Tripod seed。

原 minimum_table_clearance_root 为全 22 shape 桌面净空 2.0 mm；仅将 z 提高 0.3 mm，使其达到 **2.3 mm**，保留原 runtime 2.1 mm 门并有 0.2 mm 静态余量。该高度是新的离线准备，未写入真实 reset。

| side | 全 17 source+5 cooked 初始桌面最低值 | 初始所有 shape 对杆包络下界 | 原 SDF 接触区域对 quiet 包络下界 |
|---|---:|---:|---:|
| left archive6 | 2.300000 mm | +1.318331 mm | +0.337799 mm |
| right archive6 | 2.300000 mm | +3.457768 mm | +3.594387 mm |

所以这两种原 Power orientation 可以形成**满足本轮静态起点条件**的替代姿态。这里 quiet 包络只是原直杆轴高 z∈[0.305,0.308] m 的几何集合，不是实际沉降/稳定性验证。

但分开安全起点和合法闭合后，固定 root、原 thumb-spread 不动、其余五指各自原 0..0.1 rad 范围内，仍不能让全部五 pad 接近该固定杆轴的原 SDF 接触区域：

| side | thumb | index | middle | ring | little |
|---|---:|---:|---:|---:|---:|
| left 连续下界 | **+12.886544 mm** | −0.670111 mm | −0.939548 mm | **+5.277183 mm** | **+14.705203 mm** |
| right 连续下界 | **+7.240898 mm** | **+1.340779 mm** | −0.939548 mm | **+5.984675 mm** | **+8.647754 mm** |

正值证明该 pad 在本固定直杆轴/固定根/完整闭合区间下，不能达到原 SDF 的接触谓词；负下界仅表示无法排除，不表示接触。左至少三指、右至少四指被排除，五 pad 条件不可能在这个明确的区间内同时满足。**不宣称 Power 全局无解，不排除其它姿态、掌位路径或杆发生真实变形后的接触。**

## 原 SDF 的连续排除证明

这是本轮新增的方法，不是再次拟合凸包距离：

1. 原每个 NPZ 的节点值和 1 mm spacing 只读。cell 内理想三线性 phi 是 8 个角值的凸组合。因此 phi<R 的点，必定位于至少一个角值<R 的 cell 内。
2. 选所有 min(8 corners)<R+2 µm 的 cell，将其全部角点取凸包。它是可能 SDF 接触区域的保守上包络；包括非接触空隙也只会使排除更难。R 仍是原 5 mm，2 µm 仅为本证明的保守数值 guard，未改变 solver 参数。
3. 该区域随原命名 link 作刚体 FK。用已有范数≤1支撑方向，求区域到固定有限杆轴的距离下界。**区域已经包含 radius 的影响，不能再次减 R**；实现只读取 support witness 的 distance_lower_bound_m，测试覆盖重复减半径的错误。
4. 每个 digit 的 0..0.1 rad 区间取 9 个均匀角点。复用原 joint-frame/lever-arm point travel 上界 B；任意中间状态距最近样本的路径比例≤1/16。连续下界为 min(sample support lower)−B/16−2 µm。正值才能连续排除。
5. 因为固定连续杆轴包含所有 41 个节点，区域与轴无交集也排除了所有节点 SDF 接触。该证明不依赖九个节点采样结果都是正值。域外 phi 保持 None；没有用 0 或大常数填补，再称为“安全距离”。

原接触仍要求 phi−R<0 且梯度非零；接触区域包络只用必要条件，忽略梯度会扩大包络而不会漏掉可能接触。上述集合与插值论证对应锁定网格的数学三线性场，保留数值 guard；CPU float32 查询对照原采样公式，不作 GPU bitwise 或所有未来动态舍入的正式误差保证。

每侧五个 region 均从自己的原 NPZ/cooked capture 重建，包含约 13,915–19,052 个 potential cells，每个凸包约 241–261 顶点。这个分析区域**不是新增碰撞体**，没有写入 scene 或 solver。

## 实际节点查询和闭合预算

每 digit 9 个角点，共 2×5×9=90 次单 pad FK/SDF 查询，没有随机候选、反复求解或扩大范围。thumb-spread 保持该侧原 q；每个 digit 都使用原 active_limits/expand，最大 0.1 rad 来自 0.3 rad/s ×160/480 s。主程序未运行任何优化器。

实际节点 SDF 也显示只碰中指不能成为 Power：左 middle 首个负值样本在 delta0.05 rad，gap−0.658573 mm；右 middle 首个负值在 delta0.0625 rad，gap−0.901686 mm。其余连续排除的 pad 在所有采样点无接触。左 index 的已采样最小正 gap 1.161322 mm；右 index 2.781798 mm，但只有右 index 得到正的连续排除下界，左 index 保持未证明。

左 thumb/little、右 little 在九个状态都没有杆节点落入网格域；这些 phi 数值为 None，原 backend 域外不产生 SDF 接触。对它们的完整区间排除仍来自接触 cell 区域证明，不来自“域外”样本的简单外推。

Pinch 方法中裁剪一条已证明路径的安全后半段，不会让本轮已排除的完整 0..0.1 rad 区间出现接触。若把初始 q 移出该冻结区间、加入掌位移动或改变杆形状，则已是另一待验证问题，不能沿用这个排除结论。

## 原姿态身份与限制

原 Power archive SHA e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39；既有完整几何记录 SHA e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6。选择使用原记录，不重新计算其全部32种姿态。原 helper/current side geometry SHA 和 archive q/xy/quat 精确比对；5-pad 名称按原 canonical 顺序映射到 thumb/index/middle/ring/little，不能依赖调用方字典顺序。

~~~text
left base WXYZ =
[0.5105039993086073, 0.18544763374544743, 0.43627773435593353,
 -0.3220357739620401, -0.5751446012214666, 0.6608359594690769, 0.358883661716516]
left q =
[0.6707317317667091, 0.5595462266526852, 0.5476774792425925,
 0.5911028355468351, 0.6402046828638307, 0.7043359094005214]
right base WXYZ =
[0.5307579688771178, -0.22079301097823792, 0.42025734653082664,
 0.38290308429196845, 0.5562879629084146, -0.6245391114387701, -0.3922751950431938]
right q =
[0.41913483011807084, 0.054281451471836564, 0.6026514268006151,
 0.5705105954982795, 0.7607235842470614, 0.5849784625508111]
~~~

原 active 顺序 [finger11,finger12,finger21,finger31,finger41,finger51]。这里只说明它们作为**静态起点**满足原2.1 mm/初始杆几何约束；没有给全闭合过程桌面/非目标碰撞、arm/mount reach、双手互碰或 actual reset 准入。必要五指接触已被排除，所以未追加161点完整路径，避免把计算花在必然未通过本几何目标的候选上。12 非 pad live cooked 每侧仍未知，原 source 包络不等于真实 cooked attestation。

下一个有约束问题应先寻找能缩小 thumb/ring/little 整体距离、同时保持 2.1 mm 的掌位/方向路径，或明确使用会改变杆形状的分阶段接触目标，再分别证明 reset 和闭合。当前这些固定根界说明，在不改变其它条件时只裁初态或增加合法闭合采样无法补齐五 pad；不说明其它原 style orientation 都不可行。本轮不执行下一问题。

## 测试和证据

新增13项、相关 **79 passed，0 failed，0 skipped，5.85 s**。先记录新增模块缺失失败，再实现。覆盖接触区域包含负phi点、只有一个低值角点时必须保留整个cell、空区域、不能重复减radius、正采样不等于连续排除、0..0.1完整覆盖、非finite拒绝、非重复原姿态选择、初始相交不得选、独占输出和partial不授予reset。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_power_alternative_sdf_interval.py tests/test_dh116_power_closure_constraints.py tests/test_dh116_power_joint_palm_direction.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_flexible_style_pose_preparation.py
~~~

一次主 CPU 90/90 样本：script4.670930 s、outer7.392309 s、exit0。2线程、空CUDA_VISIBLE_DEVICES；--wall-seconds50软计时不含import/parser/最终保存，外层timeout --signal=TERM --kill-after=5s 60s。没有触发截止，未覆写旧证据。

证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/power-alternative-sdf-20261113-3a0t4gsq：

| 文件 | SHA256 |
|---|---|
| executed/final source | e1e20fe98a34edaf0899641ba9bb2cadae3df9cb48beba6e5db072b5f7b43fa7 |
| run01/result.json | 4e693d75375b2e108efcb0a40459bc2afde89ed1f4f6e47032b1a57313cc1d70 |
| outer.json | a933b1d70b0c4738bf624d179fca7aa74233d5e5e17ab5820648296dc7bcfd58 |
| offline-proof.json | cda5775a0d865906e2215277f33f1f5cfd5d5ae1aea9675fc15b458cdfd607ca |
| offline-verify.py | 64a79a9a2226143f3d634147d216d8af54432490463c8760bfe73eaecc9682b5 |

offline proof 核原 archive选择和仅z+0.3mm身份、当前source/NPZ SHA、已有90行、10个连续界，并从原网格重建10个contact-cell区域；0新FK/节点SDF查询。根代理负责集成发布，本工作树不 push，不继续搜索。
