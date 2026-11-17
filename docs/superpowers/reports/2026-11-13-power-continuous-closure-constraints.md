# Power：五个原 root 的支撑平面连续约束审计

基线 9b71010d3d591819db2642e50b9f06774109e994，仅新增独立脚本、测试和本报告。审计原负档案 5 个 root（left 15；right 1/12/13/15），0 新 root/q seed 候选，远低于 120 上限。原配置、动作/关节限制、USD、材料、R0/task 和已有证据未改。0 Isaac / GPU / physics tick / policy / optimizer 更新，无 reset proposal、native/Teacher 资格。

**结论仅适用于这几个固定 root、thumb spread 固定、各 digit 在原 +0..0.1 rad 区间的五 pad 接触条件，不证明 Power 或自由杆全局无解。** 新腕姿旋转、不同初态、双向动作、其它时长和实际杆运动都不在此结论内。

## 最终结果

最终 run03 对原 5/5 root 完成，脚本 3.793259 s、外层 6.635182 s、exit 0，无 remaining。新支撑平面计算只用了 3 条无初态冲突记录的 135 个既定 FK 状态和点积，0 新 QP 距离查询；右 12/13 的初态碰撞记录保留。原始 run01 曾只新增 left 15 的 5 digit × 9 角曲线，right 1/15 的 90 个旧角度值直接从 SHA 锁定报告复用。

下表为**最终解析支撑平面下界再扣除采样间位移和数值余量**，不是原 SLSQP gap 估计：

| 固定 root | thumb (mm) | index | middle | ring | little | 超过原 0.5 mm 门的 pad |
|---|---:|---:|---:|---:|---:|---|
| left 15 | 8.654708 | 5.978139 | −1.135374 | 5.572908 | 10.825679 | thumb/index/ring/little |
| right 1 | 6.523101 | 3.773937 | −4.997024 | 0.688786 | 5.329980 | thumb/index/ring/little |
| right 15 | 10.881673 | 5.161923 | 2.084149 | −0.420312 | −4.044000 | thumb/index/middle |

下界 >0.5 mm 可以排除该固定 root/动作区间的对应 pad 到达条件。负下界仅表示此必要条件不能排除，不意味着接触、可达或路径安全。三条完整曲线均至少有一个必需 pad 被排除，因此没有可用于双手静态配对的结果。

left 15 的原采样端点最大绝对 gap 为 11.502213 mm，thumb 对其余四 pad 的几何方向 dot 全正（约 0.856..0.977）。right 1 / right 15 的旧端点最大 gap 分别为 7.232210 / 11.584331 mm；后者原 endpoint table 门也失败。仍保留原 dot < −0.2 要求；不从距离下界推导未采样方向。

初态直接拒绝：

- right 12：source finger32 −2.892675 mm、finger33 −0.586193 mm；cooked finger33 −0.309895 mm。
- right 13：source finger13 −1.325165 mm、finger14 −0.690740 mm、finger32 −1.299637 mm；cooked finger14 −0.663156 mm。

所有 5 条记录的 17 source + 5 locked cooked 初态间隙在 run01 重新计算，和旧报告最大绝对 residual 都为 0；run03 的相同初态只按已冻结身份复用。初态相交者未做闭合查询。

## 为什么这些是下界，原 run01 为什么被降级

审查发现：旧 convex_patch_closest 的 SLSQP 可行解距离是距离上界，成功/可行不保证 ≤2 µm 最优误差。直接把其 gap 减 B/16 和 2 µm，不能严格当连续下界。本包保留 run01 原件，并用独立 run01-numeric-validity-warning.json 标记其下界声明无效；旧 offline-proof.json 只证明旧数值重算一致，**不证明几何下界**。它们不是最终结论。

最终 v2 对每个既定角度：

1. 使用同一 SHA 锁定 FK/cooked hull 顶点 V，固定原 settled rod 轴线段 P（z=0.305 m）。
2. 固定 64 个 world-XZ 方向 n；将向量归一化后略缩小，使范数 ≤1。对每个方向计算 min(V·n)−max(P·n)。
3. max(0,max_n separation) 是 convex hull 到有限线段距离的解析下界。方向离散只会削弱下界，不需要 QP 最优性。减原 radius=5 mm 得 sample surface-gap 下界，并保存方向/两个投影值等 witness。
4. 各 digit 的 9 个角度覆盖完整 [0,min(0.1,upper−q)]，thumb spread 固定。原 path_displacement_bound 使用固定 joint frames、mimic 角变化和 lever-arm 三角不等式，得到整段统一点位移上界 B；最近采样点距离至多整段 1/16。距离对点位移为 1-Lipschitz，故连续 gap 下界为 min(sample support lower bounds)−B/16−2 µm。

数值计算仍为 float64，并保留原 2 µm 保守余量；本包不声称形式化浮点区间验证。原 SLSQP 数字只作为已测估计元数据，证书 API 必须显式提供 support lower bounds，不能再只传 QP gaps。若解析下界反而高于旧可行距离估计，立即拒绝。

在相同 orientation 与动作区间下，lower−0.5 mm 的正部也是必要 root 平移范数下界：left 15 至少 10.325679 mm（little），right 1 至少 6.023101 mm（thumb），right 15 至少 10.381673 mm（thumb）。这是必要条件而非足够安全的建议动作；旋转或改初态会改变该问题。

## 动作预算与桌面限制分清

三个完整曲线 root 的所有 active upper headroom 都 >0.1 rad。当前范围受原 0.3 rad/s × 160/480 s =0.1 rad 单段预算限制，**不是机械 joint upper 已到头**。真实 source/manual/mimic 上下界和 headroom 均输出；thumb spread 的 headroom 也记录，但本闭合证明中固定它。没有提高速度、加时长/步数或改 effort/stiffness/damping。

五个旧 root 都是旧包按全部形状摆放的初态 2.0 mm 桌面净空：

| root | 最低 source shape | 对 1.9 mm 静态门余量 | 对现有 2.1 mm runtime floor 余量 |
|---|---|---:|---:|
| left 15 / right 1 | finger32_link | +0.1 mm | −0.1 mm |
| right 12 | finger42_link | +0.1 mm | −0.1 mm |
| right 13 | finger22_link | +0.1 mm | −0.1 mm |
| right 15 | finger52_link | +0.1 mm | −0.1 mm |

初态纯向下平移仅余 0.1 mm，不能整体下沉数毫米仍满足原桌面门。它们当前也均不满足既有 runtime 2.1 mm floor。本包没有证明其它方向或旋转无解，没有将初态净空当连续路径资格。

## 来源与额外真实材料

原独立 Power 负档案 SHA e5e040251c218dc90d198721b316b0a2e0bb4fec4c60b1f373de518ba8146c39；原准备报告 SHA e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6。逐条核 archive index、q、root x/y/quaternion、negative qualification，核当前每侧 source map、原数值 helper SHA。旧 root-z 改变量保留原来源，未安装进 runtime。--recheck-report 另外精确锁定 run01 result/producer SHA，不能拿其它未经核实的 ledger 混入。

检查了两个非重复真实旧档案：

- teacher-wrap-geometry-20261112-joWzRM/multifinger_pregrasp_v1.json，SHA 234eb791df392d0b78f3902d86b156b2255d22b74cfbc0a39eb8c5ba74ef29a7：五 pad 版本，geometry_candidates=0、archived_scene_matches_reference=false、native_grasp_verified=false、teacher_training_ready=false。
- teacher-wrap-local-fd-20261112-AGNuAX/fixed_patch_fd.json，SHA 183a70c0aa7db7270004d6c8b7f6a12a90de5889698704a65a6aef5cec97734e：原左右 seed 的 fixed-patch centroid Jacobian/局部结果，scope=cpu_fixed_patch_centroid_linearization，native_contact_verified=false。

后者可为后续非重复腕姿方向提供依据，但没有补足当前完整 source/cooked、桌面、杆包络门；不拿局部线性化当 qualified root。本轮不据它再发新网格。

原杆 axis z=0.308 m、r=0.005 m、桌面 top=0.300 m，原 reset 底部仍高 3 mm。沿用旧 z∈[0.305,0.308] 几何包络/settled endpoint 参考，不声称自由 settling 或 table support 已测，也不改变 registered reset/R0 seed。12 个 non-pad live cooked、自碰撞、实际材料/跟踪、371D/R0/native exit 等未知保留。所有结果仅属 17 source envelopes + 5 locked cooked pads 的 CPU 范围。

## 测试、失败与复现

新增 **21** 项，相关回归 **46 passed，0 failed，0 skipped，4.35 s，exit 0**。包括完整 9 点和预算验证、gap estimate 不可单独当下界、支撑平面与已知 box 距离比较、稀疏正 gap 不推出连续安全、原 source/curve 身份、初态碰撞拒绝、独占输出、partial 身份清单、signal handler 恢复。

开发先观察模块缺失失败；数值审查后用旧实现复现“仅可行估计也宣称排除”的失败。run02 的 5 行支撑结果完成后，finally 恢复信号 handler 时因变量 old 被上一 row 覆盖而退出 1，外层 7.054647 s；原日志/5行保留，未称成功。新增回归先复现 TypeError，再修成 prior_signal_handler，全部测试通过后 run03 重做同一 135 FK/点积，0 新 QP，exit 0。没有新 seed/候选，也没有覆盖失败证据。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_power_closure_constraints.py tests/test_dh116_flexible_style_pose_preparation.py tests/test_dh116_flexible_power_pose_preflight.py tests/test_dh116_right_pinch_closure_screen.py
~~~

最终实际命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh scripts/audit_dh116_power_closure_constraints.py --wall-seconds 50 --recheck-report /absolute/evidence/run01/result.json --output /absolute/new/run03
~~~

50 s 是 setup 起的软截止；外层 60 s TERM 加 5 s 清理宽限，不保证被终止时成功。run01 / run02 / run03 外层分别 7.310817 / 7.054647 / 6.635182 s，总计约 21.00 s；CPU 2 线程。只有 run03 是最终成功支撑平面结果。证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/power-closure-constraints-20261113-9gn5vp0a。

| 最终证据 | SHA256 |
|---|---|
| executed / final source | d1c4b0cc1218156cfa62b0fb87347618e0af9f74ba9d08003540ab1fb1218344 |
| run03/result.json | e28cdede01fa938a357b2b2cdfbf780658bf1176eabfc4ccb9ccb97a10232307 |
| run03/progress.jsonl | 9a8ebce5998ad283e7592997dcb17a6c739abaec6d485a980b23df9acd1d4f88 |
| outer03.json | fd835dd26615f0c0577fb33ac0e257f6045f26f4dbbbdcfd55d7f7d6cd823b3c |
| run01-numeric-validity-warning.json | 6e29c94fdf22a115a0f739c61f0976e26db3b0e975a24a124b90e7bc4b6e4ca8 |
| offline-proof-v2.json | 94d6b9869a0ee21b2f7ce2455e494a4a5ab3daa292a69f146cef88b6811f12f5 |
| offline-verify-v2.py | 9f0e849fb55211caf2cbc74ebdbb741fbf187e43d80334bd122628cf0ccb4664 |

最终离线 proof 不做新 FK/QP：核源码 SHA、ledger 与结果逐行相等、135 个单位方向/支撑公式、15 个连续证书重算、旧估计数据和初态未改、run02 exit1/run03 exit0。它不把旧未验证的 QP 最优误差升级为保证。原 run01 SHA 763032412514c70d34910eaa7b0bd5f4a4d3b53d751ddd41e65c26b01c0ca529 保留供 lineage 核验；旧结果不属于成功下界池。由 root 集成发布，不自行 push。

独立 run02-failure-lineage.json（SHA 92e2a37c6a1dcacdf33000f9a3f70da8acc71a9202bc1c980d4bc733e545c1c2）保留失败 executed-v2-recheck.py SHA 378ff767766003688c0a027b072d30178acbc48ee84e759dd4eab19713d5622d、outer/ledger SHA，并验证修复前后四个数学函数 AST 相同。run02 与 run03 的完整 5 行 ledger SHA 恰好相同，但 run02 的退出码仍为 1，未借此改称成功。

独立只读复核已确认 support_plane_bound / certify_curve / curve_certificates 的支撑下界与连续覆盖，且 settled axis 高度与旧曲线相同；未将范围推广到自由变形或完整垂向包络。该复核未运行测试或编辑文件；测试结果来自上述本工作树实际执行。
