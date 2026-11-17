# Tripod：原节点 SDF 与初始碰撞约束的一次局部方向

基线 bbe4efa332a83e143a632c3b060fbb8f0ce06712；几何中心严格来自已集成 87a4e1b 的 Tripod joint/root 结果，非新 Pinch pose/q/动作。新增独立脚本、CPU 测试及本报告。integration、原 USD/配置/SDF/材料/杆 seed/阈值/旧证据均未改。0 native、0 GPU、0 训练更新；没有 reset proposal、没有原生或 Teacher 资格。

## 结论

**这两个已有 Tripod 闭合端点不能直接变成安全的 pre-contact 初态，本轮一次有界约束方向也未找到合格解。** 本包补齐两个此前未量化的阻断：真实节点 SDF 与连续杆段距离的差，以及旧最佳状态对原落杆包络的实际几何重叠。不是重扫网格，也不是 Power/Pinch seed 的替代或镜像。

| side | 两尺度 Jacobian 相对差 | 起点共同违约 | 线性预测最小共同违约 | 最佳真实查询共同违约 | 最佳回溯 |
|---|---:|---:|---:|---:|---:|
| left | 0.003388381 | 1.988156 | 0.879326 | 1.583861 | 0.25 |
| right | 0.005123585 | 1.512195 | 0.662863 | 1.032684 | 1.0 |

共同违约是 55 个归一化约束的最差负值：距离除 1 mm，方向 dot 未换算单位，**不是统一的毫米数或任务 reward**。线性 LP 在本次信赖盒中仍需要正 slack；这是当前保守约束模型的数值结果，非严格非线性/全局不可行证书。两侧真实最大违约虽分别下降 20.34% / 31.71%，独立门仍拒绝，没有用损失改善掩盖碰撞。

最佳左手桌面最低值 3.526182 mm，超过原 2.1 mm 门 +1.426182 mm；右最佳 source finger32 仅 1.348051 mm，低于原门 0.751949 mm。右方向触及 root x/z、world rot-z、thumb-spread 的局部边界，不能称为已找到安全小修正。左原 finger22 下界改善为 +0.772185 mm，但其它阻断仍存。

## 新的源几何证据

读取左右各自原 17 source hull + 5 locked cooked hull、全部 5 个原 1 mm SDF、同一原 pose_reference 的 41 节点杆。SDF 局部 frame/capture/SHA 与每侧 cooked capture 独立核对。只复用新 Pinch 包的通用 CPU 三线性 sampler，未导入其候选、初态、实测姿态、速度或动作。

原最佳状态在 settled axis z=0.305 m 的三指距离，单位 µm：

| side / pad | 连续杆段到 cooked hull 可行 gap | 最近节点球面到同 hull 可行 gap | 锁定 SDF 节点球面 gap |
|---|---:|---:|---:|
| left thumb | +671.763 | +671.931 | +705.545 |
| left index | +876.370 | +1481.997 | +1489.182 |
| left middle | −446.603 | +1025.164 | +1042.010 |
| right thumb | +154.365 | +168.499 | +179.774 |
| right index | +46.221 | +1365.242 | +1377.144 |
| right middle | +8.200 | +915.430 | +917.040 |

节点名义间距 12.5 mm，半径 5 mm。这里左最近节点为 [36,36,34]、右 [2,3,4]。尤其左 middle 的连续 capsule 已与 hull 径向重叠，最近节点球面却仍有约 1 mm 的正 SDF gap。不能用“连续距离近零”推断 solver 节点接触，不能简单把差异归因于 1 mm 网格。节点 sphere 到 hull 的 QP 值是可行距离上界；与 SDF 的数 µm到数十 µm差值仅作当前点比较，未声称烘焙的全局误差界或 GPU bitwise parity。

再用 14 次受限原 QP 查询（8 个主要 source 包络 witness + 6 个节点球面对照，0 新候选）分清负下界与实际相交：

| 状态 | source shape | 到原 z=0.305..0.308 m 落杆包络的可行 gap |
|---|---|---:|
| left 原最佳 | finger22 | −1.134649 mm |
| left 原最佳 | finger33 | −1.985654 mm |
| right 原最佳 | finger13 | −0.270916 mm |
| right 原最佳 | finger23 | −1.498517 mm |
| left 本轮最佳 | finger33 | −1.574894 mm |
| right 本轮最佳 | finger13 | −0.813637 mm |
| right 本轮最佳 | finger23 | −1.019887 mm |

这些值远小于 −2 µm；可行 QP 上界为负，足以证明本静态 hull/capsule 定义下与该包络有交集，不只是支撑下界不够紧。多数最近轴点在原初始 z=0.308 m；右原 finger13 最近轴高为 0.306665 m。包络交集不等于已模拟的自由杆轨迹必然碰撞，更不等于真实力或最小分离穿透深度。

上述较近的原最佳是旧 closure endpoint，**不是已经验证的初始 reset**。将“初态必须在整个落杆包络外”和“同一 q 对 settled rod 达到 strict 三指近接触”同时施加在它上面，是本次明确测试的 staging 假设，不是对所有 Tripod reset 的必要条件。

## 模型、预算及保留门

每侧独立中心是 parent.sides[side].best.candidate；只取其已经算好的 pose/q，未再次加旧 closure delta。原独立 Tripod archive SHA c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b；父结果 SHA 726ca01dc554cb1a2a8a1ee03222c240118b536ae4cd7ad7348a19ca36fa39ba。

变量沿用原 10 维：root xyz（每轴±5 mm）、world rotvec xyz（每轴±0.1 rad）、thumb-spread/thumb/index/middle（各±0.1 rad且原 active_limits/expand 同时限制）。ring/little 不变。原物理/关节/材料阈值不改；这个盒只限制 CPU 局部试验，不赋予腕部实际可达性。

每次 55 个约束：
- 全 22 shape 桌面最低顶点≥原 **2.1 mm**。
- 全 22 shape 对原杆垂向包络的支撑下界≥0；扣 2 µm。包含目标和非目标，因此新初态不得靠初始碰撞建立接触。
- 三 required cooked pad 对 settled continuous rod 的 feasible upper+2 µm≤原 strict 0.05 mm；保留原 coarse 0.5 mm 的更严格子集，不放宽 strict。
- 三个实际锁定 SDF 的最近节点 gap∈[50,500] µm。此区间是本轮“未接触且可近接触”的诊断 staging 条件，**不是新 physics contact margin**。原 solver 接触阈值仍是 gap<0 且 gradient非零。
- thumb 对 index/middle 的几何 dot<−0.2，缺失方向拒绝。

支撑平面复用原已审查实现。XZ 法向对垂向矩形投影等价；SLSQP 上界不拿来证明正净空。SDF 仍依原 1 mm NPZ，在同一条 41 节点杆上采样；原杆初始 axis z=0.308 m、桌面 top=0.300 m 和 radius=5 mm 均不变，未把 .305 m settled 几何目标写回 reset。

每侧基线1+两个中央差分尺度各20+4实际回溯=45，共90；两尺度 h=0.01/0.005。稳定性≤0.2后只调用一次 HiGHS LP（time_limit=1 s、maxiter=200、threads=2），最小化共同 slack，再验 [1,.5,.25,.125]，不迭代。两次 CPU LP 并非训练 optimizer。SciPy 明确提示 threads 选项原样转交 HiGHS，日志保留该 warning，非求解失败。完整 labels、55维值、4个 Jacobian、方向、回溯、pose/q、原 SDF绑定都保存。

本轮没有完整运动路径、自碰撞、arm/mount reach、真实 reset/同杆双手接触或 native 验证；12 非 pad live cooked 每侧仍未知。两侧均未通过，所以不生成配对或 reset proposal。

## 下一项有约束的假设

不再从旧闭合端点同时要求“落杆初态无接触”和“settled strict 近接触”。后续可把未知问题明确拆成：先在独立原 Tripod seed 上证明整个落杆/quiet 的 reset 净空，再在已证明的无接触起点到 settled 三指近接触之间验证原速度受限路径；两阶段必须分别记录，不能把 endpoint 冒充 reset。

节点纵向位置也必须进入该路径设计。当前局部 Jacobian 中，右 index 的 SDF gap 对 world-y 平移导数约 −0.273176，middle 为 +0.403728（m/m）；同一个小 y 平移不能同时减小两者 gap。这只说明当前局部方向的冲突，不是全相位不可行证明。下一假设应联合独立 digit 初态与掌姿来改变相对节点位置，并继续保留 2.1 mm 和非目标净空；单纯扩大闭合或盲扫 y 不解决已确认的源碰撞。本轮未执行该下一假设。

## 验证及证据

新增12项、相关 **69 passed，0 failed，0 skipped，2 warnings，3.07 s**。测试先见新增模块缺失，再实现。覆盖 Tripod seed隔离/不重复closure、硬约束可行与不可行情形、不同label/非finite拒绝、两尺度不稳不得求解、原2.1门、变量上限、父SHA、独占输出、partial仍拒绝。未将其它 agent 未进行的审查记为通过。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_tripod_sdf_constrained_direction.py tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_power_closure_constraints.py tests/test_dh116_pinch_closure_margin.py
~~~

主真实 CPU 一次：90/90，script4.469239 s、outer7.354747 s、exit0。50 s 软截止从输出创建后的setup开始，不含import/parser/最终保存；外层 timeout --signal=TERM --kill-after=5s 60s。额外14 QP使用30 s外层限制，正常完成。0 native/GPU/训练。离线proof没有新几何查询，重建90候选、4Jacobian、预测约束及最优回溯，核源/NPZ/父身份与outer exit。

证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-sdf-constrained-20261113-3jh3b4xj。

| 文件 | SHA256 |
|---|---|
| executed / final script | 49d02d659205cd9e99e37e25f5b13e7bdd0a55b47e0abd3aa7a5a44e426a6372 |
| run01/result.json | 4aea5a1a55ee5d9a3919e92d19891918885c60825b6fa2044b0cac3ed7f4ced5 |
| outer.json | 634ddb3df67c344761821e84a6beb757b2f6544a0ecd6cb94b36cd44ef8a0ad6 |
| collision-witness.json | 1459036c336b7c779d57c967d392590a2fd6d561915e569c3db6188005edd2b3 |
| offline-proof.json | 9ab5c26b8b82f0f55d7053accb5bf88e8c921eb9ae0dbb57516fc045712cf142 |
| offline-verify.py | 5e2f81fdd041c2c365750f4f1aa2bc76c6d85781f030626fa363d298a58106ed |

独立分支由根代理集成发布，本工作树不 push，不继续搜索。
