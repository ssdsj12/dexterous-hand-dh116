# Pinch 初态展开角与轴向位置：有界非线性几何检查

基线 `6cab12fb2416ae3a6d6d13be18c8247c5a46b507`。本包新增一个纯CPU诊断脚本及测试，不修改旧USD、几何、物理参数、动作合同或原生冻结配方。没有新Isaac实例、原生步或PPO更新。

结论：初态桌面余量问题可以在本轮小范围内改善，但一次局部线性模型仍高估了末态接近效果。**两侧最终真实非线性FK/SDF检查都没有达到指腹接触；不生成Reset或原生配方。** 这不是全局不可达证明，也不是已发生碰撞的证明。

## 来源和单次预算

使用刚完成的双手18步原生读回最后一个SDK18端点的41个实际杆节点，作为固定几何输入。父回执已经独立验证自然退出和完整时钟；本包不会重新运行或修改该原生实例。

- 父native receipt SHA `dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0`。
- 16条实测ledger SHA `140584e4375b96343478ec973fdbdb94f62d8a2a687fc8dd3b1ba1a5db1a2ec9`。
- 原左右独立paired初态 SHA `996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37`，reset seed SHA `647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d`。
- 左右分别加载原USD、17 source形状、5 locked cooked指腹及其1mm SDF；来源全部写入结果，不将左手镜像当右手。
- 同一实际自由杆和原半径5mm，未固定节点、移动杆、补点或更改材料。固定端点CPU问题不预测杆之后的运动或反作用。

新设计变量：静态初始world Z增加0–2mm、world Y沿杆改变±3mm、初始thumb spread改变±0.06rad，以及原thumb/index闭合量。根X和四元数、其他手指固定。初始展开角不是运行时新增动作；原六关节映射、耦合及上下界继续执行。动态两指闭合仍各≤0.1rad，160/480秒对应≤0.3rad/s；没有额外腕部控制。

在求解中同时加入首态和末态约束，避免只优化末态而忽略初始桌面10微米余量：
- 初态设计桌面净空2.5mm、所有形状（包括目标指腹）的杆净空0.1mm。
- 末态沿用设计桌面2.4mm、非目标杆净空0.2mm、目标node-SDF −0.1至−0.3mm。
- 上述为搜索设计目标，**不替换原运行时2.1mm桌面保护和非目标不相交要求**。

每侧仅一轮两尺度局部Jacobian与一次LP：21个变量点×初末2状态=42，再完整检查33点线性关节路径，总75/侧。实际共150状态≤冻结160上限，脚本4.882秒、外层7.783秒，exit0。内层50秒、外层60秒，无循环重试。源代码冻结SHA `fa1ac1aebea269bfe6e319ec2841ea1c42ef26f3e1d2c6c5182f72b04ac90dcb`。

连续桌面/非目标下界以原每形状全路径位移界B扣B/64覆盖33个均匀采样点；非目标支撑面包含全部41实际节点和折线，不能只检查最近节点。目标接触仍由真实已锁定节点SDF做完整非线性重算，几何谓词不冒充接触力。

## 两侧实际结果

| CPU量 | 左手 | 右手 |
|---|---:|---:|
| 初始world Y变化 | +3.000mm | +3.000mm |
| 初始world Z变化 | +0.389103mm | 0 |
| 初始thumb spread变化 | −0.000375294rad | +0.060000rad |
| thumb闭合增量 | 0.021514079rad | 0.018724677rad |
| index闭合增量 | 0.027233801rad | 0.035404282rad |
| 初态最小桌面净空 | 2.500000mm | 3.260417mm |
| 初态所有形状最小杆下界 | 1.513605mm | 1.490603mm |
| 全路径连续桌面下界 | 2.446609mm | 3.227633mm |
| 全路径连续非目标杆下界 | 0.372876mm | 0.362950mm |
| 末态thumb node-SDF gap | +0.644915mm | +1.069333mm |
| 末态index node-SDF gap | +0.235704mm | +0.140255mm |
| 双指目标接触 | 未达成 | 未达成 |
| 原生配方生成/Teacher晋级 | 否 | 否 |

此范围内桌面/非目标下界已通过原门，但四个目标gap仍为正，不能写成建立接触。没有实际穿透的证据。所有完整pose、q、路径、节点身份、Source/SDF SHA、Jacobian和被拒候选都保存在结果中。

## 局部模型失效点

左右两尺度Jacobian相对差分别0.00112289/0.000188499，说明极小扰动处导数一致；**这不证明大位移线性预测有效**。

两侧LP均给共同slack=0，预测目标gap满足−0.1mm；实际非线性检查相对于该接触目标的缺口为：

- 左拇指0.744915mm、左食指0.335704mm。
- 右拇指1.169333mm、右食指0.240255mm。

Y变化达到允许边界3mm，右展开角达到0.06rad；导数是在归一化0.005/0.01的小邻域计算。不能把预测的可行性当成物理可行性，或把导数稳定当成完整MPC已成功。这里已由末态实际FK与锁定SDF否决，未送入原生。

下一独立设计应使用有限次数的非线性复算与信赖域，比较实际/预测约束改善再决定是否接纳，而不是继续放大动作或扩大原关节界限。该后继尚未执行，本150状态预算已结束。

## 测试与独立审查

先保存缺失模块的预期失败，再实现。实际 **59 CPU tests passed**（本包24、旧相关35），无失败或跳过。3个SciPy提示仅说明threads=2透传HiGHS，不是物理验证。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_pinch_spread_axial_direction.py tests/test_dh116_pinch_raised_reset_direction.py tests/test_dh116_pinch_closure_margin.py
```

测试覆盖左右独立静态初态变量、原耦合限位/速度、双尺度21查询计数、整体预算、初态包含目标指腹、33点连续覆盖、约束身份变化、失稳局部模型拒绝、独占输出、timeout部分证据和资格不误报。独立agent在运行真实CPU几何前只读审查源与父回执，未发现阻断；未代跑、未作原生审查。

私有证据：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-spread-axial-20261113-q3um_jla`。失败测试与最终回归在 `pinch-spread-axial-cpu-GeQJv6`，不上传大日志。

| 证据 | SHA256 |
|---|---|
| frozen_plan.json | 83cbd1497566e3eddfa5cf238d4f6b0a68079897779b6b1b81c262dd24b8958f |
| run01/result.json | de7a80babae315a71398d203cbed0cf7ec97a331aaf3b8264240ef0b6fa52c3d |
| outer.json | 31f1659e2a08bfa9b840ffc3cce0c31b676e014215ead30f800243bd4857d989 |
| run.log | 48bf4f10e010809236b517a3f6350884831d88f878592a4c2ba54c9d2f8b5314 |

本包只有固定实测节点的CPU几何证据。它没有改变已发布双手原生18步0接触结论，也未授予任何抓取、Lift或Teacher资格。
