# Tripod 节点对齐非线性闭合：局部改善，端态仍拒绝

基线728cdf86b4b9dc3f12835b6c65186c3874654c45。仅使用前包独立左手Tripod回退起点，不使用Pinch初态/动作。新增有界CPU非线性端态求解及回归；不修改旧控制器、USD、SDF、材料、关节硬限位、371D/24D、2.1mm投影门或30mm/5s/0.5mm成功标准。0 Isaac、0 GPU、0原生步、0训练更新。

## 设计和真实失败

以前固定根闭合产生多个近端碰撞；本次让原CPU信赖域中的根平移/旋转、thumb spread和三指闭合共同参与非线性FK/SDF查询。原41节点、5mm半径、固定沉降直杆z=.305m只作为条件几何模型，原reset的.308m不变。每个候选重新计算17source+5cooked形状、三pad最近节点及梯度、连续杆到凸包区间和对向指标。

- 根平移各轴±5mm、world rotvec各轴±.1rad、thumb spread±.1rad，三闭合通道[0,.1]rad且原耦合硬限位再次限制；ring/little保持旧值。这是CPU端态域，不是已获准的腕运动轨迹或新动作幅度。
- 沿用原coarse以外的strict±50μm几何门，节点SDF也记录±50μm近接触条件；优化目标−25μm是诊断目标，不改变solver的gap<0接触定义。
- 任何单项几何不合格都拒绝，不以总残差下降替代门。连续路径、腕/臂可达、自碰撞、动态和Reset均未通过。

第一次默认TRF从零下界被数值内移到1e−10，产生极小初始trust-region。484次查询后仍几乎停留原点：残差平方和45.023902→44.685548。这个数值停滞不是物理不可行证据。保留原执行源码、全部记录和失败结果。

新增边界启动回归，改用显式dogbox，保持相同原始起点、域、残差、max_nfev=22、最多500次几何查询及55秒CPU软限。第二次312次查询、4.056184651秒结束，nfev预算耗尽；solver_success=false。没有扩大原生预算、没有从旧结果续训、没有生成原生配方。

## 第二次最佳诊断候选

最佳记录为evaluation300，属于Jacobian探针，**不是优化器最终接受步**；它与最终solver_x的world-rot-x归一化分量相差.001。所有312候选均未通过端态门。下面数据只说明该局部候选可进一步研究，不能部署。

| 指标 | 结果 |
|---|---:|
| 总残差平方和 | 0.019060455 |
| thumb / index / middle node-SDF gap | +47.050521 / −10.336642 / +38.202191 μm |
| 最近节点 | 36 / 36 / 34 |
| thumb-index 凸包方向dot | −0.197670215，未通过<−0.2 |
| thumb-middle 凸包方向dot | −0.687621430 |
| 两对SDF梯度方向dot | −0.360845545 / −0.757565035 |
| 最小桌面间隙，source finger32 | 2.093919863mm，距原2.1mm差6.080137μm |
| thumb 连续hull区间 | [+33.573595,+38.019970]μm |
| index 连续hull区间 | [−77.149881,−67.102885]μm，strict拒绝 |
| middle 连续hull区间 | [−111.740299,−65.621193]μm，strict拒绝 |

未选择更有利的SDF法向来覆盖不合格的原几何对向门。节点球面与连续capsule的差异保留，不把node gap近零当作三指原生承载。

额外使用原convex_patch_closest和完整3D支撑证书，对三个近端作3次独立查询：

| source shape | 可行gap上界（含原2μm guard） | 完整3D支撑下界 |
|---|---:|---:|
| finger13 | −51.705661μm | −56.503204μm |
| finger22 | +235.077369μm | +225.468380μm |
| finger32 | +2268.135984μm | +2263.291335μm |

finger13负可行上界证明在此静态模型里仍然相交，不能通过换保守下界方法清除。finger22/finger32的正下界证明对应杆几何已分离，但finger32的桌面门仍失败。总查询数第二次312+额外3=315；没有相交深度的最小分离量声明。

最佳候选根xyz变化[+0.272099,−0.446846,−1.130388]mm；world rotvec[.03736749,−.03302504,.06411989]rad；thumb spread−.07331075rad；thumb/index/middle闭合[.03002929,.03507652,.01328699]rad。它只是独立CPU候选，完整姿态/q及全部查询保存在原始结果，不写入运行配置。

## 实现与验证

`solve_dh116_tripod_node_closure.py`：
- decode/bounds核验独立Tripod身份、原耦合限位和CPU域；
- measure每次真实FK、float64参考SDF、全部形状净空；无SDF样本用明确unsupported惩罚且原值保留None，不伪装测量0；
- jacobian使用固定绝对步长和边界内单边/中央差分；
- Budget保留完成行并限制500查询；
- bounded_fit显式边界安全dogbox；endpoint_gate逐项拒绝；
- 最大55秒和独占输出，超时/预算耗尽不授予资格。

先见12项缺失实现失败，边界修复前另1项预期失败。新增16项加旧回归共**58 passed、0 failed、0 skipped、2既有HiGHS threads转交警告，3.07秒**。真实CPU数据回归验证原SHA、完整484/312行、最佳探针身份、负上界证据和资格false。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_node_closure.py tests/test_tripod_node_solver_boundary.py tests/test_tripod_node_closure_actual.py tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_tripod_sdf_constrained_direction.py tests/test_sdf_segment_extrema.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/solve_dh116_tripod_node_closure.py --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-node-closure-20261114-run02 --wall-seconds 55
```

| 证据 | SHA256 |
|---|---|
| 原quiet-start父结果 | d0040fc6f282687b16bbc1336f7cbfdabc54bac52fc6b696facdf720e6ec4ba1 |
| run01/result.json，TRF484查询5.465568s | c4ea21c8eb95900612afee1ddd08088b3f4f25fc761ac533797c70ef7e8a9783 |
| 保存的solver-trf.py | b2920aa9d6ece4d361e6d125bd019d52c958ca17539888683bd337a6c633de88 |
| run02/result.json，dogbox312查询4.056185s | f0e36fb782299ee48e41e9282f5c6943563c1aef51332e15201ef6981772e339 |
| run02/nonrequired-witness.json | 0b156293f517c3a201d5f43c1c2d55ca327a0ad1056e0c29151b8d14a3f13f36 |
| regression.log | e0eb3609688d16a3c24d7e84fd283392b0028a64f509a67bf4c33a5d55a564c8 |
| red.log | 47948489e0df2f39e5ae345e498c54454afc92b76486edce37d63614461ce44a |
| boundary-red.log | 305cbf0766aaf053b7b9af6cd65ce0c2d9503b6ea78f8e0d910059efaa935309 |

原始记录路径前缀：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-node-closure-20261114-run01和run02。
日志/旧执行源码：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-node-closure-20261114。原数据不上GitHub。

## 阻断和下一步

此CPU诊断工作包完成，Tripod物理后继仍blocked。下次应以finger13实际相交方向、finger32桌面余量和三pad几何为硬约束作独立候选修正，并重新证明起点到末态的连续路径，而不是提高迭代次数、接触容差或取消保护。右手沿其独立起点继续，不复制本左手姿态。未执行或授予任何Tripod原生、Teacher、R0训练资格。
