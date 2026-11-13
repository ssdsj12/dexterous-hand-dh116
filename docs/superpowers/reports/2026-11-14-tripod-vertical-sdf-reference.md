# Tripod 左手起点：连续节点 SDF 区间核验

基线 35cfb7c0d9533e376968c1235aa22015c19c3a88。复用旧 Tripod 独立左手姿态和三个关节各回退 0.03 rad 的起点；没有使用 Pinch 姿态，没有修改 USD、网格、锁定 SDF、关节限位、杆参数、控制器、动作或成功标准。新增计算只解决旧粗 SDF contact-region superset 无法证明起点无接触的问题，不重复闭合搜索。

## 实际结论

旧起点的 index/middle contact-region 保守下界均为 −2 μm，不能区分潜在接触和估计过于保守。本次对原 41 节点各自的 z=0.308→0.305 m 直线参考路径，在五个固定 pad 坐标系内逐 lattice cell 计算三线性插值的三次多项式及所有内部驻点。

| 左手 pad | 进入 lattice 的节点路径 | 参考最小 gap / mm | node | 起点到终点 fraction | cell 段数 |
|---|---:|---:|---:|---:|---:|
| finger14 / thumb | 5 | 1.770638035 | 36 | 0 | 22 |
| finger23 / index | 3 | 1.747874206 | 36 | 0 | 13 |
| finger33 / middle | 3 | 2.615555249 | 35 | 0 | 13 |
| finger43 / ring | 0 | 未在 lattice 内测得 | — | — | 0 |
| finger53 / little | 0 | 未在 lattice 内测得 | — | — | 0 |

全部 205 条 pad/node 身份保留：11 条 reference_clear_with_guard、194 条 outside_sampler_lattice；共48个分段多项式。原 contact.cuh 明确规定 lattice 外 sdfSample=false，这是该 sampler 不产生接触，不能独立推断网格几何安全。没有把这些条目填零或删除。11个最小值 witness 用原 CPU float32 sampler 复核，最大差 2.140240e−9 m，无 CPU contact predicate。

另重新计算原17 source hull和5 locked cooked hull：桌面最小间隙2.403619676 mm，保持原2.1 mm门；完整直杆落杆几何包络支撑下界0.471283397 mm。全部逐形状值与原证据在1e−10 m内复现。四个曾讨论的近桌 pad 和其余形状的旧配置均未改变。

**这解除的是固定 pad、规定直线节点参考路径的 SDF 保守误拒。** 不是已经观察到自由杆在48步内沉降，也不是 GPU bitwise 或物理运动误差界，更不是完整 Reset/closure 资格。原左闭合末态 finger13/finger22/finger32 和右末态 finger13 的负可行上界仍然有效，不能用本起点结果覆盖。尚无新的 Tripod 原生配方或 Teacher admission。

## 实际代码

- `tabletop_sdf_segment_extrema.segment_extrema`：裁剪原半开 lattice，按各轴 cell 边界切段，直接由八角点构造三次系数，检查两端和导数实根。保留 fraction、cell、全部系数和最小值；上边界处可取极限下确界，固定在排除的上边界则不产生伪样本。单位保持米。
- `audit_dh116_tripod_vertical_sdf.audit_paths`：检查 proper rigid transform，逐节点 world→pad-local，保留 lattice 外条目、原 float32 sampler witness 和 source scope。沿用2 μm CPU guard，但明确它不是已证明的 CUDA rounding bound。
- `run`：核验原 Tripod parent、原侧别几何/SDF/cooked identity、实际 contact.cuh SHA，重算22形状桌面/包络，对比旧记录，再输出完整数据。新输出独占创建，不覆盖历史。
- 超时45秒软限制，partial/error不授予资格；实际完成2.242185849秒，0 Isaac、0 GPU、0新物理步、0 PPO更新、0 reset。

没有任何后端、安全投影、训练奖励或成功阈值修改。原371D/24D不变。

## 测试和证据

先观察新模块15项失败、driver8项失败，均为尚未实现的预期断言。实现后运行新增26项及原相关40项：**66 passed，0 failed，0 skipped，3.61秒**。覆盖内部极值、跨cell反向轨迹、半开边界、单点边界接触、无效输入、独立Scipy稠密插值对照、刚体坐标协变、记录不删节点、原始SHA拒绝及输出保护。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_sdf_segment_extrema.py tests/test_tripod_vertical_sdf.py tests/test_tripod_vertical_sdf_actual.py tests/test_dh116_tripod_two_phase_sdf.py tests/test_dh116_pinch_closure_margin.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_tripod_vertical_sdf.py --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-vertical-sdf-20261114-run01 --wall-seconds 45
```

| 来源 / 结果 | SHA256 |
|---|---|
| 原 two-phase result | 68b244ea5adee6dada5394b5e46e7e96270b12d3f9d6d1cfb24e8d2a9fad51a4 |
| 原 SDF summary | db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab |
| 原实际 contact.cuh | 8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb |
| 本次 run01/result.json | d0040fc6f282687b16bbc1336f7cbfdabc54bac52fc6b696facdf720e6ec4ba1 |
| 本次 regression.log | 5f39c950ef83bca46aa50bfa7b9e40c47041d0bdc13a4dbae4ea604bb2c7a30a |
| 原预期失败 red.log | 29fbe01793711bdf9a44d62ac52e1e0bdd63eb6490c707963a0d798587e56f25 |
| 原预期失败 driver-red.log | dd661d0b1ca7b1daf22c7afd658655efb1c69371682eb416a9361ee3d97c1e28 |

结果含7个直接实现文件SHA、原几何映射、5个原始NPZ/cooked身份和所有节点记录。原始结果与日志留服务器，不上传。日志目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-vertical-sdf-20261114。

## 后继

Tripod闭合仍blocked。下一项应联合节点轴向站位与三指接触方向生成受限非线性闭合候选，独立检查近端/桌面连续路径；不可继续从已知相交的端点发原生配方。当前CPU通过不继承Pinch原生资格，不改变30 mm / 5 s / 0.5 mm。Pinch材料滑移来源、Power独立几何和相机生命周期分别推进。
