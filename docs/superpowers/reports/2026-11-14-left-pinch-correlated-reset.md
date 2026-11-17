# 左 Pinch：实测来源与独立 Reset 候选

基线 d35b997adf40e90c4a649f5cbe2a20c5f2507de4；Ubuntu 独立工作目录。CPU 搜索与实际原生证据明确分开，本包新增物理步0、PPO更新0、WireSeg训练步0。

## 原生来源与纠正

最新 tracking-aware 原生 receipt SHA 83a8758210fd6aa35da239c32819b91b9b6967307d30f7e380c67f965381cf11；账本 d313e51b7857e78630f2117237bf5e48ecd9a05ad607ce8c3435077d592b178a。该实验左手24D的前6个主动关节动作始终0，右手执行独立路径。因此左侧零接触不应描述为“左手已执行闭合但失败”。

新脚本 load_packet() 分离实际 held-left anchor 和拟议闭合末端。取211～234共24个同身份实际root/q/rod场景，以原静止命令计算误差，再用于独立候选场景检查。左侧使用自己的USD、SDF、FK和关节身份；静止跟踪误差迁移到新闭合动作只是未资格化外推。

## 三次有界CPU结果

| 实验 | 方法/变化 | 查询数 | 结果 |
| --- | --- | ---: | --- |
| search-01 | 原9变量局部minimax，Reset ±2mm、±0.03rad、前三主动关节±0.04rad | 367 | 失败，最差约-0.120099mm |
| search-02 | 保留相同物理约束，独立非线性约束残差搜索 | 246 | 失败；优化器xtol返回不等于可行 |
| search-03 | 独立Reset候选范围±4mm、±0.06rad、前三主动关节±0.08rad；由线性范围审计给出不同种子 | 384 | 名义及全部24个末端场景通过 |

所有搜索单次上限600查询/60秒。没有重复原生配方，也没有将本表算作原生接触次数。最深SDF节点以外的全部局部节点已检查，没有漏选的合格指腹接触点。第一组局部失败不能解释为全局不可行。

search-03的root相对旧anchor平移[2.027421,-4.000000,0.985042]mm，旋转用明确World旋转向量左乘原四元数。前三主动关节Reset允许独立调整，但原闭合增量仍严格为[0,0.01855539082010854,0.0234063279676594,0,0,0]rad，原速度和持续时间不变。没有提高动作幅度或改变关节、材料、网格和桌面参数。

末端root(wxyz)：[0.4973969161961029,0.23906903145211275,0.41930930325897003,-0.5284449611135191,-0.4011928281159467,0.747331218409863,0.03586485371626821]。
末端主动q(rad)：[0.19812732728305926,0.48758451378032625,0.6792602739054427,1.15,1.15,1.15]。
最小场景约束余量仅5.857646e-10m，位于SDK224的食指SDF深度约束。另有10µm原规划reserve。**这不是稳健性或动态保证**，完整路径可能仍失败。

## 文件与验证

新增 solve_dh116_left_pinch_tracking.py，v1/v2/v3三份独立CPU配置和三个测试文件。未修改旧求解器/旧原生配方/原模型。

相关新旧CPU回归25通过、0失败，30.58秒；日志SHA 10e2b16069cfe5ab9ec808c17dbc9643b2eeef74b5ae95f11ab291e0904a8bda。先记录模块缺失的失败测试，再实施；失败搜索及原始JSONL保留在私有目录，不上传大日志。

命令（均Ubuntu、无Isaac）：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_left_pinch_tracking.py tests/test_left_pinch_feasibility_search.py tests/test_left_pinch_reset_search.py tests/test_pinch_tracking_targets.py tests/test_tracking_pinch_plan.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/solve_dh116_left_pinch_tracking.py --config configs/dh116_tasks/flexible_left_pinch_tracking_cpu_v3.json --output /absolute/private/NEW-output
```

私有证据 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/left-pinch-tracking-20261114：

| 文件 | SHA256 |
| --- | --- |
| search-01/result.json | 4c35c37916c7efd0a9fd470fd578195a43e589ac3119c59ade7bbb6c8508884a |
| search-02/result.json | 2dcb4d5ef49394b238e30632fb1275a373150d172e5b8dbd0620425850a80849 |
| search-03/result.json | f8acb5e5d1f54e92d037554f9d995eaa8cc0b0a797b8fdb7a08c52e25903c2a7 |
| all-node-audit-01.json | c6c674cbf878f64445967b4435c2cbddb9c8ce8cef66352155917658534ebd3e |
| linear-range-audit-01.json | f263d85299e95a8e554e3a021173491bea22d4c8b7e5ca27b16dec742b381803 |

## 资格与下一步

本包完成的是CPU来源/候选功能，native_ready=false、teacher_admission=false。下一包运行完整1025采样连续界、232实测误差场景和双手运动间隙检查。初始所有几何必须分离，原桌面2.1mm、required SDF -0.3mm和关节保护不变；通过后才能准备新的冻结原生配方。Right原短接触路径保留。Tripod/Power与完整R0、Hold/材料锚Slip资格仍独立，Teacher PPO尚未就绪。PR保持Draft。
