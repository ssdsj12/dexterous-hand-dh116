# 左 Pinch：起点保护、完整路径与独立 Approach

## 结论

新左侧末端候选不能直接执行。第一轮完整路径发现source:finger13_link起点桌面下界0.911535mm，实测误差外推最小0.912538mm，低于原2.1mm保护值。虽然未穿透桌面且末端可行，但完整路径失败，未生成原生配方。

将起点和末端同时纳入原几何约束后，固定手腕搜索仍未找到可行解。随后独立检查左手Approach：起点World-Z升高1.5mm，在原160物理步闭合时以4.5mm/s下降到相同末端。未改变手指闭合量、右手运动、关节限制、桌面、材料或碰撞；该新增左手腕运动明确区别于原静止左手。

新路径通过1025个采样及连续位移/Lipschitz下界、232个实际源误差场景和双手间隙检查。**这是CPU条件性证据，不是新原生接触、Teacher准入或成功。**

## 实际结果

| 检查 | 固定左手腕path-01 | 左手高位Approach path-02 |
| --- | ---: | ---: |
| 连续桌面最小下界mm | 0.911535，失败 | 2.107960，通过 |
| 连续非目标间隙最小下界mm | 0.010051 | 0.009319 |
| required SDF最小下界mm | -0.286367 | -0.287636 |
| 初始所有shape/rod分离下界mm | 0.366560 | 0.513586 |
| 232场景桌面最小mm | 0.912538，失败 | 2.110113，通过 |
| 232场景非目标最小mm | 0.003396 | 0.005972 |
| 232场景required SDF最小mm | -0.289999 | -0.289999 |
| 双手连续分离下界mm | 312.815694 | 312.814962 |

原保护不变：桌面≥2.1mm，非目标≥0，required SDF≥-0.3mm；初始所有shape/rod分离≥0.05mm。
path-02的五个左指腹cooked hull连续桌面下界分别为：finger14 2.999736mm、finger23 3.283304mm、finger33 38.745548mm、finger43 55.777614mm、finger53 72.650688mm。预测桌面穿透均0，**不是新实测值**。

场景来自最新原生ledger d313e51b7857e78630f2117237bf5e48ecd9a05ad607ce8c3435077d592b178a，env0/同一episode SDK3～234。实际源左手静止；误差重用到新闭合只作条件性外推。新接触产生的动力学、跟踪与杆响应尚无原生资格。

起点约束搜索569次查询后耗尽60秒查询预算，记录查询段60.00355秒（末次在途查询含退出开销），全流程69.68964秒含读取与最终验证；未继续增加查询，结果失败。不能将优化器正常退出或总流程完成解释为可行。

## 实际代码

- audit_dh116_left_pinch_tracking_path.py：锁定末端SHA并重算，独立导入旧连续审计器；没有修改旧模块全局。左右同时运动时，双手间隙下界扣除双方最大位移之和。原生源clock/joint/rod身份逐行对应。
- solve_dh116_left_pinch_start_conditioned.py：将同一候选的实际起点误差、所有source/cooked桌面和初始rod分离纳入搜索，保存失败原因。
- flexible_left_pinch_start_conditioned_cpu_v1.json与flexible_left_pinch_raised_approach_cpu_v2.json：独立配置。后一配置显式记录新左手腕路径和原手指/右手不变。
- 三个测试文件覆盖起点失败、左右动作独立性、原模块不被变更、单边失败拒绝、全部232时钟和连续界条件。

相关CPU回归23通过、0失败/跳过，71.75秒；其中真实源重算只属于离线验证。先记录失败测试再实施。新原生物理步0、PPO0、WireSeg训练0。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_left_pinch_tracking_path.py tests/test_left_pinch_start_conditioning.py tests/test_left_pinch_raised_approach.py tests/test_pinch_tracking_target_path.py tests/test_tracking_pinch_plan.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_left_pinch_tracking_path.py --profile configs/dh116_tasks/flexible_left_pinch_raised_approach_cpu_v2.json --output /absolute/private/NEW-path-output
```

## 私有证据

目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/left-pinch-tracking-20261114；完整JSONL、失败日志和对应source.py快照保留，未上传大数据。

| 证据 | SHA256 |
| --- | --- |
| path-01/result.json，固定腕失败 | c9a4d332d4d5309a1391794d905cfc34d1e9c28284a235a469d7869bd198931c |
| start-search-01/result.json，起点条件搜索失败 | 629656ee070bc695801ccd55574b76cdfbeb2900572f21f6ed60dd2dcc0976c3 |
| path-02/result.json，独立Approach条件通过 | a5ca10a1f612a20e216b545ccbd71a7aeeceb56117ddddb1101f4e7dfa1b4d74 |

## 后续门槛

准备独立左/右共同执行配置，要求逐480Hz验证两个手的请求、投影、SDK目标和实际状态；不能复用只允许右手offset的旧validator来“通过”新左手运动。保持右路径、原逐pair/wrench、完整账本、正常资源释放、外层自然退出和单Isaac锁。

新的原生配方需绑定新代码/配置指纹，在用户持续授权的有界范围执行一次（1环境、0PPO、234总SDK步、180秒）；旧预算不重复。通过后再核验双侧有效接触和R0，仍不将短接触当作5秒Hold。Tripod、Power和正式Teacher PPO继续独立资格门控。PR#2 Draft。
