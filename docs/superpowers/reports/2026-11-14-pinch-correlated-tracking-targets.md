# 同钟跟踪误差约束的右 Pinch 目标与接近路径

日期：2026-11-14。开发基线 `caef17a4eb15a94ef42128f3ac3cfab955bd2cdb`；原生来源结果已另发布于 `209cc89606e352518cdba191774ea324f1b39907`。本工作包 **仅 CPU 搜索和原始账本重放，新增物理步 0、PPO 更新 0、Teacher 准入 false**。

## 实际问题与改动

原右手目标在命令 FK / Reset 直杆上位于原有效指腹内，但实际跟踪和杆位移使 2129 个原生接触 pair 全部落在原椭圆区域外。旧报告已给出同边界证据；本包不重复认定映射错误、不扩大原区域。

新增 `scripts/solve_dh116_pinch_tracking_targets.py`：
- `load_observed_scenarios` 锁定最新完整原生 receipt 和 physical JSONL SHA，保留 SDK211～234 共24个保持段实测状态的 env/episode/reset/rod/policy/physics 身份；左右数据不混用。
- 每个场景保留同一时刻右掌根平移误差、World 旋转误差、六个命名主动关节误差及41个真实杆节点；不独立混拼各维极值。被动耦合残差另存。
- `apply_scenario` 使用明确的旋转乘法和原耦合限位，拒绝错钟、异 episode、NaN 和越界。
- `Geometry.evaluate` 重用原完整 FK、source/cooked 几何、锁定 SDF、原有效指腹判据。独立候选只改变初始位姿/端点，原关节动作差值不变。
- 三份独立 CPU 配置保留失败搜索。v1 保留旧规划额外的 -0.1mm 接触目标和0.95椭圆上限；v2单独审计原生接触可行域（gap<=0、原椭圆<=1）。**原2.1mm桌面保护、非目标分离>=0、required SDF>=-0.3mm、法向/表面、原关节限制不变**。这不是力或支撑资格。
- v3把完整24场景审计中失败的212/214物理步加入原211/222/234约束，锁定前一次CPU结果SHA，不无限循环、不重复相同原生预算。

新增 `scripts/audit_dh116_pinch_tracking_target_path.py`：
- 对冻结端点重新计算全部24场景。
- 独立模块实例复用原连续路径审计器，保留1025点之间的 FK/SDF 位移界限。原模块的全局变量始终不变。
- 保留原100微米 World-X偏移上限，将同幅值接近路径调整为平顶形状；20个60Hz线性段，最大X速度仍<=1.14mm/s；原Z运动、原关节速度和位移不变。
- 使用同钟历史误差对全部232步计算候选轨迹，额外检查整手 source/cooked 点集的双手分离；不是掌心距离替代碰撞。
- 历史误差是经验场景，不能保证新轨迹下动力学误差相同；全部输出仍 `native_ready=false`。

## 搜索、失败与最终 CPU 结果

| 执行 | 实际结果 |
|---|---|
| search-01 | 445次查询、60.11s搜索；严格额外规划域未通过，不能执行 |
| search-02 | 284次查询、37.84s；3个选定场景通过，完整24场景在SDK212/214失败（预测接触间隙约0.118/0.057微米），未舍入放过 |
| search-03 | 加入上述反例，87次查询、12.78s；名义状态和全部24个实测场景通过端点约束 |
| path-01 | 原形状静态连续路径通过，但232步经验重放首次SDK149失败；最差非目标间隙-4.292640微米 |
| path-02 | CPU审计临时parent污染旧配置读取，来源校验拒绝，保留失败 |
| path-03 | 审计发现临时bow还影响历史命令重算；其反事实轨迹结果撤回，不用于晋级 |
| path-04 | 改用隔离模块实例；原命令不受候选影响；静态连续界限、232步经验轨迹和整手分离检查均通过 |

最终连续静态路径：
- 桌面下界 **2.121758mm**。
- 非目标杆分离下界 **0.006758mm**。
- required-pad SDF下界 **-0.291259mm**（原保护-0.3mm）。
- 整手分离连续下界 **316.543071mm**。

最终232步经验场景：
- 桌面最小值 **2.155938mm**。
- 非目标杆分离下界最小 **+0.002650mm**。
- required-pad SDF最小 **-0.026526mm**。
- 实测左手/候选右手整手分离最小 **316.507626mm**。
- 没有经验场景保护失败；**不代表新原生接触、稳定承托、Lift或Hold成功**。小余量不构成未观测扰动保证。

候选静态重定位相对旧右手端点：XYZ约 **[-0.197616,+1.991211,+0.847599]mm**；这是独立初态几何候选，不是提高Actor动作幅度。左手计划、物理材料、USD、控制器和原24D合同不变。

## 原始来源与可复核 SHA

原生来源仅使用已自然退出的234步批次：
- receipt：`9b0a8eb4f37ce20e27d5054a4d95043e21fb99febb00128a878fffa739608ee6`
- physical JSONL：`4e9b42b1fc96119888cce34d4977d2fd8720b4385409dda45a10f59fcbb8b83b`
- 原生169.025464秒，独立后置CPU审计133.705085秒；后者不是新增原生运行。

CPU输出均在Ubuntu私有 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-tracking-target-20261114/`，原始数据/日志未上传：
- search-01：`14acf94cbf33ec89ce9100e2f54fd7ef68a0241f694ef2769fcb8198d2390de4`
- search-02：`c6ddc92a3748572915f7a1da35b25d64b07805c545cdf59e6f5069ac698caa3a`
- search-03：`fceb4da1085c658142c56f82d311f2488de18242158d7eb63bc69996aef847b2`
- path-01：`e0879d44e640a99807f4675cf75442665646ab99220d41f74863516cabe6164e`
- path-02失败：`583d9eda16d3e303c967bd07c8c13dac5ef13058c07492fe9d5ea156cd3accf3`
- path-03撤回：`f0ea3d5e0f2fe4330f7beb8905b6829d03a61d374ac7a0de9e19a2e9515c52ad`
- **path-04最终**：`b4a4d2a95ae97437c182a800f01f0862180e746f90149adc924e3b4fa71da600`

## 测试、保护、晋级

先记录缺失模块/约束/隔离的失败测试，再修改。全新回归 **57 passed、0 failed、0 skipped，54.30s**；保留red日志和全部失败/撤回数据。测试包括实际冻结来源、错SHA/错钟/错episode、旋转顺序、原限位、同幅值/速度、完整中间场景、全手碰撞界限和历史命令不被候选审计污染。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_pinch_tracking_targets.py tests/test_pinch_tracking_target_path.py tests/test_effective_right_tracking_replay.py tests/test_pinch_effective_targets.py tests/test_pinch_effective_targets_actual.py tests/test_deformable_pad_region.py tests/test_surface_contract.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.solve_dh116_pinch_tracking_targets --output /new/private/endpoint
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_pinch_tracking_target_path --profile configs/dh116_tasks/flexible_pinch_tracking_path_cpu_v2.json --output /new/private/path
```

发布事务再次核对528个受保护文件和两个旧Checkpoint。新增源码2个、配置4个、测试2个和本报告；Registry/readiness/README/CHANGELOG由发布事务更新。PR#2保持Draft。

下一包：冻结这个独立候选的真实Reset、动作计划、原生owner/validator和新配方SHA；CPU回归后按已有持续授权执行一次有界原生检查。旧234步批次不重跑。原生未通过前不授予有效接触/Teacher资格，不做长训练。Pinch左手、Tripod、Power、完整Hold/材料锚滑移/稳定性来源仍独立阻断；30mm/5s/0.5mm标准不变。WireSeg本包新增训练步数0。
