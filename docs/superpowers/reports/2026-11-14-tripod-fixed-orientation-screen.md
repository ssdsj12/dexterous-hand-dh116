# Tripod 固定腕姿候选：理想路径通过，跟踪滞后筛选未通过

基线 `92b5b036604242a291cd1511593fa651ebed7b33`。
本包为Ubuntu CPU离线候选筛选，新增原生步0，PPO更新0；没有修改原配方、旧模型、物理、关节限制或执行器参数。

## 为什么修改候选

上一批SDK199终止前缀已定位到实际腕旋转滞后；理想同步路径不能代表真实联合运动。
本候选将腕旋转固定到原参考终点姿态，在独立初始化提案中把掌根Z提高1.5mm，
随后在200物理参考步中下降1.5mm，同时按原q0→q1移动手指。右手不变。
这是初始姿态/接近路径提案，**不是已应用到原生任务的配置**。

原24D动作仍由已有compile_actions及原控制算子生成。腕旋转动作全0，右手12D全0；
左Z归一化动作−0.36，原物理480Hz/控制60Hz、2.1mm桌面保护和原动作率限制不变。
预期新配方若将来通过完整准入仍为2setup+8quiet+200motion+24observe=234步，
但本包**不生成可执行原生配方，不消费新预算**。

## 两层验证

1. 原USD source17、锁定cooked5、原关节耦合、原24D SDK目标重建。
   连续理想目标路径用原Lipschitz细分检查，65个采样点、32个连续区间通过，
   最小证明裕量0.000391280mm。这依赖精确跟踪，不能替代真实运动证明。
2. 从已锁定SDK3–199原生账本计算各关节位置误差/实际每步SDK目标增量，
   保存观测延迟及向上取整再加1tick的**离线筛选范围**。
   不修改控制器，不把这个纯延迟模型称为系统辨识或动态保证。
   四个活动手指延迟角点与平移延迟角点独立组合；29个时间位置×32个角点=928状态。

| 观测量 | 最大观测延迟(tick) | 筛选上界(tick) |
|---|---:|---:|
| finger11 | 9.596546 | 11 |
| finger12 | 9.636888 | 11 |
| finger21 | 9.599536 | 11 |
| finger31 | 14.455173 | 16 |
| wrist XYZ | 4.799006 / 4.904436 / 4.807542 | 各6 |
| wrist Roll/Pitch/Yaw | 47.076706 / 47.075768 / 47.048252 | 各49 |

固定腕姿候选无需动态旋转，但真实静置姿态误差/未建模响应仍待单独覆盖。
finger41/51无运动标签，记录null，不伪造延迟0的实测资格。

**928状态中24个未通过。** 最差发生在参考motion200、scenario16：
finger11延迟11tick，其余手指0、平移0，source:finger13_link保守间距
−0.029199923mm。这不是实测穿透，也不是新的原生运行；
它说明独立手指跟踪误差足以破坏几何裕量，故拒绝晋级。

## 实际修改和测试

- `scripts/prepare_dh116_tripod_fixed_orientation.py`：
  compile_candidate复用原24D生成器；
  recorded_lags逐SDK读取真实q与target；
  delayed_state分离手指/平移时间；
  screen检查全部原source/cooked约束；
  readiness明确区分理想连续证明、离散滞后筛选、实际原生资格。
- `configs/dh116_tasks/flexible_tripod_fixed_orientation_cpu_v1.json`：
  独立CPU配置，native_execution_enabled=false、maximum_new_physics_steps=0。
  配置中的保护/频率/准入字段必须严格匹配，不默默忽略修改。
- `tests/test_tripod_fixed_orientation.py`：13项；
  连同已有SDK路径、固定根路径、原生跟踪复放回归共 **44 passed,0 failed**。

先保留缺模块RED。第一次928状态筛选已完成，但结果序列化因原proof.samples使用float键失败：
原encode_owned要求字符串键。这次输出目录和日志保留，未伪造成功。
新增失败回归后，把proof.samples显式编码为fraction/measurements列表，
先序列化再打开最终文件；第二个独立CPU输出目录完成，耗时8.163911s。
原始proof算法及物理代码均未改。

## 命令与证据

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_fixed_orientation.py tests/test_tripod_sdk_path.py tests/test_tripod_fixed_root_path.py tests/test_tripod_native_tracking_audit.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.prepare_dh116_tripod_fixed_orientation --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-fixed-orientation-20261114-run02 --max-seconds 240
```

结果SHA `b229203e1c18ca1bda3b1804e3f86e2bea5ce0022a107cabbe322b343c85d973`。
原生来源receiptSHA `9bb31008160da84e31c1e089eed925ba658d40c82a830173a958e21491ecf0eb`，
physicalSHA `fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33`。
完整928状态、各约束minima、实际lag来源和候选plan均在私有run02目录。
回归、RED和run01失败日志：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-fixed-orientation-20261114/`。

## 状态与下一步

分析程序与CPU回归通过；**候选未通过滞后筛选**，continuous_tracking_envelope_proven=false，
native_ready=false、teacher_admission=false。不能把通过的理想路径写成原生物理通过。
下一工作包以真实失败形状和原边界调整抓取点/手指路径，
先增加几何裕量，再证明跟踪范围内的路径。不能重跑已消耗的旧配方，也不能仅凭离散角点通过发布原生资格。
Pinch、Tripod、Power完整状态/R0/Reset仍分别门控。WireSeg本包新增训练0步。
成功标准30mm/5s/.5mm与PR2Draft不变。
