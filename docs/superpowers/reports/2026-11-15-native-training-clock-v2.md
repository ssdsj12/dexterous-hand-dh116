# Long Clock v2：W5 Task 1 CPU 与真实来源审计

日期：2026-11-15；基线 `de02ae7a0bef6c87185fc698f6cd3a66db90b2e3`。
独立分支 `codex/long-clock-v2-20261115`，Ubuntu Worktree
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/long-clock-v2-20261115`。

## 范围

仅执行已批准 long-reset-v2 plan 的 Task 1。新增 clock policy、SDK/rod
只读数学验证、测试和本报告；不修改旧 clock、控制器、Reset、PPO、
物理、材料、执行器或成功标准。未启动 App/Isaac/GPU/native，未创建新
Reset owner、执行新物理步或更新模型。Task 2–5、消费者兼容表和真实长程
Reset 仍未实现/验收，不能将本报告称为长 Reset 通过。

## API 和数学合同

文件 `src/go2_pvcnn/control/dh116/tabletop_native_training_clock.py`：

- Frozen `ClockPolicyV2(max_sdk_step, timestamp_mode)`：exact 正整数上界，
  拒绝 bool；模式仅 `fp32_dt_float64_sum` / `float64_dt_float64_sum`。
  `sha256` 覆盖范围、模式、480Hz DT、版本和包络算法身份。
  范围是数学输入，既不是运行授权，也不证明所选模式属于真实 SDK。
- Frozen `SdkClockV2(sdk_step_abs, timestamp_s, dt_s, tolerance_s, policy_sha256)`。
- `validate_sdk_clock(*, step, timestamp_s, dt_s, policy)`：exact 非负整数
  SDK 绝对步不超过上界；实际 DT 必须是 `1/480`；拒绝 bool、非有限、
  负时间和超范围输入。保留收到的实际 timestamp，不用 `step*DT` 覆盖。
- `read_clock_v2(sim, rod, *, policy)`：读 SDK→owned rod.clock→再次读 SDK，
  两个端点完全一致才返回。复制 rod 身份、实际 solver_time、rod_tick，
  不从 SDK 反推 solver clock。构造期 rod 的 `None/None/False` 原样保留；
  已注册但 poisoned 的 False 不被升级为 True。异常 finally 清本函数借用
  sim/rod 引用；无 stepping、setter、全局 monkeypatch 或 native ownership。

包络使用 `u=eps64/2`，`n=step`，`term=float(np.float32(dt))` 或 `dt`：

~~~text
gamma_n = n*u / (1-n*u)
tolerance = n*abs(term-dt) + gamma_n*n*abs(term)
          + 8*eps64*max(1,n*dt)
~~~

必须 `n*u<1`、包络小于 `dt/2`、实际时间对名义计数的残差不超过包络。
旧 `2e-7` 上限没有被全局放宽；这是独立数学合同。
同一次 read 拒绝 SDK step、timestamp 或 dt 在 rod capture 中发生变化。
跨多次调用的连续性、Reset epoch、episode elapsed owner 和实际 source
资格由后继链路验证；本函数不创建新的时间原点。

## 安装源码只读核对

私有 `clock-source-map.json` 包含原始路径、SHA256 和源码摘录，没有复制安装源码进仓库。

| 实际安装文件 | SHA256 | 结论 |
| --- | --- | --- |
| SimulationContext `simulation_context.py` | `0be3b0eaa44e6110cec520be9ac2c08cea17ba97db14b1b0aa60d87824deceb3` | current_time 返回 Python accumulator；callback line1360–1362 执行 `_current_time += step_size` 和整数 step +=1 |
| PhysX `_physx.pyi` | `b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9` | Callable[[float],None] 不是 native callback 数值精度的证明 |
| SimulationManager `simulation_manager.py` | `97d1ef88202f370eac00281b193581ba02b5c95ed64131f153c006e3ab104357` | get_physics_dt 来自配置频率，不能仅凭 Python 类型确定底层 callback 精度 |

## 已有原生短窗原始读数，未新增原生运行

只读来源是旧 `approved-numeric-reset-pinch-20261114-xkec5s6v`
批次、`attempt-iwm7090o/native/`。该批次结果仍为
`process_exit_verified=false`；本包不修改该失败状态，也未重跑其预算。
原文件通过现有私有服务器路径引用，不复制原始账本到 GitHub。

| 文件 | SHA256 |
| --- | --- |
| `sdk_clock.jsonl` | `b4af5c5d6fba1c6e311cdf1ca0c96aa4e0ffbc922ce8a7ccef44ca80bfba53e6` |
| `result.json` | `209a4395253d8f91a4345271e0cfe87355565f5c457057fafcfdec2e1eb9589e` |
| `mesh_reset_numeric_state.json` | `0b3f96e0290c4515e50c228e89eab72326ff0cc9d77c201301e2ab24e41c7e5e` |

SDK JSONL 完整末尾换行，**232 行，连续 SDK step3→234**。
需严格分开两个原始字段：

| 端点 | 实读 `sdk_time_s` | 名义 `timestamp_s=step*DT` | 实读减名义 |
| --- | --- | --- | --- |
| 3 | `0.006250000325962901` | `0.00625` | `3.2596290076847279e-10` s |
| 234 | `0.4875000254251063` | `0.4875` | `2.5425106298104794e-08` s |

本窗口实读对 `float(np.float32(DT))` 累计的最大残差为
`0.0` s；
对 repeated float64 DT 累计的最大残差为
`2.5425107408327818e-08` s。
numeric Reset clock 的 step2 实读 SDK time 为
`0.004166666883975267`，原 rod clock 名义 timestamp 为
`0.004166666666666667`，solver_time 为0。三者未混用或回算。

这些数据支持**该旧短窗数值符合 FP32 callback term 累计**，不能据此
声称当前 backend/新 worker 的长程模式已获资格；源锁、环境、原始身份、
新长程范围及自然退出仍须对应证据。Task 4 必须按实际 source 路径核验
所有连续消费者，不能将任一名义时间字段当 SDK 实读而掩盖兼容问题。

## TDD、回归和失败证据

私有证据目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/long-clock-v2-20261115`。

| 日志 | 真实结果 |
| --- | --- |
| `task1-red.log` | 89 failed / 1 passed；新模块缺失，旧 v1 258步拒绝仍通过 |
| `task1-green.log` | 90 passed，2.13s |
| `endpoint-read-red.log` | 3 failed / 90 deselected；SDK读取竞争被真实捕获 |
| `clock-old-reset-regression.log` | exit4，错误引用不存在的 readback 测试文件，未运行测试；日志保留 |
| `clock-existing-reset-regression.log` | 161 passed，15.68s；此首次组合日志已完整，但外层状态未单独保存 |
| `task1-final-regression.log` + `.json` | **161 passed，16.36s；外层 exit0**，93新 clock +68旧 Reset；完整状态与SHA保留 |

最终日志 SHA256：`92b1f34c454b13806937c8889676f5a4f2336662a81c967b227223b244da877b`。
clock source-map SHA256：`9a927be6a7f225bed95df05fda752d4eed3d8f23339c0677ef82818e5d8a0032`。

最终执行：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_training_clock.py tests/test_mesh_reset_order.py tests/test_mesh_reset_numeric_state.py tests/test_dual_hand_reset.py
~~~

覆盖两种模式的0/1/240/241/258/2402/9600边界、超 max、bool/NaN/负值、
错误 DT、越误差包络、不可用超长包络、策略模式身份、owned快照、
constructor unknown、poison、SDK/rod步不一致、capture期间时钟变化、
SDK绝对步不归零，以及以实际 Reset 端点差计算 elapsed 的 CPU例子。
v1 Reset 及其 order/readback/numeric 代码和保护均未改动。

既有 Repository(...).verify_protection() 在最终回归前后及提交前执行，
528个保护源码和2个历史 Checkpoint SHA均通过；私有 protection-final.json
保存清单身份，未覆盖历史模型或原始数据。git diff --check 通过。

## 交接与剩余门

Task1 数学和只读 source audit 已实现；后续仍需：

1. Task2 独立 order/readback/numeric/dual Reset 完整链。
2. Task3 真正推进至240/258/>2400后零步 Reset、下一episode和预算续接；
   原 ClearanceGuardedExchange 及2.1mm保护不丢失。
3. Task4 实际消费者兼容表、源SHA、同一真实SDK累积模式资格。
4. C06 actual provider、resolved S0、同owner实际来源、独立预算和资源门。

没有新原生、抓取、性能或 PPO 资格；旧标准30mm/5s/0.5mm完全不变。
共享 README/CHANGELOG/Registry/readiness 与 Windows发布由root集成；
本任务只在 Ubuntu 对精确三文件独立提交，等待 SPEC/QUALITY。
