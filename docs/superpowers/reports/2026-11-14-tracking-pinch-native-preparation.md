# 2026-11-14 Tracking-aware Pinch 原生配方准备

本包基线 `b5b0cff76be09a7b293604104959e49837139134`。仅建立独立、可审计的接触候选执行路径；没有产生新的原生接触、抓取或 Teacher 资格。

## 修改范围

- `prepare_dh116_tracking_pinch.py`：以既有 correlated-tracking CPU 证明派生独立 Reset 和动作表。
- `tabletop_tracking_pinch_geometry.py`：复用原完整 cooked geometry、安全投影和有效指腹验证；独立绑定候选，不修改历史 plan 模块。
- `prepare_dh116_tracking_pinch_native.py`：复用原234步、真实参数/材料 director、原生命周期所有权；仅改变当前候选。
- `verify_dh116_tracking_pinch.py`：冻结新来源、原真实 scheduler parent 与独立 worker 身份。
- `run_dh116_tracking_pinch_work_packages.py`：原单实例/GPU预算门控与完整退出后审计，不复制调度器。
- 新 task/execution JSON 与四个测试文件；不修改原 controller、资产、材料、关节限位或历史模型。

## 候选与限制

右手相对原末端位姿平移 [-0.197616, +1.991211, +0.847599] mm；姿态、左手与自由杆保持原候选定义。右手主动关节初值采用冻结 CPU 解，接近期间主动关节 delta 与原方案逐项相同。腕部X最大100µm、最大速度1.14mm/s，Z总接近0.5mm；原动作24D顺序及单位不变。

来源 `pinch-tracking-target-20261114/path-04/result.json` SHA：
`b4a4d2a95ae97437c182a800f01f0862180e746f90149adc924e3b4fa71da600`。

既有 CPU 证明：1025点路径最小桌面净空2.121758459mm、非目标净空0.006758322mm、所需SDF最低-0.291258791mm；232条历史同钟残差场景最小桌面净空2.155937751mm、非目标净空0.002650453mm。这些是候选预检，不能当成本轮新物理测量或接触成功。finger12 初值离下界仅约5.82µrad，仍有实际跟踪风险；没有改变保护。

## 数据与生命周期

成功采集后，进程内仅封存完整 physical.jsonl 的SHA、字节数和行数，明确 pending_post_exit。原 App 正常关闭后，独立完整重放审计每条动作、参数、director、物理时钟、投影、geometry、实际退出码。新增 `tracking_pinch_scope.json` 证明封存/clock作用域已恢复，不能用CPU夹具或单个 exit marker替代自然退出。

未完整完成的执行保留原终止审计，不封存为成功；封存失败、记录变化、标记缺失、预算改变均拒绝资格。完整审计不裁剪账本，不省略接触校验。

## 测试与失败证据

Ubuntu现有Python，CUDA_VISIBLE_DEVICES为空，不启动Isaac。测试命令：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_tracking_pinch_plan.py tests/test_tracking_pinch_native.py tests/test_tracking_pinch_worker.py tests/test_tracking_pinch_owner.py tests/test_effective_right_pinch_plan.py tests/test_effective_right_pinch_geometry.py tests/test_owned_clock_projection.py tests/test_deferred_physical_audit.py -q
```

新增测试先失败：plan模块缺失5例、native模块缺失8例。首批夹具还发现预算测试缺必要entry/source身份，以及真实CPU子进程缺新增恢复标记；保留原失败日志后修正夹具，不削弱生产验证。修正后独立worker回归15项通过（20.52秒）。修正后的完整源码回归88项通过（158.06秒），0失败/0跳过。集成环境仍会重新运行相同88项；计数、退出码和日志SHA由Registry/readiness记录。

证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tracking-pinch-native-20261114`。原生次数0、物理步0、PPO更新0；CPU完整owner的合成SDK缓冲不能计入原生。

## 后续执行与门槛

冻结新配方后，依据本聊天“后续自动批准”的有限诊断授权，独立输出目录、GPU3、单Isaac、1环境、0PPO，最多一次234总SDK步（2初始化+232测量）、180秒。原参数任务父回执SHA `f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`，必须读取真实回执。

不重跑已耗尽的旧实验。原生进程墙钟与退出后CPU审计时间分别记录；失败保存终止账本、GDB栈、动态库映射，不重新赋予预算。即使234步接触诊断通过，也不是5秒Hold或30mmLift成功，不能开始正式Teacher训练。左侧有效承托、双侧接触、材料锚Slip、完整R0及三抓法资格继续分别门控；WireSeg本包新增训练步数0。
