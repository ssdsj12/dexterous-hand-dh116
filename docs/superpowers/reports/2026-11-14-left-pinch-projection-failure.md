# 左 Pinch Approach 原生失败前缀审计（2026-11-14）

本包只对已经消耗的一次原生实验作 CPU 复放；没有启动新 Isaac、PPO 或补跑剩余步数。原试验
测试 HEAD `25b4a199c79795b459fb7ca3d955ada4ead855f5`，结果已发布为
`7134f874da44c666aa3e7d4608721c42588d173a`。

## 实际结果及首个失败

- 授权上限234 SDK步、180秒、1环境、0PPO；实际176步（2初始化＋174测量），105.506410秒。
- env_id=0，episode_uid=`dual-pinch-94b46db866bb46eda96ab3301583bb4a:one`。
- 首个失败SDK176，policy21/substep5，时间0.366666667秒。
- 左掌请求偏移Z=-1.181250000mm；原保护计算safe_z=-1.180440967mm。
  实际投影向上修正0.000809033mm，safe_z_clipped=[true,false]，reason=[2,0]。
- SDK目标读回与实际提交均完全一致（两手误差0）。不是SDK漏执行。
- 原joint/mimic/force保护未触发；该步源碰撞净空左2.151626mm、右2.686892mm，仍高于2.1mm。
- 本次实验额外要求“完全不发生目标投影”。这个条件先报
  `frozen both frozen World offsets differs`，随后转为
  `closure_geometry_source_invalid`，正常终止采样。
- 因此不能称作已经发生桌面穿透、joint_limit或SIGSEGV。也不能移除保护来使实验通过。
  旧CPU路径证据明确仅将静止左手误差转移到候选动作，尚未覆盖移动手腕下的投影可行性。

## 接触、几何与缺失信息

SDK3–175有173条完整几何记录；SDK176的物理动作/杆/关节记录仍保存，
但几何计算在目标检查抛错，完整link几何和区域分类缺失，标记unknown，不能从前一帧补写。

全174测量步原始杆接触pair：右食指420、右拇指404、左手0。
SDK3–175原指腹区域内有效pair为0；终止步24个手-杆pair没有区域资格。
不能用这条中止轨迹推断完整闭合后的左右接触结果，也不能借用上一条右手实验的后段数据。

已记录前缀的实际cooked指腹最低桌面净空（mm；各项峰值不保证同一时刻）：

|侧|finger14|finger23|finger33|finger43|finger53|
|---|---:|---:|---:|---:|---:|
|左|3.039740|3.380458|39.119632|56.151764|73.024848|
|右|3.687740|3.350092|32.291481|46.986254|60.034826|

最紧的整体原始碰撞包络并非这些指腹cooked hull。
原始完整shape保护仍保留。精确env/episode/tick保存在报告JSON每个minimum中。

## 生命周期

owned手、ArticulationView、Rope与composer在释放阶段消失；
两侧asset SimulationView在SDK STOP前仍存活、STOP后均不存活；
SimulationContext clear返回，app.close返回。GDB退出信号=null，保存Py_FinalizeEx栈及动态库映射。
外层与inferior退出码均为1，来自已保留诊断异常：**不是自然exit=0通过**。
本结果未获得新生命周期或物理资格。

## 实际代码与测试

新增 `scripts/audit_dh116_left_pinch_projection_failure.py`：
`validate_prefix`绑定begin/action/每个480Hz物理行/end，拒绝跨episode、重复、断行与terminal覆盖；
`projection_decomposition`独立核对所记录safe-z公式、标志、reason和SDK提交；
`lifecycle`将关闭返回与外层零退出分开；
`run`对冻结失败账本作前缀接触、最小净空与缺失区域审计，不允许提升为完整原生通过。

新增 `tests/test_left_pinch_projection_failure.py`。
先运行缺失实现的失败测试（16失败），补实现时发现账本含每个policy的begin/end，
修正解析后16通过，0失败、0跳过，31.65秒。均为CPU测试与既有原生账本复放。

命令（Ubuntu当前项目解释器、无GPU）：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_left_pinch_projection_failure.py -q
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' bash scripts/python.sh scripts/audit_dh116_left_pinch_projection_failure.py --output <fresh-private-output.json>
```

## 证据与后继

原始证据只读保存在Ubuntu私有evidence目录，未上传78MB账本。
- receipt SHA：`25a75775ec43e03f6c722ae395df8584ac149e38c87e18ac77aba52f6d8d745d`
- physical SHA：`1896cdccf9fd8975219d24993661e9e3b3db3442bd0a1fd9639b6362328135f1`
- CPU结果JSON SHA：`27206552adc2858ea7637099626afd5d82b18d6404f51cff4db8e136631ae853`
- 16项CPU日志SHA：`f0c81747ff9a263168b69f7fe9e0ea1b13cc49d4c494746cd47fd81f2858944e`

当前审计能验证记录的投影算术，但没有把记录的safe_z当成独立重算的pre-action碰撞证据。
下一包补齐原投影的真实输入、前一SDK时刻身份和失败时原始几何留存，再建立受原安全投影约束的独立实验合同。
旧精确路径合同和这次失败保留；不静默放宽容差，不更改原物理/关节/2.1mm/30mm/5s/0.5mm标准。
Teacher正式准入仍blocked，Tripod/Power与离线数据工作不由此自动取消。
