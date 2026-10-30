# 连续提升与异常隔离验证 — 2026-10-30

## 实施范围

用户批准三条规则：正常联动偏差记录并连续纠偏，允许 lift；现有角度、速度和扭矩配置由执行器限幅；
无效状态先尝试隔离 reset，复位本身无效或在复位检查期立即复发时才停止。

实现使用 `--continuous-motion`。标准手动启动脚本及 policy PPO 启动脚本默认启用；
直接调用 Python CLI 时仍需显式传入。未启用时保留历史语义。
新模式记入版本化合同；旧模式检查点不能静默混用。

本次是提升执行与异常生命周期修复。GraspXL A 的逐指 25 维观测和新奖励方案尚未实施。

## 代码行为

- 原始 mimic 误差照常记录。新模式解除其单独触发的双手运动否决、0.02 rad 回合失败和0.03 rad全局中止；
  继续使用有界的关节纠偏，并保留默认手动的独立拇指/食指分配、姿态方向及位移/速率预算。
- 原有压力释放与执行器位置、速度、力矩配置保留；没有提高驱动能力、改变质量、接触材料、摩擦或联动模型。
  配置参数仍不是厂家标定能力。
- 有限的手腕跟踪/速度偏差作诊断警告；非有限的手腕位姿、目标或速度读数才进入状态恢复。
  真实适配器不会因单槽坏四元数或力读数在返回测量前中止整批。
- 有限的归一化动作越界会饱和；NaN/Inf 不会被限幅伪装成合法动作。
- 无效物理或观测状态只结束受影响回合，保留一次失败惩罚和最后有限任务快照；
  在共享世界下一物理子步之前复位对应槽。
- 初始、即时 reset、复位后的检查期和周期末验证使用完整状态检查：
  原始物理量及派生竖直力有限、重量为正、接触载荷非负并满足分组/合计一致性、策略 float32 观测有限。
- 新模式下可恢复的手动失败继续 Run，保留提升目标并清零旧提升累计值。
  用户主动的 Reset+Pause 仍保持暂停。无法恢复的异常向外报告，GUI 不启动第二层无限 reset。
- 非有限诊断保存为明确标签，原始测量不改写。
- PPO采集对异常动作、value、log-prob及已恢复的无效物理状态隔离处理；清除整个部分rollout和待写transition，
  完整重采后才计算GAE，不跨异常间隙计算信用。策略参数本身无法恢复时停止，不修改权重伪装成恢复。
  实际回合失败统计包含丢弃片段，奖励统计只计有效的完整rollout。
- 启动配置、tensor运行合同和checkpoint记录同一连续模式身份与复位规则；tensor专属故障/警告字段有独立启动校验。

## 测试证据

修复前已有 143 项定向回归通过。新测试先对未修改的相关路径执行，观察到正常 mimic 导致运动否决/失败、
非有限状态直接全局中止，以及 reset 验证缺漏等预期失败；再修改生产代码。

最终相关 pytest：**27 个文件，493 passed in 11.66s**。
涵盖 169 个新增控制、适配器、状态及PPO采集恢复用例，以及既有 Task/Env/World、手动、force-region、hold、奖励与训练合同回归。
PPO采集恢复单独77项通过；复位后持续无效packet的5项回归先在修复前失败，再验证恢复失败只停止一次，不进入reset循环。独立规格和质量复审已关闭本次报告的P1/P2问题。

可复查的文件列表与命令：
`/root/hand/runs/deployment/continuous_motion_20261030/final_regression_files.json`、
`final_regression.log`。
命令为 `bash scripts/python.sh -m pytest -q <final_regression_files.json 中的27个路径> --tb=short`。
两个修改过的 shell 启动脚本的 `bash -n` 通过。

测试保留了关闭新模式时的旧行为，并覆盖一次失败、复位失败、立即复发、未受影响环境继续及有限的终止观测。

## 真实 Isaac 对照

实际运行均串行执行，采用 GPU/PGS，无策略网络优化或 PPO 更新。
服务器：`a21681897614602240331160`，Isaac Sim4.5 / RTX4090。
诊断结束后进程正常退出；不启动长训练或图形窗口。

### 单环境：原抓握路径的 10 mm 提升

诊断：`deploy/diagnose_continuous_motion.py`，复用此前的场景和手动指令发生器。

- 圆杆是动态独立刚体，不是运动学物体；Wire/Table之间没有 joint。
- 自由下落对照：0.05 仿真秒下落 12.7734 mm，桌面位置不变。
- 绕过抓握控制的空手驱动：指令 10 mm，左右实际均约 10 mm，原执行器及独立物理检查保留。
- 当前 5 N/手抓握控制路径：150 个控制步、1200 个物理子步，未发生回合终止/reset；
  左右手腕最终实际为 10.000004 / 10.000003 mm。
- 记录到 441 次达到0.012 rad联动诊断线的更新，仍能推进。
  18 次阻塞属于过载退让；首次阻塞时右手约15.21 N，mimic约0.00999 rad。
- 实测物体离桌约为0，结束时两手接触力均为0：手已提升，但物体没有随手抬起。
  **不是稳定抓取或物体提升成功。**
- 选定主动关节力矩配置仍为0.4 Nm；没有为通过此测试增大驱动上限。

原始报告：
`/root/hand/runs/continuous_motion_lift_20261030/lift_binding_diagnosis.json`。

### 两环境：一次无效观测的隔离恢复

诊断：`deploy/probe_continuous_recovery.py`。
只在第3个子步的返回测量中给环境0的 object_relative 注入一个NaN；
不是向PhysX求解器内部写NaN，也不代表已验证所有GPU内核故障的恢复。

- 12控制步、96物理步完成。
- 隔离 reset 列表仅为 `[[0]]`；回合数与恢复数均为 `[1,0]`。
- 环境1继续物理运行；返回策略观测和奖励有限。
- numerical_resets=1，PPO更新次数=0，grasp_qualified=False。

原始报告：
`/root/hand/runs/continuous_motion_recovery_20261030/recovery_probe.json`。
该报告复用了部署诊断模板的 source_commit 字段，它表示模板基线763969b；
实际交付源码由部署回执中的逐文件 SHA-256 识别，不把模板基线当作本次修复提交。

### 两环境：返回测量前的适配器故障

诊断：`deploy/probe_continuous_adapter_recovery_final.py`。最终版本在复位读回时也检查手腕速度与物体参考姿态；在SDK缓存读数中依次注入：
第3子步环境0的物体四元数NaN、第35子步环境0的水平摩擦力NaN、
第67子步环境0的手腕线速度NaN。没有通过状态写入API向求解器发送NaN。

- 故障位分别为 `[1,0]`、`[4,0]`、`[8,0]`，确认进入了实际几何、力及手腕适配器路径。
- 96物理步完成，隔离reset列表 `[[0],[0],[0]]`；回合数/恢复数均 `[3,0]`。
- 环境1继续运行，策略观测和奖励有限，PPO更新0。
- 这验证了SDK缓存读数异常的恢复，不宣称可以恢复全部PhysX/GPU内核错误。

原始报告：
`/root/hand/runs/continuous_adapter_recovery_final_20261030/adapter_recovery_probe.json`。

### 两环境：真实CUDA PPO采集与异常隔离

诊断：`deploy/probe_continuous_policy_recovery.py`，使用实际RSL-RL3.1.3算法和GPU物理环境，禁止optimizer.step与algorithm.update。

- 实际执行训练入口的 `validate_world_adapter`：host参考误差在2e-6以内，非对称映射及选中环境复位隔离通过；新增tensor字段独立验证通过。
- 两轮各保留3步的完整采集。第一轮第二步注入环境0动作NaN；第二轮在返回测量中注入环境0物体速度NaN。
- 共10控制步、80物理步；隔离reset恰为 `[[0],[0]]`，恢复数 `[2,0]`，环境1继续。
- 2个部分rollout被丢弃，共4个控制步／8条环境transition；两个完整新rollout才进入有限的GAE计算。
- `invalid_action`与`nonfinite_state`失败计数各1；已完成物理恢复的状态没有触发额外NaN marker或重复reset。
- 策略state_dict逐张量相等、optimizer状态未改变，PPO更新0。
- 该测试验证采集和恢复连接，不是一次策略优化或合格抓取验收。

原始报告：`/root/hand/runs/continuous_policy_recovery_20261030/policy_recovery_probe.json`。
所有复用诊断模板的source_commit字段都表示模板基线；部署源码由交付回执识别。

## 保存与限制

服务器修改前逐文件核对HEAD原件并保存：
`runs/deployment/continuous_motion_20261030/source_before.tar`（11文件）、
`source_extra_before.tar`（2文件）及基线哈希清单。
手腕适配器修改前另保存 `source_adapter_before.tar`（1文件）。
未修改已有服务、SDK依赖位置、资产迁移manifest或旧训练数据。
原始日志和报告留在runs中，不上传到GitHub。
最终文件SHA-256（统一LF比较）、本地提交、CPU/物理报告及进程/服务核对保存到
`/root/hand/runs/deployment/continuous_motion_20261030/delivery_receipt.json`，本机同时留存。
服务器Git HEAD与部署源码分开记录，避免把工作树部署误报为远端分支提交。

当前可确认：联动诊断不再单独截断提升，手腕可以完成10mm命令，异常隔离路径已实际运行。
仍需解决：双手实际夹持力、四指触点及提升中的持续抓握。0.4Nm及手腕驱动配置未完成硬件标定。
