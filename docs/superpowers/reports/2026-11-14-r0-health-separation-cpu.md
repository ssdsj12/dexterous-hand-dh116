# R0 数值健康、原保护与 Hold 运动的独立 CPU 合同

## 范围与交付

基线：3dcb25096dc243223fdefad82f17a4df0eaa70d5。
独立 worktree：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/r0-health-v2-20261114，
branch：codex/flexible-r0-health-v2-20261114。

本包仅增加 opt-in 模块、JSON 合同和两个 CPU 测试文件。旧 R0、原控制器、旧原生配方、
材料、动作、安全门槛和成功标准均未改；默认不开启。没有 Isaac、CUDA solver、PPO、
新原生预算或训练运行。按主线程分工保持未提交，由主线程审查后统一提交、集成、发布。

新增路径：

- src/go2_pvcnn/control/dh116/tabletop_r0_health_separation.py
- configs/dh116_tasks/flexible_r0_health_separated_v2.json
- tests/test_r0_health_separation.py
- tests/test_r0_numerical_health_source.py
- 本报告

## 原定义冲突与可复用边界

在 tabletop_deformable_grasp_task.py:474–479，stable 是 Hold 的必要条件；
同文件 :509 又把 not stable 定义为 instability。
tabletop_deformable_control_step.py:146–152 则在第一次 instability=1 时立即终止。
因此，若把“尚未满足成功 Hold”当作 stable=False，正常 Approach 或负接触也会提前结束。

刚体原标准不是通用数值健康条件：

- tabletop_tensor_task.py:646 的 stable 同时要求底部抬高30mm、刚体根线速度<0.03m/s、
  slip 与承托条件；:675–676 只把它用于 Lift 成功计时。
- surface_force_sharing 路径走 tabletop_surface_success.py:109–117 的有限性、
  30mm 抬升、刚体根速度和深度检查，并在 :151–177 检查原对向承托、
  普通接触比例、固定锚位移和连续5秒窗口。
- tabletop_lift_hold_window.py 的连续窗口与固定参考算法可以复用其语义，
  但刚体 pose 派生 marker 不自动成为柔性杆的固定材料点。
- 原 joint_safety_failure / force guard / 2.1mm source-clearance 仍可按原值重算；
  无 joint guard 只说明这一检查未失败，不能推出运动稳定。

本包不采用刚体0.03m/s作为柔性阈值，不把节点最大速度、端点位移或几何应变
转换为柔性 stable。原30mm/5s/0.5mm成功条件和所有原 R0 系数不变。

## 新合同和真实接入 API

VERSION 为 dh116-flexible-r0-health-separated-v2；三项独立记录：

| 项 | 定义与来源 | 未知条件 |
|---|---|---|
| numerical_health | 13个实际端点数组全部有限，且同一个 RopeBatch 的显式整数 last_error=0 | 缺 getter、getter 异常/类型错、缺同 owner 锚时 NA |
| constraint_health | 原 joint/mimic、30N force guard 和原2.1mm source-clearance validator | 非有限输入不进入比较，值保留 NA；source/clock/reason 损坏直接拒绝 |
| hold_motion | 独立、经过限定的运动条件 | 当前柔性原生无已准入来源，因此 NA；仅 CPU fixture 可输入显式真假来测试旧 Hold 行为 |

13个数组为 rod p/v、两手 q/qd/backend_targets、两手每个原5指腹的 force/torque、
以及 rod distributed_contact。真实 shape 分别是 [1,41,3]、[1,17]、
[1,5,3] 和 [1,41,3]。NaN 的 norm 比较不会被当作保护通过。
只计算 actual active_indices 的 qd 诊断；被动关节 backend_targets 不是命令目标。
以上有限性是数值读回健康，不是求解器收敛或材料运动稳定证明。

capture_solver_health(runtime, backend=..., sim=..., expected_clock=...) 可由后继
Exchange 的同钟 observer 调用。它验证实际 DualHandRodRuntime/FreeRodRuntime、
backend.RopeBatch 类型、已锁实际 .so 路径及 SHA、实际 owner、env1/41节点，
在 getter 前后核对同一 SDK/rod 时钟和 backend/owner。SDK 时间使用既有2e-7秒
误差界；solver_time 使用同次实际 recorded time，不要求与理想 tick 时间逐 bit 相等。
只读取已有 host last_error getter，不 step、不 gather、不执行资源启动。
getter 异常只保存纯字符串与 NA；finally 清除 runtime/backend/sim/rope/closure alias。

公开 make_solver_status 只接受显式 CPU fixture；调用者不能把传入整数0贴成 native。
真实 native_getter packet 还必须和 exchange.material_director 的已记录
NativeParameterEndpoints 版本、同钟身份、实际 source_instance_id 及 backend SHA 一致；
缺锚保持未知，异实例拒绝。摘要 SHA 用于复算一致性，不是独立的来源资格。

observe_health 是纯 owned-data 观察器；validate_health 从原 exchange 重新计算，
拒绝被替换的状态、clock、source 或结果。所有 native_readback_qualified、
native_qualification、teacher_admission 固定为 false。source_scope 字符串不授予资格。
本包未给任何已消耗原生配方自动加观察器，也未创建新配方。

## 奖励、终止与统计

HealthSeparatedR0Episode 复用原 NecessaryHoldR0Episode 和旧奖励项：
数值健康、原约束和独立运动三值合取只控制 Hold；instability 项单独来自 numerical_health。

- 已知健康、原保护通过、已证明缺少双侧必要接触：hold_motion 及 Hold raw 仍可 NA；
  既有 NecessaryHold 合同已证明不适用的 Hold contribution 为0，数值 instability 为0。
  在 reset approach 基准、其他奖励来源与显式诊断系数均齐全时，可结算正常8子步
  Approach/负接触动作，不要求先成功抓取。
- 这是条件性诊断结算。原首 Approach 基准和非 bilateral slip 的适用性继续由既有
  applicability 合同处理；本包不补造缺失基准、滑移或外部支撑。
- 健康=False 或原约束失败时，即使 CPU hold_motion=True，也不能开始/继续 Hold。
- 已知数值错误在首个实际样本终止；实际约束失败保留原 physical terminal。
  没有给原约束错误另加一份数值 instability 惩罚。
- geometry/table/director source-invalid 等已记录最终 physical terminal 也必须在首个样本
  结束，不能被 original_guard_failure=None 覆盖。数值健康可仍为True，
  但 reward 被 physical_source_validity 阻断；不会把缺源当实测约束违例或数值错误。
- 原 clearance 的 source SHA、SDK时间或 terminal reason 损坏直接拒绝。
  不允许吞成 constraint=None 后用负接触不适用项生成完整 reward。
- 继承原 control reducer 的 terminated/truncated、单次 settle 和预算关闭语义；
  physical terminal 后拒绝下一动作。没有新增 GAE、Reset 或观测66/371实现。

super.step 曾更新旧 instability 统计，所以本包先保存旧统计对象，
调用原项后恢复该对象，再只用新数值项更新一次；同时重算 raw/contribution、
term_stats、blocked_terms、diagnostic_reward。避免返回项改了而累计统计仍混入 motion。
source-invalid 的整个控制区间也保持 reward=None，不能被父 reducer 的求和覆盖。

所有正式 training coefficients 继续为 null；训练奖励仍 None。
测试中的完整非负 coefficients 是标为 CPU 的显式诊断输入，不是默认训练权重。
对非有限 rod p/v，观察器可记录 numerical_health=False，但旧 snapshot validator
仍严格拒绝它；不保证这类损坏状态可形成标量 R0 transition，也不替换 NaN 以生成惩罚。

## 实际旧账本回放：有健康所需数据，但没有持久化 last_error

本包读取已消费 projected Pinch 真实账本，未增加物理步：

- receipt SHA：04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3
- physical.jsonl SHA：9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4
- 原执行234总SDK步（2 setup +232 measurement）、0 PPO、163.496670751秒、
  自然 outer/inferior0；仅既有 bounded diagnostic 资格。
- 本包 module fixture 流式 SHA 校验整个文件与232个 physical rows，只解码SDK3/176/234。
  三个实际 exchange 均通过真实 [1,5,3] wrench、13路有限性、原 guard、
  source-clearance 和 validate contract；没有给旧数据补 last_error=0。
  numerical_health=None、constraint_health=True、hold_motion=None、original_stable=None。

独立 physics agent 只读全232行还确认：没有 last_error/solver_error 等持久化字段。
旧 native parameter reader 在运行时检查 getter 成功，不等于账本已保存可重放的0。
实际使用 backend SHA 为 2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec；
现有 bindings 的 host getter 可复用，但本包只进行了 CPU 形状/调用边界验证。

其独立运动汇总（不是本包重跑或稳定判定）：全段 max node speed
0.0179633813m/s，末帧0.00734426324m/s；reset至末帧最大节点位移0.402734914mm。
实际没有30mm抬升。这些端点统计不能替代固定材料滑移或连续 Hold 资格，
更不能因低于刚体0.03m/s而填 stable=True。

## TDD、回归与独立复核

最终命令：

DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short
tests/test_r0_health_separation.py tests/test_r0_numerical_health_source.py
tests/test_r0_hold_necessary_domain.py tests/test_dh116_r0_applicability.py
tests/test_dh116_r0_applicability_history.py tests/test_dh116_deformable_grasp_task.py
tests/test_dh116_deformable_control_step.py tests/test_flexible_parameter_readback.py

结果：155 passed，24.05秒（47新 +108旧）。
Repository.verify_protection() 成功，530个原保护指纹保持不变。

开发中每个具体缺口均先有失败断言：旧 hold/instability冲突、坏 health 不能 Hold、
backend/owner同钟变化、异实例 source、最终 source terminal丢失、NaN wrench、
真实 batched wrench 以及吞掉 source/clock/reason 异常。
保留的 red-terminal-force-shape 日志有17失败/21通过，定位 capture_errors 的局部变量
覆盖；修正为独立局部名。真实账本 shape 的3失败以及 source完整性的3失败分别保留。
最终全部转绿，不把旧失败标成成功证据。

physics agent 独立只读审查曾发现 owner锚、最终terminal、变量覆盖和真实 batch形状，
均已修复；最终源码 SHA50ddb2b2f8b4491765df7651ee3c0cfaa58b9a54ea883e5d5b5888be913f6285
静态复核没有剩余已识别阻断。其实际末帧与 source-terminal 补充执行属于先前
294710e...中间 SHA；最终改动仅将原 validator 异常严格上抛，最后复核未重跑155项。
不把同行审查当本包原生验证。

## 文件与证据

最终源码 SHA：50ddb2b2f8b4491765df7651ee3c0cfaa58b9a54ea883e5d5b5888be913f6285
配置 SHA：f2f4d65f78dee461721c02f9bbc61738540038197a13fb2921997d94ad6f25cb
episode/reducer tests SHA：ed452a00765f005c784a1bd5271c54423ba1482c065a064b538b51e50b07d5d8
source/real-ledger tests SHA：7fb9f3acb1917a8ae460e0027673de431cd12a97584fb32b582dd72430df8604

私有日志目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-health-v2-cpu-doc0yi9g

日志 SHA：

- new-suite-01.txt: afa6b243f67baeaa9693426a7832a76240a0318503604ba3b2ec45d580aaa754
- protection.txt: 41edfd5ca62eac8f920c8ff24fe7c51a3d6708ca673572d503f7e3002717c874
- red-invalid-constraint-source.txt: a3e29dc1d363eda68e067c8f223eab5edffc942e26e83cf886ddfbdc87869bbd
- red-real-ledger-shape.txt: bcfa17bd7f2544683c4a4f5a6df08c7109d0b0dac8ecbc0fd8ed290fd837efd7
- red-terminal-force-shape.txt: f780a530069f1cd67121c79e81f1d47ad2cdc52ca84f806b9d44d35b5d0fa086
- regression-01.txt: dba5b1b420119a2fb9dce3ae566b5cc2b2b68dd4b9b0ed5484bc46c191c56711
- regression-command.json: c644e88d14e549f6ec2cb68053f7e3c59f81acc239141d5db5d1b52fe9d6c996

## 后继实际连接与仍未解除的准入

主线程可在新的独立 opt-in Exchange Mixin 中，在既有 _step 返回的完整 boundary 后调用
capture_solver_health，并把 owned packet 与 observe_health 结果存入新的健康账本；
独立退出后 validator 用原 physical rows、实际 owner/source/clock 逐条复算。
观察器失败必须保存实际失败并结束，不能补0、丢步或改动作。
这需要新的精确代码/配置冻结与原生结果，不能借用或重跑旧234步预算。
本包没有实现该新 launcher，也没有改变任何既有关闭路径。

尚缺：实际同钟 getter health ledger 的原生资格、柔性 Hold运动来源与适用定义、
完整原成功窗口以及正式训练系数选择/资格。新合同解除的是
“正常 Approach 必须先满足成功 Hold 才能诊断结算”的语义冲突，
不是宣布 Teacher PPO 已可训练。
