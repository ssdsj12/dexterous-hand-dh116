# Flexible R0：CPU 控制步结算准备

本包基于 07db9eb461f8db067beb294c12c077d1ea1b824e，在独立 Ubuntu worktree 完成，只新增 reducer、CPU 测试和本报告。原 reward/task/budget、冻结配置、模型和 checkpoint 未改；没有接入环境、启动 Isaac/solver/GPU 或训练。新增物理步 0，optimizer updates 0。

## 实现边界

FlexibleGraspEpisode 已逐个 480 Hz 物理样本校验 rod、episode/reset、固定材料参考、滑移、hold 和原 R0 raw terms；plan_flexible_experiment 已分别计算 setup、measured 与 total 预算。新增 tabletop_deformable_control_step.py 复用两者，补充 60 Hz 单控制步的结算状态机：

1. 创建实例时必须显式传入十项完整、有限、非负 CPU diagnostic coefficients。原任务配置中的 NA 不被覆盖。缺项、None、bool、NaN 均拒绝；没有默认权重。
2. 只接受 setup-aware v2、无更新的 frozen-eval 预算，setup 数量及 solver substep 数必须已知。配方 SHA 绑定 task、grasp、预算、诊断系数、聚合/终止规则以及 reducer/task/budget 三个实际源码 SHA。Pinch、Tripod、Power 身份不同，调用方修改原 dict 不会改变实例。
3. 单实例只接受一次 reset，reset 的物理时钟必须与声明的 setup 数相符。begin() 绑定顺序 control_index、配方 SHA 与一份独占 action24 副本；action 只声明，不由 reducer 下发。
4. push() 复用原 Episode 接纳连续的物理样本，并检查该配方的 rod substep stride。通常必须完整接纳 8 行才能 settle()。待结算时拒绝下一物理行或下一控制步；同一个窗口仅能结算一次。
5. 原 Episode 的首个 success_candidate，或已知 joint_limit、instability、prohibited-support 标志会触发本诊断 reducer 的提前 terminal。此时结算实际接纳的 1–8 行，不继续消费，也不补零凑满窗口。桌面接触在接近阶段不自动成为 terminal。此保守诊断终止规则是新增独立逻辑，**没有替换或批准原生环境的终止合同**。
6. 漏步、错 episode/reset、错误测量身份或非有限奖励使物理窗口失效；不能修补或重放后结算。配方/control token 不匹配在消费前拒绝。预算耗尽产生 truncated，禁止继续 begin 或通过第二次 reset 复用该实例预算。

exactly-once 的范围是**单进程中单 reducer 实例的消费与结算**，没有跨崩溃持久化或分布式去重承诺。新的实例也不构成新的原生预算授权。

## 奖励与 terminal 的含义

聚合规则为已有 per-physics contribution 的逐项求和，再求控制步总和。没有改符号、物理尺度、raw term 定义，也没有推定获批的控制步训练权重。每项保留 expected_samples、valid_samples、raw_sum、contribution_sum、来源与显式系数。

任一物理行的某项 raw 未知，控制步 diagnostic_reward 就是 None，并列出 blocked_terms；即使该项测试权重为 0，也不能把未知当成 0。第一物理样本的 approach progress 没有前一测量，因此首窗口正常保留该未知。training_reward 始终 None，native_qualification 始终 false；CPU success_candidate 不构成真实抓取成功或 PPO 准入。

数值溢出同样使窗口失效，不能输出 Infinity 或绕过未知项。此包只准备结算逻辑，真实 R0 系数、完整来源、371D/Reset、接触/支撑、原生执行预算仍须各自满足。

## 验证与证据

先新增待实现测试，确认因缺少模块失败；实现后测试通过。再先增加 helper SHA 配方绑定失败测试，随后补实现。最终 **23 项新增 CPU 测试，相关总计 78 passed / 11.39s，退出码 0**。

覆盖完整窗口、重复 settle、提前结算拒绝、下一步越界、漏步/错 episode/测量身份、跨配方与 control、三 grasp 隔离、输入别名、系数缺失、未知 raw 与零系数、已知故障提前终止、原有五秒成功首端点、预算截断、第二 reset、有限物理项累积溢出和实现 SHA 变化。

- 原有五秒 hold fixture：第 2401 个物理样本（全局 step 2403）首次 success_candidate，control_index=300 的窗口只接纳 1 行即 terminal；后续结算和推进被拒绝。
- 预算 fixture：360 个控制步、2880 个 measured 样本，声明的 setup=2，总账 2882；最后控制步 truncated。
- 以上都是 CPU 输入样本计数，不是本包新增原生物理步。

~~~sh
PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformable_control_step.py \
  tests/test_dh116_deformable_grasp_task.py \
  tests/test_dh116_deformable_task_budget.py \
  tests/test_dh116_deformable_privileged_teacher.py \
  tests/test_dh116_deformable_privileged_ppo.py
~~~

私有 CPU fixture 结算证据：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-control-step-cpu-20261113/fixture-settlements.json

SHA256：89d5d3d297c69c43bab33c7112e25560e84fd7962e1c0f0e420bea9b6e9216c6。
配方 SHA256：0615bdd663e5bda4dafdf414004965c5e455525b051fcb63aff456365c109257。

证据包括首窗口 null reward、第二窗口在明确 fixture 权重下的 48.05，以及第 2 个样本已知 joint-limit 故障的提前 terminal。文件明确标记 cpu_fixtures_only_no_native_or_training，不能进入原生成功池。未修改此前原生账本或任何历史证据。

下一步可在独立获批的环境整合中接入真实来源和已批准系数；本包不提出或执行追加训练。
