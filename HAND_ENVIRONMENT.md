# hand 环境适配

项目位置 `/root/hand`，解释器 `/opt/conda/envs/hand/bin/python`。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate hand
cd /root/hand
bash scripts/python.sh -c 'import sys; print(sys.executable)'
```

通过 `scripts/python.sh` 运行项目，它会配置 USD 动态库及模块路径。设置
`DH116_PYTHON` 可显式覆盖解释器，默认已经选择 hand；VS Code 解释器也应选择该路径。

## 本机适配

- Python 3.10.20、PyTorch 2.7.0+cu128、Isaac Sim 4.5.0.0、RSL-RL 3.1.3。
- 修复 hand 环境中 Isaac Lab editable 安装失效的旧 `/root/quad/isacc` 路径，
  当前源码来自 `/root/quad/isaac/IsaacLab/source`，不能移动/删除该源码目录。
- USD 库由解释器 Python 版本动态定位，不再写死 cp311。
- Shell 脚本和 JSON 配置恢复 LF；pose_reference 校验恢复为
  `da3d5a6fdd5e8bb4f5db1911a3659ad3b2b92c2fb99511e4ccbc055d3600512e`。
- 资产 manifest 的四个本机依赖路径迁移到 `/root/hand/assets`，USD 二进制不变。
- Python 3.10 兼容异常附注；Isaac Sim 4.5 缺少求解器排序读回接口时明确记录
  `actual=null, verified=false`，显式请求 contact-last 功能仍拒绝运行。

## 验证与限制

扩展 CPU 回归 214 项通过（环境兼容、tensor_training、tensor_env、tensor_world、
policy_control、coordinated_grasp、lift_grasp_hold、grasp_shaping、compliant_pads）。
初次双环境 probe_06 在手腕运输保护处停止：速度 0.0637655 m/s 大于 0.06 m/s。
后续发现逐轴 0.05 m/s 限速允许对角线合速度达到 0.0866 m/s；已将 policy
模式改为三维合速度限制，保留 0.06 m/s 保护。新合同 policy_owned_grasp_v2
记录 cartesian_vector_norm，拒绝静默复用旧合同检查点。新回归测试先复现失败后通过。

policy_cartesian_speed_probe_01 完成双环境 48 样本及一次真实 PPO 更新，退出码 0。
随后启动 `runs/grasp_lift_policy_256_20261028_v2`，256 环境全新策略/Adam训练，
核验到第6轮 36864样本：策略有限、Adam有13项状态，成功0，主要失败为mimic_error，
失败回合已自动复位。不能宣称已学会抓取或提升。

物理/控制频率 480/60Hz，每控制步8个物理子步；未按现实时间 sleep。
monitor.json 和 TensorBoard timing/* 记录实测 realtime factor，第6轮约0.026倍。
215项环境/控制/训练/物理适配相关CPU回归通过。

## 抓取学习反馈更新

旧 run `grasp_lift_policy_256_20261028_v2` 在第304轮/1867776样本停止，成功0，
未进入提升；检查点和日志原地保留。其近期接触载荷接近0，主要失败为联动误差和
未建立抓取。为提高接触建立的学习信号，新增显式 `--grasp-acquisition-shaping`。

该奖励使用真实指垫位置、物体姿态、有限圆柱表面距离及双手载荷构造潜势，
每控制步按 gamma=0.995 的势差结算；终止下一势置0，回合复位前结算，
rollout边界不清势。未引入脚本动作，未修改抓稳/成功/物理保护。
policy训练启动脚本默认启用；`DH116_GRASP_ACQUISITION=0` 可显式恢复旧奖励配置。
不同奖励合同不能静默续训，旧模型仅保留，当前新run从零开始。

首版距离反馈误用了局部Z轴，审查发现与实际CylinderCfg(axis='Y')不一致，
`grasp_acquisition_256_20261028_v1`在第12轮73728样本停止，产物保留、不继续训练。
已修复为局部Y轴，合同版本 finite_cylinder_grasp_acquisition_v2；回归测试从
真实场景读取轴向，覆盖杆侧、端盖、姿态旋转及终止前结算、复位后下一步和奖励叠加。
223项相关回归通过。`grasp_acquisition_yaxis_smoke_01`完成双环境48样本一次PPO更新。
当前 run `runs/grasp_acquisition_yaxis_256_20261028_v2`为修正后全新256环境训练。
短暂双手载荷满足还不等于稳定抓取；须实际离桌30mm并稳定保持0.5秒才计成功。
`runs/active_tabletop_run.json` 指向当前进程、monitor和检查点。

复现小规模检查（每次用新输出目录，确认没有另一个 Isaac 实例）：

```bash
bash scripts/check_hand_runtime.sh
```

该脚本目标为 2 环境、48 transition 的短训练，所有物理保护保留。
运行输出存放在 `runs/hand_runtime_*`，失败原因以 monitor.json 为准。
环境版本见 requirements-hand.txt；该文件不包含完整的 NVIDIA 系统依赖。

## Lift 抓握保持候选控制

新增 `--policy-lift-hold`：仅在实测 close-to-lift 准入后捕获已施加的手指、
手掌相对位姿和刚度；lift 保持抓姿并作有限载荷补偿。失接触、明显滑移或
跟踪偏差时暂停上升，取消尚未执行的手掌高度目标。只有双手策略明确请求
上升时才共同升高，不自动执行提升。压力释放及原物理保护优先。

新增30维保持状态，当前训练观测186维；旧156维检查点不能直接续训。
policy 启动脚本默认启用，`DH116_POLICY_LIFT_HOLD=0` 可关闭。
73项保持/控制/集成回归通过；`runs/policy_lift_hold_smoke_20261028_v1`
在真实Isaac完成2环境48样本一次PPO更新，模型及优化器有限。
该短测没有有效lift，不证明抓取或稳定提升成功。

旧 `grasp_acquisition_yaxis_256_20261028_v2` 在156轮958464样本中断，
检查点保留，成功0且未进入lift。新256环境训练从零开始：
`runs/grasp_lift_hold_256_20261028_v1`，活动指针见 `runs/active_tabletop_run.json`。
本次没有因参考DexterousHands而修改奖励或放宽成功条件。

## 实测提升进展和低滑移保持奖励

用户随后确认增加奖励。新增 `--stable-lift-reward`，policy 启动脚本默认启用，
可用 `DH116_STABLE_LIFT_REWARD=0` 关闭。旧 lift 高度差项替换为每回合最高
实测高度进展（最高30mm、总计5分）；增加离桌至少3mm后的低滑移保持项，
4分/秒乘实测高度/30mm，加权累计.5秒封顶2分。失去抓握、速度>=.03m/s、
滑移>.002m、手对物体合力向下或当前子步失败时不发放、不推进奖励额度。
额度不会因抬落、失接触或rollout边界而重置；只有对应回合reset才清零。
3/10mm里程碑、最终30mm连续.5秒成功判定及物理保护保留。

新增两维奖励状态，当前观测188维，新旧奖励合同不能静默续训。
177项相关CPU回归通过。实际Isaac短测 `stable_lift_reward_smoke_20261028_v1`
完成2环境48样本一次PPO更新，合同、奖励项、模型和优化器有限检查通过。
`stable_lift_reward_budget_resume_20261028_v1` 验证已完成1轮的checkpoint恢复后，
预算仍为1轮时不额外rollout或更新，iteration仍1、timesteps仍48。
两项实测均未成功抓取提升，不能据此声称抓取稳定问题已解决。

旧256环境训练在29轮178176样本停止并保留checkpoint。
当前指定启动：4096环境、24步rollout、2048次PPO更新，从新模型训练。
总预算201326592 transitions；显式 `--max-iterations 2048` 优先于回合/timestep预算。
目录 `runs/grasp_lift_reward_4096_2048_20261028_v1`，PID见活动指针。
TensorBoard绑定远端127.0.0.1:6006，通过专用SSH映射至本地127.0.0.1:16006。
训练进程、TensorBoard及隧道独立运行；浏览器可直接打开本地地址。
实际4096环境已完成首两轮PPO更新（196608样本），monitor明确iteration_budget=2048；
188维合同、模型及优化器有限检查通过，TensorBoard可读到reward/lift和
reward/stable_lift_hold。首两轮尚无lift或成功，这两项奖励均为0。

## PPO奖励接入审计

确认当前PPO路径是 Task.consume 累加奖励 -> Task.end_step -> TensorEnv.step
叠加grasp_acquisition -> algorithm.process_env_step(obs,reward,done,{}) ->
compute_returns -> PPO.update。TensorBoard分项与PPO使用同一个reward来源。
无需重复叠加奖励，避免双计。

`scripts/audit_ppo_rewards.py runs/grasp_lift_reward_4096_2048_20261028_v1`
核对37轮已保存记录，各奖励项齐全、分项之和与总奖励最大误差3.43e-7。
本次33项奖励和集成回归通过。高度/保持尚为0，因为还未实际进入lift。

该run在完成37轮3637248样本后发生独立物理保护停止：环境2022接触力
30.03195N且联动误差0.0213375rad同时超限；不是奖励未接入。
检查点保留第37轮，2048轮预算尚未完成。未为续跑放宽物理保护。

## 保护停止根因修复与关节越限隔离复位

故障测量32帧回放证实：旧恢复条件只允许force>30且mimic<=.02，或
mimic>.02且force<=30，组合轻度超限时两个候选均false，导致全局abort。
已将两者统一放在同时force<45、mimic<.03的原有限恢复边界内分类。
严格健康边界仍30N/.02rad；超限仍失败、-25分，立即只复位失败环境。
硬force>=45、mimic>=.03、非有限值和独立几何故障仍中止。

用户进一步明确关节限位直接reset：有限关节越限（包括完整关节数组和
已有joint_limit标签）结束对应回合，保留原失败原因和惩罚，不停止其他环境。
不修改关节物理限位、不夹平测量；reset后立即及下一物理子步仍需通过检查。
新增recovery/combined_failure_resets与recovery/joint_limit_resets，计数保存到checkpoint。
两项新合同只允许精确记录的checkpoint迁移，模型及Adam状态保留。

当前172项相关CPU回归通过；实际2环境48样本一次PPO更新通过，
目录runs/combined_guard_joint_reset_smoke_20261029_v1。
故障回放是记录测量经过分类逻辑的回放，不是完整PhysX状态重放。
新续训目录runs/grasp_lift_reward_4096_2048_20261029_resume2，保留原错误日志和trace。
TensorBoard合并显示before与resumed两个run；新增reward/total直接记录PPO平均总奖励，
原run已从iterations.jsonl补齐37点，不重复计入PPO reward。

resume1在Isaac原生初始化、未加载checkpoint前出现malloc invalid size，
进程已退出；主机内存/显存/磁盘充足，原生日志保留，根因尚未确认。
干净重启的resume2已恢复并完成39/2048轮，checkpoint_migration记录两项精确
合同新增且policy_and_optimizer_retained=true。实际模型/优化器有限检查通过。
组合故障和新joint_limit复位计数当前仍0：相关行为有回归和故障测量回放验证，
不应声称本次续训中已实际遇到并恢复该组合。未来真正硬故障仍保留停止保护。

## 每手2到12N目标力与中轴以下接触修正

后续检查发现旧policy模式的last2动作是stiffness，12N仅保护ceiling，
close准入仍1N且contact_target_z_m=None，未落实用户要求。
旧run已在506轮49741824样本保存并停止，未混入新模型。
截至停止，组合故障隔离复位14次、关节限位隔离复位1次，未因这两类事件停机。

新增显式--force-region-control（默认policy launcher启用；可用
DH116_FORCE_REGION_CONTROL=0关闭），20维动作改为12关节+6手腕+2目标夹持力，
每手目标2–12N，刚度固定10Nm/rad。新合同拒绝静默恢复旧stiffness动作模型。
目标与实测分开，闭合力反馈仅在有效轴下接触后正向补偿，过力可释放；
夹持目标不是世界Z方向承载力，未夹到物体时实测可以接近0，绝不夹平传感器值。
实测准入要求每手2–12N、与目标误差<=max(.5N,目标10%)，双手承力报告均低于中轴。
接触位置按实测物体姿态的局部Y轴计算径向Z，支持杆倾斜。

用户指定接触奖励系数：轴上0.5、轴下2、无有效接触0。
该系数接入contact奖励及acquisition载荷势项；向下力惩罚仍用未加权真实负Fz。
force_tracking和contact_region分项进入同一PPO总reward，reward/total继续记录。

按用户提供的抓取力稳定方案，close完成捕获当时的目标F_ref，不用单帧实测替代；
之后包括滑移恢复均不改F_ref。滤波tau=.03s，PI Kp=.01rad/N、Ki=.02rad/(N*s)，
积分限幅2N*s，位置偏置限幅.08rad、速率.03rad/s；无有效接触/保护/饱和时
冻结积分或按最终施加目标回算。力偏差超容差时暂停手腕上升，PI继续有效补偿。
这是候选参数，不是已证明的实测恒力控制性能。
新观测198维，含接触几何和PI历史；原观测198以前的checkpoint不直接续训。

当前300项CPU回归通过；真实Isaac 8环境最终短测和从0开始4096环境训练
状态以活动指针和对应monitor为准。TensorBoard新增force_control组，分开显示
请求目标、实测载荷、误差、有效轴下接触比例与已捕获hold环境数。

最终8环境短测 `runs/force_region_constant_hold_8env_smoke_v2` 完成192样本1次PPO更新，
198维合同及模型/优化器有限检查通过；实测请求范围2–5.683N（属于2–12N），
未建立有效hold，不能声称物理恒力或成功lift验收完成。
用户明确改完从0开始，当前新run为
`runs/force_region_constant_hold_4096_2048_fresh_v1`，显式DH116_RESUME为空，
4096环境、24步rollout、2048次PPO更新，TensorBoard只显示该新run。
原506轮检查点和历史日志保留，新旧动作语义不混用。
新4096环境训练已完成前两轮196608样本；training_start记录resume_source=null、
initial_iteration=0、initial_timesteps=0、optimizer_state_entries=0，确认为从零开始。
198维模型及优化器有限检查通过；两轮分项奖励与PPO总奖励最大误差2.62e-7。
TensorBoard已打开并聚焦reward/total及左右手目标/实测力。
当前仍无有效lift，实际恒力保持/抓取提升成功尚未验收，不将代码接入等同物理成功。

## 教师候选准备与手动Isaac

用户要求先教师探索成功案例，PPO已在1023轮100564992样本保存并停止。
教师是实测反馈参数搜索，不是已验证成功的预训练模型。已实现动作接口教师模块，
24个源码FK候选保存在本地teacher_geometry_candidates.json；尚未进行实际成功搜索。
随后用户要求直接手动控制Isaac，自动教师不与手动会话并行运行。

新增--manual-control与scripts/manual_dh116_control.sh，单环境GUI通过既有XLaunch
display10显示本地；左右手目标法向力2–12N、各6主动关节角度、掌部XYZ、
公共提升高度0–50mm及仿真速度0–10mm/s均可调，初始暂停。
Apply Force/Pose允许用户显式修改锁定参考/抓姿；实际命令仍经过slew及物理保护。
手动模式单独关闭训练4秒close/15秒回合期限，达到提升标准仅显示而不立即reset；
不作为合格教师数据。故障仍暂停、reset并清除旧提升请求。
manual Snapshots只保存未验证候选；详见MANUAL_ISAAC.md。

相关93项首轮测试通过；修复held XYZ/下降与旧提升目标残留后85项相关回归通过。
修正版GUI runs/manual_dh116_control_v2 已实际打开，能看到右侧控制面板，
已观察到用户Apply Force回调更新为5N；当前状态见manual_status.json。
用户随后要求关闭且不保留本次手动会话产物：Isaac已结束，manual_v1/v2会话目录、
其中的日志/参数快照及本地对应转发日志均已删除；上文manual_status路径已不再存在。
PPO保持停止，未自动启动教师搜索。

## 轴下初始化与分阶段作用力奖励（2026-10-29）

默认初始化现为configs/dh116_training/grasp_initialization_below_axis_v1.json。
真实8环境复位测得四个指垫参考点低于杆中轴0.893–1.107mm，最低桌面间隙1.479mm。
启动检查区分指垫位置与实际接触；初始没有接触，不将几何位置当成成功抓取。

新增grasp_upward_force（抓取阶段、法向向上力、双手合计上限2分/秒）和
lift_upward_force（lift阶段、法向加摩擦向上力、双手合计上限5分/秒），
按半物重归一化并饱和，只有轴下有效接触计分；向下合力不获正奖励。
两项进入PPO总奖励与TensorBoard reward分项。每手2–12N目标仍为法向夹持力，
close锁定目标和原有安全保护/实际成功条件不变。

278项相关CPU测试通过。8环境192 transition/1次PPO更新真实验证完成，
模型和优化器有限、奖励加总最大误差1.38e-7；抓取向上力奖励出现非零值，
尚未进入lift。随后8个教师候选在241控制步内完成，0成功/8失败，未获得合格示范。
验证记录分别在runs/below_axis_phase_force_8env_smoke_v1和
runs/below_axis_phase_force_teacher8_v1。4096环境PPO仍停止于旧运行1023轮；
新奖励与初始化不静默恢复旧checkpoint。GUI保持关闭。

## 初始指垫深度改为中轴以下2mm（2026-10-29）

用户撤回3–4mm深度要求，最终采用2mm。3–4mm姿态搜索未用于训练。
默认初始化改为grasp_initialization_below_axis_2mm_v1.json，独立audit保留。
只调整root平移，原关节姿态、root旋转、物体/桌面和物理参数保留。
静态构建和重复verify通过，初始17碰撞网格/手均不穿透杆，双手AABB分离。
保留的旧closed command仍存在杆重叠，不能当作已合格抓取姿态。
8环境真实复位的四点径向Z为-2.00534/-1.99464/-2.10658/-1.89343mm，
最小桌面间隙0.479112mm；初始接触数0。1次PPO更新192样本完成。
27项初始检查、分阶段奖励、迭代预算及大网格参考精度测试通过。
原10240训练第18轮checkpoint保留，新姿态10240环境2048迭代从零重新启动，
run=runs/phase_force_below_axis_10240_2048_20261029_113859。
2mm是初始指垫参考点的平均深度，运动中实际接触仍由传感器判断，
不宣称整个闭合过程中接触点被硬性固定在2mm。

## 无阶段自主抓取与抬升

按用户最新要求，训练默认启用--free-grasp-lift，取消close/hold/lift状态流程、
准入条件和close4秒期限；不创建PolicyLiftHold，不锁定抓姿或因接触不足暂停上升。
PPO全程控制关节、掌部XYZ和2–12N法向力目标；每手实际12N卸载保护、
关节/耦合/桌面/非有限数等物理保护，以及15秒总回合期限保留。
旧phase槽仅作为固定内部兼容值3，无转换；4维阶段onehot移除，观测160维。
TensorBoard不写阶段占比和close完成事件；新增exploration双手接触比例/实际提升高度。
向上法向力2分/秒和向上法向+摩擦支撑5分/秒两项全程评估，原接触/轴下/负合力/
超12N计分规则保留；持续向下法向力惩罚-.5*max(-Fz,0)*dt保持。
接触区域权重上.5下2、目标力跟踪、接近/双手抓取势函数、向上力改善势函数、
实际高度进展、3/10mm里程碑、稳定保持和成功奖励保留；动作/超力/滑移/
接触区域/超速/耗时/失败惩罚保留。stage/contact_dwell/grasp_ready/close_downward为0，
不再靠完成阶段得分。实际抬升奖励及成功检查采用真实双手接触，不再用旧close
力跟踪和轴下准入作为先决门槛；实际30mm、速度/滑移/非负支撑和.5秒成功标准保留。
285项CPU测试通过；真实8环境192样本1次PPO更新完成，160维模型及优化器有限，
分项奖励总和误差9.49e-9，完整回合向上/向下计分均开启。
QA run=runs/free_grasp_lift_8_1_20261029_120015。
旧分阶段checkpoint保留，新模式10240环境2048迭代从零启动：
runs/free_grasp_lift_10240_2048_20261029_120148。

用户随后将环境数减为2048。原10240运行保存于第2轮491520样本，原checkpoint保留；
resize_ppo_checkpoint.py生成独立resume_2048_envs.pt，只修改合同环境数和前三项
逐环境计数器长度，策略/优化器/迭代/样本/RNG保留，采样缓冲区和物理场景重建。
2项缩容回归测试通过，迁移来源SHA与统计写入同名.resize.json。
新运行runs/free_grasp_lift_2048_2048_20261029_120908，仍以总计2048轮为预算，
从第2轮继续，无阶段模式/2mm初始深度/奖励和物理保护保留。

## 物体跌落改为单环境reset

原运行在84轮停止：环境1747的bottom=.28938158低于.29，Task判定object_dropped，
独立physical_guard却将同一掉落视作全局致命，阻断terminal reset。
按用户明确要求新增isolated_object_drop_reset_v1：live环境bottom<.29且其他
有限数/桌面/可恢复力和耦合等检查通过，即保留失败标签及-25惩罚，物理子步内
reset对应环境。其他环境继续，reset即时和下一物理步严格检查保留。
掉落可与原允许reset的超力、耦合、有限关节越界一起处理；非有限数、严重
力/耦合、手部穿桌等不因掉落被豁免。新增recovery/object_drop_resets计数并入checkpoint。
298项回归通过；真实8环境主动掉落验证仅环境0复位，底部回到.299999998m，
保留-25惩罚，完成192样本1轮更新；模型/优化器有限，奖励加总误差2.50e-7。
QA runs/object_drop_reset_8env_smoke_v2，drop_reset_validation.json保留。
新合同只允许增加这一reset语义，策略/优化器/160维观测和其余训练参数不变。
从原84轮checkpoint继续2048环境、总预算2048轮：
runs/free_grasp_lift_2048_2048_20261029_125853。

## 手动提升诊断

PPO已按用户要求停止于259轮，未恢复训练。当前工作仅为单环境手动Isaac。
Lift按钮修正为同时恢复物理运行；默认目标10mm。新增四个拇指/食指的原始接触点、
负载、中轴相对高度、接触侵入量、真实手腕/物体高度和reset前快照。
80项相关单元回归通过，真实GUI已运行验证这些遥测。
联动200/400s^-1及刚性对照均未完成稳定物体提升，当前恢复200s^-1。
刚性对照腕部确实上升约1.33mm，物体仍在桌上，出现轴上接触/深侵入/超力reset；
不能把手腕指令、实际手腕运动或四点某一时刻低于中轴当成成功抬升。
详见MANUAL_ISAAC.md与manual-lift400-failure.json、manual-lift-rigid-failure.json。

按用户追加要求，手动快速力控改为拇指/食指各自微调，保留每手总力目标，
实测大力指不超过小力指的1.5倍；不强制均分。弱指微调到限时降低强指目标。
左右拇指需分别连续0.3仿真秒稳定承力（>=0.5N，峰峰值<=max(.1N,5%均值)）。
自动提升还要求两手力差达标和四处实测接触都在轴下；0N平台不算稳定。
PPO和默认训练控制路径保持停止/不变；149项相关回归通过。
