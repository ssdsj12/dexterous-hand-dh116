# DH116 独立工作目录

## 2026-10-22 GitHub阶段快照：`10-22`

上传目标：[ssdsj12/dexterous-hand-dh116，10-22分支](https://github.com/ssdsj12/dexterous-hand-dh116/tree/10-22)。本次从当前`10-21`工作状态新建分支，保留既有提交历史，不覆盖远程`10-21`。

- 本次包含：手部12D并行MPC、独立Panda搬运接口，CUDA Cosserat杆的剪切/拉伸/弯曲/扭转及接触反作用力，完整v2专家采集与教师/学生蒸馏接口，以及10ms执行窗口预载纠偏。MPC预测仍为线性化指腹模型；控制、几何检查等CPU环节没有被描述成全GPU或实时运行。
- 实际物理阶段：V23闭指与完整1秒夹持通过，lift失败、整线未离桌；170个完整区间仅作诊断，尚无合格完整成功专家轨迹或由其训练并验证的教师/学生。此次上传没有新开Isaac或修改抓取姿态。
- 本次新鲜验证：下列专项**99 passed in5.19s**；全量`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q --tb=no`为**874 passed、71 failed in29.03s**，不是全仓库通过。失败涉及接触/源几何、序列与旧O6等测试；本次仅上传现状，未扩展修复。`git diff --check`通过。
- 仍待完成：允许并筛选合适的双掌/整臂小幅位姿、实际稳定lift与整线离桌、成功专家闭环采集、正式教师/学生蒸馏及各自Isaac验证。当前窗口纠偏已经生效，不因此重跑approach。

本次专项命令：

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest \
  tests/test_gpu_cosserat_physics.py tests/test_gpu_cosserat_runner.py \
  tests/test_dh116_contact_collection.py tests/test_dh116_contact_imitation.py \
  tests/test_dh116_contact_replay.py tests/test_dh116_grip_load_reference.py \
  tests/test_dh116_complete_contact_policy.py tests/test_dh116_close_initialization.py \
  tests/test_dh116_contact_mpc.py -q --tb=short
```

上传保留源码、测试、必要接触网格/配置、既有USD和厂商资源；不上传`runs/`、`archive/`、日志、缓存、密钥和本地`1332`，不删除其原件。下文run报告/状态备份均为本地证据，不随本次分支上传。部分配置保留本机源资产绝对路径及SHA，异机运行需重新定位资源并校验，不宣称任意路径克隆即可复现实验。此处为最新阶段摘要，下方按时间保留开发记录。

## 2026-10-22 当前：close与1秒hold通过，lift未完成；局部四点继续筛选

补充受力证据：对V23实际12025/14151/16921杆状态及记录的接触网格末端位姿做CUDA瞬时力分解，共3帧×3种速度工况×6通道。原模型/网格SHA匹配，54项与原CPU接触函数在相同距离/法线输入下的节点力、杆力矩及机器人反力/力矩一致，最大力误差2.23e-16N、力矩误差3.47e-18Nm。这仅验证该接触力律与分配，不是独立网格距离、全积分轨迹或成功抓取验收。

在lift入口保存几何中，线缆重量0.98621N，实际边界速度对应法向竖直力−0.53699N、摩擦+0.000427N；与原积分帧合力相差约0.000019N。标为反事实的“杆暂时静止、双手仅以5mm/s向上移动”工况，摩擦只有+0.097998N，法向仍−0.53502N，手对杆合力仍−0.43702N。因此该姿态下单纯增加夹力并不能保证lift。CPU/GPU当前都使用`min(10*滑动速度, 0.5*法向载荷)`的正则化摩擦，并非GPU遗漏摩擦；未将它替换成静摩擦或修改参数。只读报告在V23的`contact_support_audit{,_cpu_parity}_v1.json`，脚本两次实际exit0，未执行Isaac/杆时间推进或生成新训练标签。

V23已实际结束，不再LIVE：第16922步在物理步后被右拇指非指腹`finger13_link`源几何检查停止，保守间距0.999886mm低于原1mm；6个指腹区域均通过。最后完整有效16921步仍四点接触有效、材料滑移1.413684mm<2mm，但线缆站点未抬升。**close=PASS、hold=PASS、lift=FAIL，没有成功专家或正式蒸馏。** 实际170个完整区间（21close/100hold/49lift）时序与状态链通过，末区间如实标记失败，正式加载器拒绝该源。

本轮不重开相同配置，先做只读几何复核和已批准的Panda固定四点筛选。用失败步实际全18关节状态重建USD，独立支持平面证书给出约1.148mm（保留原78.700µm运动包络）；这支持棱角附近面法线界偏保守的解释，但不是逐位实测PhysX位姿或实际通过证书，运行算法和1mm要求未改。

在V23最后close2024状态上筛选，3mm按相对最初V20四点的**累计**位移计算。四个对齐/下移目标试验中3个通过完整源几何与原6区域，最大累计点移<1.6mm、保持原10µm加载深度限制，但接触法线仍偏向下。进一步每手82个保持加载深度的拇指姿态扫描，选出的对齐候选完整静态检查通过，仍未得到向上承托。不能把几何通过当作真实lift成功，也没有据此生成新运行初态。

额外只读对照将双掌整体下移2mm：四点累计移动<1.34mm，但左`finger13_link`对桌面保守余量0.966722mm失败，未采用。掌心位姿变化属于超出当前Panda固定方案的下一设计选择；尚未改控制器、USD、接触/摩擦物理或成功阈值，没有新Isaac实例。下一步需要选择是否允许小幅掌心/整臂位姿调整，再筛选承托姿态和验证实际闭指、夹持及lift。

实际执行的诊断均exit0（本地run不上传）：

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh runs/dh116_contact_mpc_close_gpu_20261022_v23_tgs1/source_clearance_witness_probe_v1.py
DH116_NUM_THREADS=1 bash scripts/python.sh runs/dh116_local_pinch_alignment_20261022_v3/bounded_pose_probe.py
DH116_NUM_THREADS=1 bash scripts/python.sh runs/dh116_local_pinch_alignment_20261022_v3/bounded_pose_probe.py --support-scan
DH116_NUM_THREADS=1 bash scripts/python.sh runs/dh116_local_pinch_alignment_20261022_v3/bounded_pose_probe.py --palm-down-mm 2
```

报告以独占方式写入；复查已有`*.json`，不要原命令覆盖。`support_scan_v2.json`修正了初版根区间筛选会漏掉同号端点内两根的问题，初版不是可行性结论。编译与`git diff --check`通过；未更改生产源码，因此未重复不变的138项历史专项，也不宣称全量测试全绿。当前说明优先于下方历史运行状态。

## 2026-10-22 V21保持通过、lift失败；隔离复现低速关节位置冻结

原V21在第14700步/time1.4701s通过原100ms稳定夹持，第24700步/time2.4701s完成原1秒保持。它随后在27063步/time2.7064s（lift0.2363s）失败：右拇指分离速度1.2154mm/s超过原1mm/s条件，仍有0.9544N载荷、gap约−0.3070mm；材料滑移0.1852mm未超过2mm，站点高度变化仅约+0.0019/−0.0018mm。**close、hold通过，lift失败，整线未离桌**。PID157969/session21797已结束；不重开相同失败配置。Panda搬运、hand_mpc_hold保持12D手部目标的分工不变。

这项物理进展来自旧版已加载代码，不能归因于本轮磁盘上的窗口修复。原manifest没有`preload_feedback`，原数据没有`preload_window`。终态完整v2回放核验272区间（148close/100hold/24lift），时序/token/状态链通过，末6.3ms区间如实标记失败；仍为诊断数据，正式加载器明确拒绝，没有成功蒸馏。进入hold前最后有效close14699完整物理状态及旧源码备份保留，不带阶段成功资格。

实际lift前0.2126s中，Panda目标在变化，左右各4个实际关节角却逐位未动，掌心反而下移0.2654/0.0957mm。独立、无重力/无线缆的GPU Isaac单关节对照复现：100µs步长、0.005rad/s、1000步、16次TGS位置迭代时实际位移0；仅改为1次后位移0.000476837rad，期望0.0005rad。说明至少存在低速位置积分的数值问题；**不证明1次迭代已满足整机约束或抓取成功**。局部float32算术与冻结轴吻合；TGS按位置迭代细分时间步的机制见[NVIDIA说明](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html)。报告及源脚本仅存本地V21目录的`tgs_joint_integration_{16,1}.json`和`probe_tgs_joint_integration.py`；16次原脚本及SHA另行保留。原诊断退出清理等待后已终止退出，1次对照用项目现有快速退出正常结束；两个实例从未同时运行。

本轮按已批准“实测欠载即恢复调节”的闭环约定修复采样缺口：每100µs物理帧保留最低载荷/最大间隙/已知接触标记，在下一次既有10ms决策消费，避免只看决策帧而漏掉区间内欠载。原90%门槛、接触/滑移/保持/抬升判据、力矩和物理参数不变；12D手部和独立Panda控制不变。新增统计及来源区间保存在决策`goal.preload_window`，新运行标记`completed_physics_window_v1`，原网络/检查点维度不变。旧调用没有窗口时保持原行为。

测试先复现8个预期失败，再focused27 passed in1.47s；相关9文件138 passed in7.52s，编译/diff检查通过。这些历史专项不冒充新整机验证或全仓库全绿。原16次迭代的V22窗口接线基线已自然完成500步：原逐帧检查通过，5个完整close区间和真实100帧窗口来源通过；限长50ms未完成技能，明确标为失败/诊断而非成功轨迹。

唯一V23从同一14699实际close状态继续，位置迭代改1、恢复200000步完整预算。两份实际manifest仅这两项不同。装配场景前400帧对照：两组原检查均通过，V22左右Panda七轴全部零位置变化，V23各七轴均恢复非零变化；报告`runs/dh116_contact_mpc_close_gpu_20261022_v23_tgs1/first400_integration_comparison.json`仅证明局部积分响应，不是lift验收。V23真实decision300两侧当前拇指载荷已>.9N，但前窗口最低.893/.893N，仍正确保持未锁；decision400左侧当前.911N、窗口最低.899N，也未误锁定。新窗口接线已实际执行，不再只是离线验证。

V23实际2025步/time0.2026s通过原稳定夹持，在12025步/time1.2026s完成原1秒保持并进入lift；最新12348步有效lift，elapsed0.0323s，四点接触有效、材料滑移0.2859mm。新完整回放122区间（21close/100hold/1lift）时间/token/状态链通过，2.6ms交接短区间原样保留。**进入lift不等于抬升成功，当前wire站点高度变化仍仅约0.0012/0.0016mm，整线未离桌，不正式蒸馏**。可用完整备份仍为原2024步close状态，外部SHA`fb520d675454f8ea4239b669e04a48d3edb5878de55f7d25bc662e9d7516c365`，不导入任何阶段资格，旧备份保留。额外hold12024备份被原close-only工具拒绝、未创建文件；没有绕过校验或停止当前实例，原轨迹行仍保留。

同一PID167430/session76710继续，日志`/tmp/dh116-contact-mpc-cuda-v23-tgs1.log`；恢复时先核对存活状态，不因慢而重开。原USD、GPU杆、100µs外步、16接触采样、力矩/增益和全部成功门槛不变。只有完整真实成功后才正式训练和分别验证教师/学生。数值库启动前设为1、CPU固定core0；Isaac随后覆盖的OpenBLAS环境记录12不等于已测量的实际库线程数，未据此中断主实例。无提交/上传。执行计划见`docs/superpowers/plans/2026-10-22-dh116-preload-window-feedback.md`及完整专家计划末节。

只读lift对照已保存到V23的`lift_tracking_probe_v1.json`及同名`.py`：lift0.2126s左右掌心实际上移0.0994/0.1321mm，但wire仍贴桌；手对线缆的竖直合力约−0.527N，仍向下。这是当前跟踪/接触方向的实际证据，不是成功轨迹或最终唯一根因。最新14853步/lift0.2828s四点仍有效、滑移0.4723mm，继续原实例等待真实结果；未因此改抓力、物理参数或成功门槛。

另用14701个真实物理帧、148次旧close决策做了离线掩码对照：旧逻辑与原记录全部一致，新窗口反馈在32次决策（左31/右1）会补识别已发生的区间欠载/接触缺失，使用的帧均早于决策。报告为本地run内`preload_feedback_window_replay_v1.json`；它不生成新动作或物理结果，不是新抓取轨迹或成功训练源。

## 2026-10-21 四点局部对齐：固定 Panda 候选已筛过，V21实际验证启动

按用户选择，仅优化现有站位附近的真实拇指/食指表面位置，不改标定区域或启用直接平移对向目标。Panda七关节和双掌姿态完全不变，非夹持手指停放。实际USD网格上的轴向错位：左4.163→2.588mm、右1.755→1.029mm；四点最大移动1.672mm，mesh有符号距离变化最大2.871µm。完整50体源几何（最小保守余量1.215mm）及6个原pad区域静态通过。**这是几何候选，不是物理抓取成功**；仅离线比较的掌心微调提案未采用。

新状态在`runs/dh116_local_pinch_alignment_20261021_v2/`，保留原V20/3103的完整杆动态、关节速度及主动关节初始驱动偏差，Panda状态/目标不变，不带任何阶段成功资格。继承载荷/预测偏差仅作控制参考，不能充当重构后实测载荷。现有恢复/IK专项：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_close_initialization.py tests/test_dh116_arm_ik.py -q` →18 passed in1.01s；没有修改运行源码、原USD或物理阈值。

已启动唯一V21（PID157969，工具session21797，日志`/tmp/dh116-contact-mpc-cuda-v21.log`），从此新close状态验证，不重放approach；仍为完整GPU柔性杆、12D手部MPC、Panda独立对称lift、原载荷/滑移/持续时间/抬升门槛。运行状态及阶段须实时重查；启动不等于close/hold/lift通过。正式成功专家、教师/学生训练及实际验证仍未完成。执行计划：`docs/superpowers/plans/2026-10-21-dh116-local-pinch-alignment.md`。无提交/上传。

首帧实际源几何检查通过、四夹持点已有载荷。已核对至11683步（本轮物理时间1.1684s）仍有效close，四点接触有效，载荷约.898/1.038/.953/1.009N；双手preload，左拇指仍在原.9N门槛附近波动，尚无hold/lift成功。物理时间超过1秒不等于完成原1秒保持，因为尚未进入hold。逐帧诊断发现最长基本合格区间98.4ms，左拇指随后低至约.8999N重置原100ms计时；10ms决策时刻可能仍读到>.9N，指令保持看不到该时刻之前的短降。缺口已定位，但不将它视为唯一原因：后续欠载被采到时已有实际解除保持。窗口纠偏方案待用户选择，未改运行源码或重载当前实例。

实际已核验101个完整v2区间（全部close，累计1.01s，末outcome10100），12D动作、时间/token/状态链通过、不含Panda关节状态。元数据仍为`partial_action_diagnostic`，源未终态、未grasp/lift成功或整线离桌；正式加载器正确拒绝。另已保存1755及较新的11517步实际完整close物理状态，供必要时获批重载续接，均不带阶段资格，旧文件未覆盖；当前V21继续同一PID。未正式训练，不把静态改进、接触出现或单帧载荷达标当成hold/lift成功。

当前close性能采样：每个100µs物理步约耗时0.35s，GPU杆与接触约0.10s，源几何前/后检查各约0.10s；`state_and_sensors`与后检查重叠，不能全部相加。只读静态微基准（三组轮换、每组十次）测得CPU耗时：源几何24.8ms、已有证书缓存3.7ms、六指腹区域64.6ms，静态检查均通过。输入为1755保存状态的18D关节USD重建位姿，不是实际PhysX位姿捕获；与单核V21并行竞争，**不能据此宣称整机提速**。缓存不能消除指腹检查成本，因此未单独为其重启当前实例。报告在该run的`source_guard_cost_probe.json`，原运行/物理参数不变。

## 2026-10-21 V20已结束：close与1秒hold通过，lift失败

V20在3104步（物理时间0.3105s）通过原100ms接触预载验稳，在13104步（1.3105s）完成原1秒保持并进入lift。它随后在15805步（lift约0.2701s）触发停止：右拇指仍有0.9295N载荷、间隙约-0.2973mm，但分离速度1.0943mm/s超过原1mm/s有效接触条件；材料滑移仅0.2437mm，未超过2mm。终态明确为`close=PASS / hold=PASS / lift=FAIL`，**没有抬升成功或整线离桌**。PID145546/session15643已结束，当前没有Isaac运行；不再沿用下方历史LIVE说明。

实际控制所有权为Panda搬运、DH116保持，手部动作始终12D。终态完整v2回放读入32个close、100个hold、28个lift，共160个真实区间；时间/token/状态链通过，末段明确`passed=false / terminal=true / grasp_lift_passed=false`，真实时长0.1ms；close→hold交接0.5ms也如实保留。正式入口拒绝此失败源，未正式蒸馏。

只读跟踪核对发现：lift起点的关节目标FK掌心比实测掌心低约0.98/1.19mm；0.2322s时目标已上移约0.660mm，但实测掌心反而下降约0.297/0.207mm。报告关节速度与位置差分也不一致，PhysX传感器未报告额外>1mN碰撞载荷。它支持优先调查搬运跟踪，不足以断定唯一根因。当前建议给独立Panda搬运加入实测掌心位姿闭环，保持原手部12D与物理/接触/成功判据；方案待用户选择，尚未改运行代码或开新实例。

进一步只读核对：实际关节角FK与实测掌心在进入/失败时均相差<0.0004mm，未发现该坐标比较错误；当前配置未显式启用额外stabilization（安装版默认False），不据此猜测改求解开关。仍待方案确认后实施，不将这些诊断当修复或抓取成功。

用户确认若仍不稳，先在现有站位附近局部对齐，每个目标不超过3mm，原物理判据不变。只读筛选表明：直接约2.22/1.44mm对向目标会使较后预测姿态的左食指局部区域/压入失败；缩小到约0.111/0.072mm可通过16个预测姿态的静态几何，但实际mesh最近点的左右轴向错位由对照4.164/1.755mm变成4.134/1.806mm，**不能称为双手改善，也未施加到运行器**。若后续确需换点，应联动求解真实指腹姿态，不能只移动目标坐标或降低接触区域门槛。方案见`docs/superpowers/specs/2026-10-21-dh116-local-pinch-alignment-design.md`。

已保存有效close3103完整状态供必要时对照：`runs/dh116_contact_mpc_close_gpu_20261021_v20/pre_hold_initialization_step3103.json`，不携带阶段资格。所有筛选报告留在该run的`local_pinch_probe_*.jsonl`，只是预测/静态诊断。此次未改运行代码、不重跑未变化的195项专项，无提交/上传。

## 2026-10-21 TouchFirst 同步实际 MPC 载荷目标（当前 V20）

按用户确认，MPC 模式下 TouchFirst 从第一帧同步实际控制目标（每手拇指/食指1N、中指停放），不再等待未参与执行的0.05→1N名义坡道。触碰50ms、对指50ms、实际载荷/接触100ms、保持1秒及原滑移/抬升/整线离桌判据不变；普通非MPC调用仍保留原坡道。

V19保存有效1752步的完整物理状态后正常停止，单实例V20从该close状态重载，不回放approach，不继承阶段资格。实际已记录 `preload_target_source=controller` 和1N目标；1062步双手进入preload、目标直接就绪，不再等待旧坡道。最近1982步有效close、四点严格接触有效，载荷约.889/1.009/.923/1.028N；左拇指低于原0.9N要求，已释放左手指令保持并恢复调节，验稳计时须重新累计。已写19个完整v2控制区间。**尚未稳定夹持、hold或lift成功，未正式蒸馏。** 后续继续同一实例，以实测结果验收。

受影响专项195 passed in14.26s，启动后重跑同一套195 passed in11.32s，编译/diff检查通过；不是全量测试通过。新测试确认同步目标不会替代实际接触/载荷，也不会跳过接触和1秒保持时长。未提交/上传。

## 2026-10-21 统一到原有载荷要求（当前 V19）

已去掉预载暂停中的额外95%条件：现在达到原抓握要求的90%目标载荷且实际接触已知时，即保持该手的当前指令；低于原要求或接触丢失时恢复调节。独立物理成功判据没有降低。此前约.93N已达到原要求，但额外95%条件让MPC仍持续追踪，妨碍验稳。

V18保存有效500步后正常停止，单实例V19从保存的物理状态继续，仍不回放approach。最近607步有效close，双手保持已触发，四点连续有效57.3ms，两手均由touch进入oppose，实测载荷约.912/1.017/.911/1.027N；完整v2时序文件已写6个控制区间。**仍未通过最终稳定抓握、hold或lift，也未开始正式蒸馏。** 保留1ms位姿速度窗口、100µs物理步长、完整GPU杆和所有原物理门槛。

最终受影响专项184 passed in13.54s（包含本次新增门槛一致性测试）；上一次全量858 passed/71 failed是此单行控制条件修改前的结果，其失败集合与此前相同，不宣称本次全局全绿。编译/diff通过，未提交/上传。接下来沿同一V19观察阶段推进。

## 2026-10-21 实测位姿速度的精度处理（当前运行 V18）

V17已能产生四点真实有效接触，但100µs位姿差分的孤立速度尖峰仍会重置持续计时（687步时644帧有效，最长连续17.5ms）。新增显式 `--mesh-motion-window-steps 10`，用至多1ms连续实测位姿估计速度，位置仍取当前实测值，物理步长仍100µs，GPU杆内步10µs。首帧不伪造零速度，跨冻结/动态切换重置历史，记录实际估计区间；这是接触运动边界的仿真近似。原1mm/s接触速度阈值、碰撞/载荷/滑移门槛不变。测试确认量化尖峰消除时，真正4mm/s的快速运动仍保留。

已保存V17有效1000步后正常停止，单实例V18从该物理状态继续close，不重跑approach。最近93步有效，四点载荷约.939/1.058/.938/1.045N，已连续7.2ms通过四点严格接触检查；**尚未完成稳定抓握、hold或lift，尚未正式蒸馏**。下一步沿同一实例继续观察实际阶段进展，不因等待过期而重开。

专项183 passed in15.12s；最终全量858 passed/71 failed in58.93s，与上次失败集合相同。编译/diff通过；未提交/上传。

## 2026-10-21 接触运动边界与恢复线缆几何修正

V15的保持指令已不再变化，但报告分离速度仍高。实际500–900步中右Panda七关节位置完全相同；拇指根部按位姿变化约.005mm/s，按报告关节速度经Jacobian换算却约3.5mm/s。新增显式 `--mesh-motion-velocity-source pose_delta --motion-prediction-order 1`：接触运动边界使用连续实测位姿的平移/旋转差分，记录原报告速度供比较。第一帧无位姿历史时仍使用报告速度。**这是仿真运动边界的一阶近似，不是硬件参数校准，也没有替换GPU柔性杆物理或降低接触速度阈值。**

续接检查改用保存状态中的实际线缆节点和体积守恒半径，而不是默认直线wire。同一保存姿态：默认直线余量.9435mm失败，实际恢复线缆余量1.2428mm通过；原1mm要求不变。V16因此在步进前退出，保留了证据，未覆盖原状态。

单实例V17从V15有效1000步继续close（不回放approach），最新114步有效，累计89帧四个夹持点同时通过原严格接触判据，载荷约.930/1.066/.955/1.061N；拇指分离速度约.013/.036mm/s。**仍未完成稳定抓握、hold和lift，正式蒸馏尚未启动。**专项182 passed in13.31s；运动边界修改后的全量856 passed/71 failed，失败集合与之前一致；恢复入口修改后的最终全量正在运行，完成后更新。未提交/上传。

## 2026-10-21 预载达标后的目标保持交接

V14从保存的接触状态续跑至2891步仍未稳定夹持，四点已有接近1N的载荷，但MPC指令持续改变。已保存有效1200步，正常停止旧实例后加载控制修改；不是重跑approach。

新增逐手预载保持：当前实测接触达到目标95%时固定该手的上一关节指令，跌至90%以下或接触丢失时恢复调节。MPC发布的预测指令也受同一保持约束；实测Jacobian/受载预测继续保留。教师/学生使用相同交接，完整原始采集记录保持状态。保持时参考接触深度跟随实测，不积累未执行的过闭合请求。它不授予抓取资格，原接触速度、载荷、滑移、碰撞和阶段持续时间要求不变。

V15已在原双Panda/DH116/GPU柔性杆场景续验，最近378步有效close，双手保持已触发，实际指令不再变化；接触运动仍在衰减，**尚未通过稳定抓握、hold或lift**，尚未用失败数据训练成功模型。专项179 passed in13.49s；全量854 passed/71 failed in59.33s，与上次失败集合相同。编译/diff通过，未提交/上传。运行状态须实时重查。

## 2026-10-21 指腹中心对准与过闭合回退

V13 已终止在468步：此前四个真实夹持点达到约 .950/1.058/1.090/1.213 N，但左食指局部指腹余量 .9985 mm 未通过1 mm要求；外层非指腹检查通过。不是抓握成功，不训练为成功示范，也未放宽区域。

新增离线 `align_contact_centres` 和 `scripts/align_dh116_contact_initialization.py`，只调整接近终点，close时仍冻结Panda。保留原标定点/区域、碰撞间距、源USD和wire状态；同时约束当前mesh-wire距离、掌心位移/转角及关节范围。V13有效467步的静态候选：双掌位移均<1 mm，旋转约1.407°/.835°，间隙变化<.06µm；原左食指局部余量提高到约1.578 mm。完整50体源几何、六个pad区域及运行器的名义初态筛查通过；这些只证明静态候选，不是已执行路径或抓取成功。生成文件在本地 `runs/dh116_contact_alignment_20261021_v1/`。

载荷参考增加反积分饱和：实测达到目标后，从真实当前接触深度回算，避免保留历史深闭合参考还继续挤压；原力矩、接触刚度和成功门槛不变。新初态保留wire动态、关节速度与主动关节初始驱动偏差，不导入任何阶段资格。

V14 正在上述候选上实际续验（不回放approach）：完整双Panda/DH116、GPU杆、Kp10/.15Nm模拟配置不变。最近确认1430步有效close，四点约 .974/1.082/.948/1.042 N；但四点同时通过稳定接触判据的帧数仍为0，主要是拇指相对分离速度，未hold/lift。下一步需核对达到预载后的目标保持交接，让原判据验证稳定；尚未实现该交接，不直接删判据或将载荷达标当成功。

专项171 passed in6.39s；最新全量 `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q --tb=short` → **846 passed, 71 failed in58.58s**，与上次失败集合完全一致；编译/diff通过。无正式成功轨迹模型，无提交/上传。进程状态须实时重查，不凭本段静态记录重启。

## 2026-10-21 实测预测偏差校正与受载限位续验

完整区间记录揭示旧 MPC 的受载跟踪偏差：对 V11 的已执行动作做因果离线核对，最近十区间关节预测 RMSE 为 0.001077 rad；只用此前实测误差的有界偏差校正后为 0.000152 rad。这是预测核对，不是新的物理成功。现 MPC 增加该因果观测器，并在受载平衡目标中保留 1 mrad 限位余量；每步实际命令仍遵守原关节限位和速率上限，不通过钳制预测状态伪造可达性。

V11 已终止：2815 步右拇指实际值低于仿射耦合模型下限约 2.49e-7 rad，最后有效2814，未 hold/lift。V12 从同一 close 的2400步接触状态继续，携带原载荷参考与既有完整区间校准的偏差，避免恢复时把预载参考重置到初值；四点升至约 .761/.926/1.066/1.153 N，但200步右手跟踪误差 .017648 rad 超过原1度门槛而停止。原判据未放宽。

V13 从 V12 第199步有效 q/qd/杆动态继续，无 approach 回放。保持 .15 Nm 手部力矩上限，将**仿真 Kp 从6调到10**，按旧/新增益比例重算目标以保留初始比例力矩；增益变化时旧预测偏差作废并重新估计，径向预载参考保留。这不是 DH116 实机标定。最近确认第53步仍有效 close，四点约 .706/.838/1.032/1.112 N，尚未持续夹持或抬升成功。

正式完整数据训练另外要求 `whole_wire_clear=true`；原观察器区分“抓取站点提升”和“整线离桌”，训练不再只看前者。原诊断观察器行为不改，接触区/1 mm碰撞余量/穿透限制不改。专项 **159 passed in12.56s**，编译和 diff 检查通过；最新全量 `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q --tb=short` → **840 passed, 71 failed in58.54s**，失败用例集合与上次811/71完全一致，无新增失败但全量未通过。V13随后第259步仍有效close，四点约.769/.913/1.063/1.211N。尚未正式蒸馏成功源数据，未提交/上传。

## 2026-10-21 完整专家采集与反馈历史蒸馏

按用户确认的“完整采集＋成功专家闭环＋蒸馏”推进。新 `contact_decisions.jsonl` 使用 `dh116-contact-decision-v2`：每次动作前保存手部 q/qd、上一指令、实际指腹/目标/Jacobian、接触载荷/间隙/有效性及其已知标记、掌心/法向、完整杆位置/速度/director/角速度/半径和非目标部位余量；保存 MPC 预测与实际提交动作，并在下一控制时刻补齐整个执行区间的实测结果。hold/lift 明确记录保持动作，不伪造新 MPC 规划。手部动作仍为 12D，Panda 保留在场景中并只负责接近/抬升。

新增反馈/三帧历史策略：学生 267 维，40 单元杆的教师 1265 维（包含完整杆动态、点 Jacobian 和临近余量）。教师特权状态和未来执行结果不进入学生输入。输出仍是以上一指令为参考的下一次关节目标，由连续反馈执行，不是一次性保证成功的完整轨迹。当前学生几何/接触观测来自仿真，尚未接实机视觉或触觉估计器。旧 `.pt`/JSON 模型保持显式兼容，新版不混用旧输入维度。

`train_dh116_contact_imitation.py` 默认要求新版完整采集文件、成功运行和实际 close/hold/lift 三阶段区间；`--legacy-contact-replay` 才使用旧 42/78 维诊断格式。`--allow-incomplete` 仍只能生成明确标记的部分动作诊断。正式训练以**实际提交且执行验证的专家指令**为标签，不把预测序列中未执行的后续动作当成成功示范。当前未生成成功的新教师/学生模型。

实际专家进展：V10 在第 1031 步因局部指腹粗采样的保守余量停止。已复现并仅对临界局部采样细化：同姿态细采样余量约 1.20–1.21 mm，原 6 mm/30 度区域、1 mm 间距和穿透判据不变；可移植测试保留真实非前部接触拒绝。V11 从 V10 第 1030 步的 q/qd/目标和 GPU 杆动态继续 close，不回放 approach。已实际读回 13 个完整控制区间；第 1325 步仍有效，四点约 0.608/0.701/0.815/0.886 N，尚未进入 hold/lift。运行状态须重新查进程/末行，不能根据本段假定仍活。当前是在线 MPC，不是学生控制。

本轮专项命令（新策略/采集、MPC、接续、区域、CUDA、回放/控制）**150 passed in 12.29s**，编译和 diff 检查通过。正式训练入口对尚未成功的 V11 明确拒绝，未创建模型输出目录。新训练入口的合成夹具测试仅验证数据/模型接线，不是物理成功证据。本轮未重跑全仓库；上次全量 **811 passed, 71 failed**，未宣称全量通过。杆物理在 CUDA，但在线 MPC 仍是局部模型而非逐候选完整 Cosserat rollout。

待完成：实际抓握/保持/Panda 抬升、成功源数据正式训练、独立教师/学生整机验证及留出场景评估。设计与逐项进度见 [确认方案](docs/superpowers/specs/2026-10-21-dh116-complete-expert-distillation-design.md) 和 [实施计划](docs/superpowers/plans/2026-10-21-dh116-complete-expert-distillation.md)。本轮未提交/上传 GitHub，未修改源 USD。

## 2026-10-21 受载 MPC、模型运行接口与接触续验

局部 MPC 预测加入 `J.T @ normal_load / Kp` 的手指受载平衡偏差，避免把接触载荷造成的关节偏差误判为“继续保持就能自动闭合”。实测接触几何的 USD/Jacobian 核算显示，每个夹持点 1 N 对应主动轴约 0.074–0.079 Nm；采用仿真 Kp=6 Nm/rad、原 0.15 Nm 力矩上限，增益变更时重新计算目标以保留初始比例力矩。此为仿真调参，非 DH116 实机标定。

`--hand-policy` 已支持新教师/学生 `.pt` 和旧 JSON，`--hand-policy-role teacher|student` 明确角色；新模型验证控制周期，保留未参与夹持手指的停放指令，记录训练范围。模型数据仍是 hand-only 12D，Panda 不进入动作或学生输入。已修复 pinch 模式同步 lift 误要求停放中指接触的问题；默认 middle-hook 模式仍保留承托要求。部分轨迹模型的 CUDA 推理探针（200 次，未向机器人执行）约 teacher 0.556 ms / student 0.679 ms，不代表已学成抓取或整个仿真实时。

V8 从约 0.14 N 加载到 0.50–0.55 N，随后第 485 步因指腹非接触区保守下界 0.999842 mm 停止。细化计算得到约 1.20 mm 下界；现仅对临界线段加密采样，**不改 1 mm 间距、接触区域或穿透门槛**。V9 从其第 484 步继续 close，在第 259 步因非指腹 `finger13_link` 的粗估余量 0.826 mm 终止，四载荷约 0.527–0.589 N，未夹持/lift 成功。

已对临界凸包线段使用平面包络凸最小化及 Lipschitz 误差界，保持原保护要求；V9 失败姿态复算的完整源几何下界为 1.0694 mm。另一个离线手部求解表明，仅拇指两主动轴反向配合即可保持当前 wire 接触间隙并把该指节余量提高到 1.6 mm，但尚未验证物理路径。下一步将此避让需求加入手部 MPC 目标，而不只依赖事后停机。V9 第 258 步 q/qd/目标、杆动态和初始载荷已保存；当前没有运行中的 Isaac 实例，不重跑 approach。

本次 GPU、MPC、接续、模型接口及区域回归专项 **111 passed, 4 deselected in 9.21s**；新增凸包误差界与源几何专项 **27 passed, 1 deselected in 5.94s**。排除项为已知旧四通道测试，不冒充通过。最新完整回归 `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q --tb=short` → **804 passed, 71 failed in 29.00s**；编译和 diff 检查通过。全量仍未通过，实际持续夹持/lift、成功源训练及模型整机验收均未完成。

## 2026-10-21 Cosserat 杆全 CUDA 与接触状态续验

Cosserat 杆物理已接入 `--rod-backend torch_cuda`：剪切/伸缩、弯曲/扭转、平移/转动惯性、重力/阻尼、有限桌面与运动指腹三角网格接触、摩擦和双侧反作用冲量在 CUDA 计算。40 单元、50 cm 长、8 mm 半径及原 USD 保持不变。CUDA 不可用时明确报错，不自动退回 CPU。这里的“全 CUDA”指杆求解与接触；Isaac 位姿/反力交换、源几何检查、日志和显示仍有 CPU 工作，尚无实时提速结论。

本次验证：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_gpu_cosserat_physics.py tests/test_gpu_cosserat_runner.py tests/test_gpu_cosserat_backend.py -q` → **27 passed in 5.26s**；含 PyElastica 内力/力矩及阻尼对照、摩擦反作用平衡、移动网格预测、生产尺寸柔性形变与状态恢复。控制、GPU、接触初始化和教师/学生读取相关专项合计 **93 passed in 10.41s**，脚本编译和 diff 检查通过。

全量 `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q` → **788 passed, 71 failed in 53.79s**。多项旧四通道测试/资源与当前六通道契约不一致（含 CPU moving-contact、抓取证据与反力桥），另有旧 approach 时序预期等失败；相关四通道测试及六通道 CPU 实现在本次工作前的 HEAD 已存在不一致。全量未通过，不将专项结果当作全仓库通过，也不通过退回四通道来掩盖这些失败。

完整双 Panda/DH116 的 V6 实测四个夹持点均接触 wire，但仅约 0.012 N，未夹持/抬升成功。已补 MPC 法向优先目标与实测载荷反馈，并保存第 4727 步完整关节 q/qd/目标、柔性杆位置/速度/director/角速度。V7 使用 `--grasp-entry close --close-initial-state ... --rod-backend torch_cuda` 从该接触状态开始，不重跑 approach，也不导入成功资格；第 220 步四点约 0.047–0.049 N，仍在 close，尚无 hold/lift 成功。后续以实际持续夹持和 wire 抬升为验收，再采集成功轨迹训练；离线损失下降不代表抓取成功。

## 2026-10-21 close 直入与恢复姿态保持

从 `stage_close.json` 恢复时，`restore_validate` 阶段保持当前实测关节姿态，校验通过后直接进入 `close`，不会重新回放 `approach`。这样可以把接近、闭指、夹持和抬升作为连续阶段运行，同时保留源几何碰撞保护与真实接触负载证据。

## 2026-10-20 连续 approach 的 wire 计算策略

正式连续序列支持 `--approach-wire-mode frozen`：`settle` 阶段正常推进 DeformX，使 wire 落桌并稳定；进入 `approach` 后复用该稳定快照，只推进 Panda/DH116，并持续执行源几何与碰撞保护；进入 `close` 后自动恢复 DeformX、指腹接触冲量和机器人反作用力，随后连续进入 `hold/lift`。不需要人工暂停或重启。默认 `dynamic` 保持原行为。该模式只减少 approach 的 wire 求解，不代表已完成闭指、夹持或抬升验收。

## 2026-10-20 性能优化 Task 1

已加入可选 `--profile` 阶段计时：输出逐步 `perf.jsonl` 与按阶段汇总的 `perf_summary.json`，记录控制、DeformX 杆/接触、反作用提交、PhysX、状态传感器、源碰撞保护和证据写入耗时，并把提交 SHA、设备、时间步、接触采样数、CPU 亲和性和数值线程环境写入运行清单。默认关闭 profiling，不改变物理参数、单核约束、接触反作用或碰撞保护。

本阶段验证：`bash scripts/python.sh -m pytest tests/test_dh116_perf_meter.py tests/test_dh116_sequence_runtime.py -q` → **31 passed**；`python -m py_compile scripts/verify_fixed_dual_panda_dh116.py src/go2_pvcnn/control/dh116/perf_meter.py` 通过；`git diff --check` 通过。尚未完成完整回归和 Isaac 性能对照；实际闭指、持续夹持、抬升仍未验收。

## 2026-10-20 性能优化 Task 2

wire 的 GUI 发布改为墙钟调度，默认 15 Hz；每个物理步仍固定调用 `sim.step(render=False)`，到期后只调用一次 `publish_wire` 和 `sim.render`。网格拓扑在首次发布时缓存，节点数、面计数或索引变化会拒绝复用。物理步长、接触反作用、源碰撞保护和单核约束未改动。

本阶段验证：新增调度和拓扑测试，完整回归 `bash scripts/python.sh -m pytest tests -q` → **734 passed in 24.68s**；脚本编译和 `git diff --check` 通过。尚未做正式 Isaac 三次同物理时长性能对照，也未完成闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 3

新增不可变运行快照和可复用反作用缓冲。每个物理步保存独立的 pre/post token，接触反作用记录对应的杆状态 token；左右臂反作用张量和环境索引只分配一次，后续原地清零复用。接触反作用仍按每步一次提交，安全门、重力保持和碰撞保护不变。

本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **736 passed in 24.57s**；脚本编译和 `git diff --check` 通过。尚未完成证据缓冲、三次同物理时长 Isaac 性能对照，以及闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 4

`EvidenceJournal` 增加 `audit` 与 `buffered` 两种 flush 策略。默认 audit 逐行落盘；使用 `--evidence-flush-policy buffered` 时按 `--evidence-flush-steps` 或 `--evidence-flush-seconds` 刷新，阶段切换、失败和关闭始终强制 flush。轨迹行数和字段保持不变。

本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **739 passed in 24.52s**；证据、静态配置和脚本编译检查通过。尚未做正式 Isaac 三次同物理时长性能对照，也未完成闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 5

新增 `scripts/run_performance.py` 和 `scripts/collect_performance.py`。runner 按 variant、phase 和 repetition 串行启动单个 verifier，固定资产、`physics_dt=1e-4`、wire `py_dt=1e-5`、16 点接触采样和单核启动约束；collector 要求每组至少三次、检查物理身份和完成步数，再输出中位数、P95、SPS 与 RTF。输出目录必须位于 `runs/` 且单次运行独占。

本阶段验证：`bash scripts/python.sh -m pytest tests/test_dh116_performance_runner.py -q` → **3 passed**；完整回归 `bash scripts/python.sh -m pytest tests -q` → **742 passed in 24.56s**。尚未启动正式 Isaac 对照，闭指、持续夹持、抬升仍未验收。

## 2026-10-20 性能优化 Task 6：首次 Isaac 对照

按相同物理时长对 baseline、buffered 和 GUI 15 Hz 各串行运行 3 次，全部 1200/1200 步通过 settle/diagnostic 物理门。baseline 中位墙钟 **36.953951 s**，buffered **36.933635 s（1.000550×，处于噪声范围）**，GUI **44.014520 s（0.839585×）**。因此当前证据 flush 没有确定整机提速，GUI 显示增加墙钟成本；主要热点仍是 DeformX 杆/接触求解。详见 [性能报告](docs/superpowers/reports/2026-10-20-dh116-performance-report.md) 和本地 `runs/perf_20261020_settle/report.json`。

这组运行只覆盖 0.12 s settle/diagnostic，没有进入闭指、持续夹持或抬升，不能作为抓取成功证据。下一步需针对杆/接触求解热点单独设计优化，并保留完整物理验收门。

## 2026-10-20 上传快照

分支：`feat/dh116-simulation-10-20`。当前阶段为固定双 Panda + DH116 的抓取控制开发、半隐式接触离线验证和 Isaac 接触前探针验证；持续夹持与抬升尚未通过实际验收。

本次汇总：前伸掌心朝下与阶段内对称姿态配置、分阶段速度控制、按渲染帧发布 wire 网格、接触粗筛与运动预测优化，以及实验半隐式接触求解器、杆 Jacobian/动力学适配和基准脚本。半隐式后端尚未接入正式 Isaac runner，尚无整机实时提速结论。

本次上传前验证：`bash scripts/python.sh -m pytest tests -q` → **726 passed in 26.94s**；`git diff --check` 通过。配置仍含本机路径；`forward_top_grasp_v1` 为历史未完成配置，使用前需核对各配置验证记录。下一步继续半隐式长时与实际指腹数值/性能验证、运行接线以及接近、闭指、夹持、抬升实测。

运行数据 `runs/`、历史归档 `archive/`、缓存和日志保留本地，未包含在此上传快照中。下方为历史阶段记录。

## 最新 Isaac 探针

独立单核 Isaac `closure-pilot` 已启动并正常步进，wire 保持在桌面上；该模式固定预姿态，约 `0.108 s` 仿真时间内四指负载均为 `0 N`，指腹距 wire 约 `19.5–22.4 mm`。它不是 approach，因此尚未进入闭指或抓取。输出：`runs/isaac_contact_probe_20261019/`。下一步运行带 pinned approach 的正式序列。

## 2026-10-19 最新数值验证

半隐式接触实验后端已完成 10 ms 静态与移动边界的步长减半检查（`2.5→1.25 µs`）。两种工况相对旧显式后端均满足位置、峰值力、冲量和线性反作用平衡门槛；两组新后端末状态差异为：静态位置 `0.33 µm`、移动边界位置 `0.33 µm`，反作用平衡均为 `0 N·s`。报告：`runs/implicit_contact_10ms_{2p5us,1p25us, moving_2p5us,moving_1p25us}_20261019.json`。

这只证明合成杆/网格工况的离线数值一致性；实验后端尚未切换正式 Isaac CLI，也没有证明 DH116+Panda 的真实闭指、持续夹持或抬升成功。

## 开发中：先轻触再夹持

路线1短时数值门已跑通：实验后端与旧显式后端各 0.001 s、内步长 2.5 µs 对照。静态工况位置差 `1.65e-7 m`、峰值力差 `0.00468 N`、冲量差 `2.83e-6 N·s`；10 mm/s 移动边界位置差 `6.43e-6 m`、峰值力差 `0.00523 N`、冲量差 `6.36e-5 N·s`；线性反作用平衡均为 0，短时门槛通过。证据 `runs/implicit_contact_short_{static,moving}_20261019.json`。这不是 100 ms 收敛或真实 Panda 抓取证明，下一步仍需延长时段并做步长减半。

路线1 Task3 已接入实验后端 `co_sim/implicit_moving_contact.py`：不注册旧 `RodMeshContact`，在每个 Verlet 子步联合桌面与四个指腹接触，使用真实杆质量/伸长响应和 Jacobian，一次写入节点力/材料力矩并记录网格反作用。专项 **7 passed**（含无接触与原显式后端逐状态对照、指腹反作用单次提交），全量 **726 passed in 24.88s**。尚未切换正式 Isaac；仍需多接触冲量平衡、细步长收敛和速度基准。

路线1继续：新增单次接触冲量提交适配器 `co_sim/implicit_stepper.py`，通过实际 Jacobian 将联合接触力转成广义力与冲量，明确由调用方只提交一次；5项专项测试通过，全量 **724 passed in 34.84s**。该组件尚未接入 MovingContact/Isaac，仍不能宣称速度提升或抓取成功。

最新实际杆适配：已实现带伸长因子的质量/转动响应及接触点Jacobian，8项先失败后通过，直接对照DeformX实际加速度核和接触力核（1/4/16子采样），全量 **719 passed in 26.09s**。下一步为一次施力的Verlet接线及反作用统计；目前尚未启用新算法运行，也未取得稳定夹持或抬升证据。

最新接触积分进度：详细方案已确认，新增独立联合半隐式速度求解器（尚未接入杆后端/Isaac）。19 项先失败后实现通过，补充旋转、转动力臂与分离检查后针对性 22 passed；本轮 `bash scripts/python.sh -m pytest tests -q` 为 **711 passed in 25.90s**。保留原摩擦律，多个接触共享速度求解，不收敛明确报错。下一步适配实际杆的质量/伸长转动响应、桌面及一次冲量提交，再做细步长对照。没有整机提速或抓取成功结论。下文待确认等记述为此前历史状态。

最新：独立指腹以 10 mm/s 竖直运动的 100 ms 工况中，2.5→1.25 µs 对比也通过容差（位置差 0.000649 mm，峰值力差 0.000122 N）。不是实际机器人闭环抓取证明。已去掉运动预测内部对已冻结输入的重复校验，保留公开输入及完整输出检查，并将预测模块加入后端指纹。先失败后实施，全量 **689 passed in 51.68s**；函数微基准约减时 46%，不是整机速度。旧 Isaac 未热加载，尚未稳定夹持/lift；新的稳定接触积分方案待确认。

最新参考解：静态双侧接触 100 ms、原摩擦参数下，2.5→1.25 µs 的位置差 **0.000625 mm**、峰值力差 **0.000950 N**、累计冲量差最大 **1.35e-7 N·s**，三个工程容差均通过。仅证明这一固定工况的细步长收敛；耗时约 47.5/94.8 s，不是实时提速方案或实际机器人抓取证明。生产内步长未改，后续需以此参考验证稳定求解及移动边界。

数值诊断补充：仅诊断关闭切向摩擦后，10→20 µs 的中心线差降至 0.000749 mm，表明主要敏感性在摩擦阻尼路径；生产摩擦未改。保留原参数时 5→2.5 µs 的位置差 0.123 mm 已满足位置容差，但右拇指峰值力差 0.0587 N 超过 0.05 N，整体仍未通过收敛门槛。后续需更细参考解或改进阻尼积分，不能把关闭摩擦当抓取方案。实际夹持/抬升仍未完成。

**最新精度限制：**100 ms 静态双侧接触内部步长实验中，20 µs 相对 10 µs 的线缆位置差约 1.585 mm；10 µs 与 5 µs 仍差约 0.548 mm，超过设计的 0.2 mm 容差。不能直接扩大内部步长提速，当前 10 µs 的该工况数值收敛也未确认。旧 Isaac 保持原参数运行作调试证据，不据此宣称最终抓取验收。新配置状态迁移尚未执行，旧状态未改写。

接触查询优化已实现：保守点到 AABB 距离粗筛编译化，剔除包围盒角落伪候选，不改真实网格、接触参数或采样数。全量 **686 passed in 32.96s**，另后端接触测试 **43 passed in 5.37s**。三组静态双侧后端交替基准耗时中位数 1.908→1.591 s，采样位置与累计冲量逐位一致；不是 Isaac 整机实时速度证明。现有旧实例未热加载，最新 step 47263/close，载荷仍零，夹持/抬升未完成。

最新后端诊断：固定真实指腹网格，外层 0.1/0.25/0.5/1 ms、杆内 10 µs 的落体/单指/双侧对比已完成。双侧四路峰值约 0.642/0.631/0.970/0.767 N，数值基本一致，但耗时 1.753→1.804 s，未测得提速。单次静态网格测试不含机器人反馈，不批准更大运行步长。下一步优先优化杆内部接触查询。实际 Isaac step 45802 仍为 close、载荷全零，未完成夹持/抬升。本轮针对性测试 32 passed in 0.51s。

最新状态：用户已接受候选 14 的倾斜前伸姿态，`configs/forward_top_grasp_v2` 在 `runs/forward_gui_render10` 实际运行，接近通过并保存闭指起点；本次检查 step 41119，仍在轻触闭指，未完成有效夹持/lift。当前实例曾发生 GUI 暂停恢复，不作为不间断端到端验收。

提速实施中：线缆视觉网格按渲染帧更新，静态拓扑只提交一次，物理/接触/保护仍逐步执行。新增测试先失败后通过，完整回归 `bash scripts/python.sh -m pytest tests -q` 为 **655 passed in 32.09s**。离线 USD 视觉发布基准 1001 次循环、3 次中位数为旧逻辑 3.25348 s、新逻辑 0.05485 s；不是整机加速倍数，当前 Isaac 未热加载新代码。分阶段动作提速、较大耦合步长及真实抓取验收仍未完成。用户已授权全核，覆盖历史单核要求；不能将开放 CPU 亲和性等同于自动并行。详见 grasp-speed 设计和计划。

最新组件进展：已接入显式 `--grasp-speed-profile staged`，包含实测间隙调速、载荷闭合、10 mm/s 候选同步抬升、恢复模式接线和关节速度分配日志。自由空间触碰速度包络最高 20 mm/s、最后 2 mm 为 1 mm/s、带载速度上限 1 mm/s；初始认证接近路径尚未重新定时。旧默认行为不变，当前实例未切换新模式。针对性 58 passed，全量 **682 passed in 27.41s**。离线当前姿态分配发现共同限速约 3.019 倍，候选拇指 9.327/1.000 mm/s 变为模型分配 3.089/0.331 mm/s；不是实测跟踪，不能称已达到人手速度。后续仍需速度/跟踪可行性、耦合对比和实际抓取验收。

以下为历史阶段记录，不能覆盖以上最新状态。

对称抓取调整实施中：镜像仅用于本任务抓取/lift，不约束后续非对称操作。功能坐标镜像、双侧独立运输速度分配、lift 高度协调和运行接线已实现；运行接线后完整回归 634 项通过（历史验证日志 `/tmp/dh116-symmetric-runtime-regression.log`）。`runs/symmetric_gui1` 实际首步因右拇指初态紧贴耦合下限失败，不是抓取成功。新 `configs/symmetric_grasp_v2` 在预抓求解中留 0.005 rad 关节余量，130 次离线路径检查及 USD FK 点 Jacobian 校验通过；已启动 `runs/symmetric_gui2` 实测，尚未完成闭指/夹持/lift 验收。旧证据保留，未放宽限位或碰撞保护。

最新演示：按用户要求切换到真实 Isaac Sim GUI，`runs/touch_first_gui1` 已显示双 Panda、DH116、桌面与橙色 DeformX wire，并开始实际步进。修复了 GUI 仍固定 `render=False` 的问题，增加观察相机及照明；未修改物理控制参数。针对性测试 3 passed in 0.14s，完整回归 **595 passed in 52.51s**；尚未完成抓取抬升验收。原 `touch_first_trial2` 在 step 6400 通过协作停止退出，原件保留，属于操作员中止而不是抓取算法失败。

独立目录实际启动发现并修复遗漏的共享接触检查导入；已补纯 DH116 模块且保持原检查逻辑。修正后完整回归 **594 passed in 24.51s**。`runs/touch_first_trial1` 为启动失败证据，`runs/touch_first_trial2` 已实际物理步进（检查时 step 361，settle），**尚未完成新策略接近/闭指/夹持/抬升验收**。新实例从初态验证新指纹版本，不把旧 checkpoint 冒充兼容恢复。

功能分支已接入“实测到位低速 → 拇指轻触 → 食指闭合 → 平滑增载”，包括实测关节速度、限速接管和恢复后重新确认接触。前次完整离线回归 **592 passed in 51.03s**（包含启动载荷校验修正）；**新版本尚未完成 Isaac 物理验收**，不应把测试通过当作抓取成功。旧控制器实例已结束，始终未热更新。细节见 CHANGELOG。

## 最近上传：2026-10-18

当前阶段：独立工程隔离与离线验证；实际双手接近试验进行中，尚未确认闭指、持续夹持和抬升成功。

本次改进：独立整理 DH116 控制、USD 依赖、接触配置和 DeformX 后端；增加上传排除规则及持续进度记录。本次本地回归 **559 passed in 51.52s**，运行配置预览通过，未启动第二个 Isaac。详细变更见 [CHANGELOG.md](CHANGELOG.md)。

GitHub 版本不包含约 4.7 GB 的 `archive/` 历史证据、运行日志及缓存；原件保留本地。部分测试和配置重建工具依赖该归档，**仅克隆仓库不能复现全部本地测试**。当前运行配置仍含此机器的绝对路径与指纹，异机需重新生成配置，不可直接修改校验值冒充兼容。厂家资料与第三方资源保留原有权利归属，本仓库不授予额外再分发许可。

当前为固定双 Panda + 普通 DH116 V2.0 + 桌面 DeformX 柔性线缆项目。不是 DH116S，不包含机器狗任务或 O6 资产。`fixed_dual_panda_o6` 命名空间内仅保留 DH116 使用的接触桥和线缆可视化两个共享模块。

## 当前迁移状态

- 1380 个静态来源文件已复制并 SHA 校验，原目录未删除；详见 `migration_manifest.json`。
- 主模型的 5 个 USD 层已隔离到 `assets/`，依赖相对引用；`OmniPBR.mdl` 由 Isaac 软件环境提供。
- 新运行配置已重新校验指腹表面坐标、接触几何、源路径 94 个区间和 loader 指纹。
- 主运行入口读取本目录的 DeformX；Python 启动脚本重置 PYTHONPATH，仅使用本项目及安装的软件库。
- 主运行、空载验证、资产生成及测试资源路径已适配，独立目录全量 **559 passed in 50.95s**，见 `regression.log`。共享测试中仅排除了不适用的旧 O6 资产专项测试，DH116 接触身份测试保留。
- **实际运行迁移尚未完全验收**：新目录实际 Isaac 验证尚未执行。原目录当前试验结束前不启动第二个实例。
- 原活动试验 `table_margin_raise1mm_trial1` 未纳入静态归档，结束后才能同步最终证据。

## 内容

## 2026-10-21 — DH116 hand-only parallel MPC / teacher-student / GPU rod

- 训练对象已与 Panda 机械臂完全解耦：动作空间固定为左右 DH116 各 6 个主动关节（12D），保留上一动作作为连续性输入；不包含 approach 阶段，只收集 grasp→lift。
- `ParallelJointMPC` 支持批量状态并行规划，动作受 DH116 主动关节限位约束；`scripts/train_dh116_teacher_student.py` 生成教师/学生 JSONL 回放和带 SHA256 的 manifest。
- 新增显式 `TorchCudaCosseratRod`（`backend=torch_cuda`）。CUDA 不可用时直接失败，不静默退回 CPU；该后端是 gravity/damping 的显式 Torch 近似，不能宣称与 PyElastica 或实机动力学等价。
- 离线验证：手-only MPC **9 passed**；GPU 后端选择/无 CUDA 失败路径 **2 passed**；教师/学生采集 smoke 已生成 12 条回放。尚未完成 Isaac 物理抓取或实机验收。
- 新增 `scripts/verify_dh116_hand_only_wire.py` 接入 IsaacLab 双 DH116 + wire proxy（无 Panda、无 approach）。首轮 12 行 headless 接入发现当前回放姿态/资产碰撞会产生数值爆炸，已明确判定失败；尚未宣称接触或 lift 成功。
- 验证器可直接加载学生 checkpoint，以 Isaac 关节反馈推理动作，按单条 episode 顺序执行，并限制目标速率。完整运行 24 个策略动作/2400 个 PhysX 步后手部状态稳定，但安全 wire 高度下接触为 0 N、抬升为 0 m；把 wire 降到现有手指轨迹高度会在首个物理步产生超过 500 N 冲量并立即停止。抓取和 lift 尚未通过。
- `--wire-z` 现可用于几何定位。学生策略在 `wire_z=0.36 m` 时触发接触，但 8 步内出现 363.99 N 峰值、wire 非预期上移 33.65 mm，稳定性门禁停止；这不是合格 lift，说明手指/线缆存在初始干涉或过大接触冲量。

```bash
PYTHONPATH=src /home/xk/miniconda3/envs/go2/bin/python scripts/train_dh116_teacher_student.py \
  --device cpu --episodes 4 --steps-per-episode 24 --out runs/dh116_hand_only_teacher_student
PYTHONPATH=src /home/xk/miniconda3/envs/go2/bin/python scripts/verify_dh116_hand_only_training.py \
  runs/dh116_hand_only_teacher_student/manifest.json
```

`src/` 控制与共享依赖；`scripts/` 运行/校验；`tests/` 回归；`assets/` 新运行 USD；`configs/` 新配置；`vendor/DeformX/` 实际后端；`resources/manufacturer/` 厂家资料；`archive/` 历史原件；`docs/source/` 设计与记录；`tools/` 可审计迁移工具。

归档中的旧路径和 SHA 保持不变。它们不是新目录可直接恢复的状态；禁止修改旧 stage pin 冒充兼容。新配置的模拟参数不代表实机额定参数。

## 环境和命令

使用已有 `/home/xk/miniconda3/envs/go2/bin/python`、Isaac Sim/Isaac Lab、GPU0。可用 `DH116_PYTHON` 指定兼容 Python 环境；不复制整个 conda 环境。所有入口通过 `scripts/python.sh` 限定 core0，数值库单线程。

```bash
cd '/home/xk/coding/dexterous hand'
bash scripts/python.sh -m pytest tests/test_migration_copy.py tests/test_asset_isolation.py tests/test_standalone_layout.py -q
# 完整回归（包含真实 DeformX 后端测试，不启动 Isaac）
bash scripts/python.sh -m pytest tests -q
# 只打印命令并验证配置，不启动仿真
bash scripts/python.sh scripts/run_current.py
# 原实例结束后再运行；存在其他 DH116 verifier 时会拒绝启动
bash scripts/python.sh scripts/run_current.py --run
# GUI 版本需指定独立输出目录
bash scripts/python.sh scripts/run_current.py --run --gui --output runs/gui_trial1
```

这些入口尚待新目录实际物理验收，不能将命令可打印当作抓取成功。

## 抓取进度

**本次原实例已结束，整体未通过。** step 76165 的右拇指非指腹区域间隙下界降至0.99994 mm，触发1 mm保护；不是已确认发生穿透。最终 settle/approach/close/hold=PASS、lift=FAIL。最后完整步两站升高约1.139/0.219 mm、滑移1.332 mm，未完成目标抬升。保留全部阶段状态和失败现场，下一步修正右手对夹姿态并验证全程间隙；不放宽阈值。

原实例进一步在 step 68872 完成连续 1 秒 hold，保存 `stage_lift.json`，进入 lift。step 68935 检查通过，四指载荷约 0.965/0.918/1.040/1.043 N；仅刚启动抬升，**尚未证明线缆抬起或达到 30 mm 高度目标**。

2026-10-18 原实例 `table_margin_raise1mm_trial1` 在 step 58872 通过 close 并保存 `stage_hold.json`，进入实际夹持保持。step 58968 四指法向载荷约 0.948/0.896/0.992/1.014 N，hold 基线后滑移约 0.008 mm。这是原运行控制器的历史阶段证据，不代表新 touch-first 版本已完成 Isaac 验收。

旧恢复试验成功加载状态，但接近末段桌面余量不足触发保护，未闭指/夹持/抬升。当前调整为左手路径上移1mm，名义桌面余量约2.20mm；实际试验仍在原项目运行。未使用绑定伪造抓取，1mm安全阈值不变。
