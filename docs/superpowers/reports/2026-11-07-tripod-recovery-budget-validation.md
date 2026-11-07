# 三指独立恢复、非线性指腹几何与原生 A 验证

本阶段基于 `46dc091738023503d21da980c944afbceef986d7`，工作分支仍为 `codex/dh116-lift-repair-20261106`。PR #1 保持 Draft，不合并到 `11-5`。首次原生启动在进入 A 控制前退出；第二次启动进入真实物理子步后被操作者停止，独立审计将其标为 censored/invalid evidence。A没有取得资格，不把CPU验证、A02或旧二指4/4结果称为本次三指成功。

## 原始失败与本次边界

前一阶段实际 `geometry_constrained_pilot_03` 的二指4/4、三指0/4和三指最高净抬升0.019719 mm均为历史结果。此次保留旧 Pressure、Acquisition、Constrained 和二指路径，新增显式 `--tripod-recovery-stage A` 的独立实验。该入口在 PPO/成功 BC 数据池构造之前返回，拒绝 resume 和后续阶段。没有增加 PPO 更新轮数。

3844D Visual Actor、371D Privileged Critic、24D动作、Tanh Gaussian 与默认旧 v2 路径保持。原物理、材料、关节/执行器限制、原生净30 mm、连续5 s、相对漂移0.5 mm标准不变。PhysX 接触与力矩估算只供专家控制/训练诊断，没有追加到实机 Student。

## 实现及审查修复

`tabletop_tripod_contact_recovery.py` 为每个物理 episode 的 env/hand/digit 独立维护 LOST、REACQUIRING、CONTACT、STABLE。只有原生对象过滤指腹数据可建立接触。无效数据为未知；几何接近不等于接触。20 ms恢复稳定判据只用于恢复控制，不代替5 s原生成功。

每个指腹1 mm累计接近命令路径、1 s累计 REACQUIRING 仿真时间；0.5 s无实际几何进展停止当前尝试。接触闪烁、恢复、求解失败、重试都不补充预算；耗尽后锁定直到真实物理 reset。候选/预留不收费，原控制 guard 之后、左右手 `write_data_to_sim()` 均返回后才核实实际逻辑命令及 float32 SDK targets 并提交预算。实际指腹位移、几何命令路径上界和物体运动分别记录。相同/回滚 episode ID 不能清空预算。

逐子步 `recovery_ledger` 同时保存实测 source 区域间隙、区域见证点/法向、有效性、最佳间隙、round身份/状态、实际接触/稳定时间、预留与耗尽状态，供独立重算进展与预算。它们来自实际 SDK 状态的区域计算，不是几何预测接触或实机传感反馈。非有限值保持 null 并保留原非有限值标记，不补零；此补充仅影响日志，不改变控制或预算。

独立审查复现并修复：CONTACT 后剩余的60 Hz接近动作继续执行却不记账。现在每480 Hz子步前按动作来源撤销已不适用的接近动作；相反，旧压力动作在真实接触丢失后也不能变成无预算接近。控制步原动作与子步实际动作、来源、SDK target均分别保存。原二指和压力退压不由恢复过滤器取消。

`tabletop_tripod_pressure_hold.py` 每个实际反馈子步重建右三指压力参考与非线性 FK，不复用旧 MPC 的永久 `bad_predictions` 停止状态。目标1.5/.75/.75 N、原增益、1 mm/s、0.6 mm退压与0.05 N拒接规则沿用。它输出几何运动提案，不预测压力/摩擦；必须重新审核最终实际 post-guard 命令。无法认证强制退压时终止 A，不把零动作或部分候选当作压力恢复成功。

`tabletop_tripod_nonlinear_geometry.py` 读取左右原 USD 全34个凸 hull、10个真实指腹区域和全部源层 SHA。每次绑定实测11关节 q/qd、全部 SDK link frames 和实际 base_link；保留从动关节残差。使用关节局部增量重算完整 FK、法向和 source facet∩hull∩ellipse 的同点距离/方向；审计全 rod、桌面、原支架、自碰撞和289个双手 hull 对。

`tabletop_tripod_candidate_planner.py` 保留原任务 dtype 的逐480 Hz逻辑积分，与 float32 SDK targets 分开；DOUBLE 原控制器8步逐值对照通过。双手跟踪、逐子步限位/lead/slew及整个 joint-box 的保守扫掠证书共同审核；无证书即拒绝，不用端点采样替代。命令弧长上界按每480 Hz预算检查，不能从后续子步借距离。模型响应仍为明确标记的近似，不是原生驱动/物体动力学证明。

源几何策略保留静态筛选的0.3 mm self/interhand/obstacle clearance。已有真实承载且同点几何合格的指定 pad 才可使用单独标明的0.6 mm源包络策略，数值来自原压力退压阈值；它不是旧静态0.3 mm规则，也不等于原生 manifold depth，未修改物理深度或成功判据。真实 source 非接触区域使用保守 inscribed16gon，可能拒绝实际可行接触；UNCERTIFIED 不能解读为已证明物理碰撞/不可达。

## 逐物理子步证据与门槛

`tabletop_tripod_native_evidence.py` 从同一真实 SDK 时钟读取实际全关节、body-link、base_link、物体位置/姿态/速度，与 TensorWorld 缓存严格核对。保存完整normal/friction ownership，审计原34个view的count/start/capacity；满容量、重叠、陈旧、reset或非有限读数保持未知。原 normal/friction 缓冲在共享索引被复用前分别复制。没有扩大 sensor 容量。

`tabletop_tripod_support_diagnostics.py` 独立记录每480 Hz的左右逐pad载荷/点/法向/深度、六pad连续接触、对向状态、法向/摩擦支撑分量、拒接载荷、实测掌位移、固定锚点滑移，以及显式前馈、原总限额、PhysX drive限额、模拟 applied/computed torque及各自有效率。模拟估算不称真实电机测量。

诊断审查实证发现并修复9个原型反例：初始单帧即通过 A、unknown 伪造 loss、仅一个指腹恢复冒充三个、注入历史洗成自然恢复、B 在共同20 mm位移时通过、空/非有限raw冒充完整支撑、末帧更换锚点抹去0.7 mm滑移、同 episode reset重开终止。进一步分离实际 CONTACT 与稳定性条件，深度/拒接异常不能伪造真实丢失；raw逐view/slot/指腹载荷向量与真实source bindings核对。

固定阶段用最后已观测真实子步显式 `begin_stage`，每 episode/stage 仅一次；不重置物理、接近预算、接触历史或净lift基准。A/B/C固定区间中发生的实际腕位移违例不能通过返回原位擦除。C/D的固定锚点不能在末帧重新指定。5 s使用2400个完整480 Hz步计数。

独立回归还复现了“前缀已完成的恢复被计为A阶段恢复”的假通过。现分别保存整物理episode历史与阶段内恢复：A入口只承认此前确曾接触且入口仍真实缺失的指腹，或A中实际发生的丢失；三个指腹均须在A固定腕区间重新建立实测接触并完成10子步稳定验证。前缀已恢复、单帧接触或几何预测不计入A；69项诊断回归及6项独立阶段范围回归通过，均为CPU证据。

原手部 sensor 仅过滤 Wire，**没有接通原桌面和端部支架的独立载荷读数**，因此 `passive_support_coverage=None`。接触保持不冒充物体重量承托；C/D严格支撑认证保持未知/未通过。0.01重量卸载比例是新诊断策略，0.9重量比来自既有 Teacher qualification，二者都不冒充原生成功门槛。`tabletop_tripod_support_allocation.py` 仅提供基于实际抓取点/COM/方向/容量和原电机限额的静态wrench提案；没有默认50/50。不可行提供独立残差诊断，不输出可执行假平衡。

终止前由原 TensorEnv observer 保存物理证据；硬 guard 失败有独立 failure/context 和 raw-first 持久化。中途物理 reset 有独立递增的 episode 身份，不能用控制步结束后才更新的计数替代。最终 first-episode 预算在 reset 前保存。

## 已执行 CPU 验证

以下全部为 CPU/合成 fixture/原数据重放，不能计为原生接触恢复或抓取成功：

- 原路径定向基线：168 passed，21.94 s。
- 集成定向 `integrated_cpu_01.xml`：335 passed、1 skipped，79.28 s；含原 Pressure/Acquisition/Constrained、34-hull几何、传感器与恢复组件。
- 新压力保持：16 passed，5.49 s；早期root集成/部分reset命令记账10 passed、3.37 s，最终runtime相关15项包含新子步撤销和部分reset保护，已纳入下述冻结全套。
- 独立 source 数学审查：600随机区域、1200个独立 SLSQP min/max无反例；真实左右USD40个rod案例一致；34,850个运动界检查无越界。仍为离线几何，不能称 native。
- 独立诊断/manifest/gate回归105 passed，19.29 s；SDK证据最终独立回归95 passed、1 skipped，4.96 s；真实原控制器的动作来源切换6项外部回归通过。
- 历史 `full_cpu_05_stage_a02.xml`：5883 passed、96 failed、28 errors、49 skipped，91个subtests通过；该1331文件A02快照结果现由下面的最新完整源码回归取代。其XML SHA256为 `87d83ad90af0e39146f68cda5684771c11ed990844f5dd3ff850f86d6bd2ecbe`，原始输出仍保留。
- 最新 `full_cpu_final_source.xml`：5917 passed、96 failed、28 errors、49 skipped，91个subtests通过；与基线 `full_cpu_final_05.xml` 的5786个节点逐项比较，无缺失/状态变化；共6090个普通节点，304个新增全部通过。XML SHA256为 `fc518fdbe4e29761db99cc61933d0c4ebfbc583cfc003724ff583f1116cc3440`，逐节点证明见 `full_cpu_final_source_comparison.json`；最终 `final_source_manifest` 的1334个文件测试后保持不变。96失败/28错误均为原有状态，不称全仓库全绿。
- `full_cpu_05_stage_a02.xml` 使用与A02相同的 `stage_a_02_code`、1331文件冻结快照，涵盖当时的只读场景核验与日志字段补充；对照记录 `frozen_source_unchanged_after_cpu=true`。随后新增的cooked owner-first读取修复不在这份历史全套快照中。其后定向最终CPU为52 passed、13.34 s，模块/测试SHA前缀为 `ddf703d...` / `e55cb...`；最新完整源码回归已在上一条记录给出。场景/日志定向18 passed、5.26 s；场景修复另有独立USD合成测试，总5项通过、4.93 s，场景导出前后字节相同。

定向和完整套件的源码范围明确分开；上述数字不可相加作为额外原生试验次数。原生A结果单列如下，不从合成fixture或旧运行推断。

主要定向命令：

```bash
cd /mnt/mydisk/xk/projects/dh116-tripod-recovery-20261107
USD=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$USD:src:.:vendor/DeformX" \
LD_LIBRARY_PATH="$USD/bin" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
  tests/test_tripod_contact_recovery.py tests/test_tripod_execution_guard.py \
  tests/test_tripod_raw_contact_capture.py tests/test_tripod_native_lifecycle.py \
  tests/test_tripod_recovery_runtime.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_nonlinear_geometry.py tests/test_tripod_candidate_planner.py \
  tests/test_tripod_support_diagnostics.py tests/test_tripod_support_allocation.py \
  tests/test_vision_teacher_pressure_mpc.py tests/test_vision_teacher_constrained_mpc.py \
  tests/test_vision_teacher_acquisition_mpc.py tests/test_native_sensor_batch.py \
  --junitxml=/mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z/integrated_cpu_01.xml
```

完整冻结回归命令（源码/输出均独立）：

```bash
cd /mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z/stage_a_02_code
USD=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$USD:src:.:vendor/DeformX" LD_LIBRARY_PATH="$USD/bin" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_DEBUG_TEMPROOT=/tmp OMP_NUM_THREADS=2 \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider tests \
  --continue-on-collection-errors -q --tb=short \
  --junitxml=/mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z/full_cpu_05_stage_a02.xml
```

## 原生首次启动与场景核验修复

`stage_a_01` 使用独立输出、冻结1330个源码/资源文件和单实例锁，于2026-11-07 09:57:37 UTC启动，45.117 s后进程exit2。原生初始化/物理配置前置检查完成后，只读场景核验抛出 `original static collider absent: /World/envs/env_0/Table`，尚未创建A恢复控制或产生A动作/RGB-D/阶段记录。因此此次为**A前置检查失败，0个A试验、0个完整A回合**，不能记作三指抓取0/4。

原桌面是父Xform下的真实CollisionAPI子节点，旧核验只检查父节点。修复只读遍历启用的子碰撞体并合并其世界BBox，忽略禁用碰撞体与纯视觉节点；仍与原世界边界比较，数值容差仍为2 μm。没有写USD、改变碰撞体、物理参数或原门槛。测试覆盖父Xform/子碰撞体、视觉诱饵、仅禁用碰撞体、错误物理边界、多个子体并集及clone世界坐标；3项定向与2项外部测试通过。独立审查证据 `independent_runtime_review/scene_descendant_review.json` SHA256为 `d36f6b517a7b7cd9a385320dfdedd562b7aa94ba8bf29de24633248353c92f98`。

01的冻结源码及13项受保护原件前后SHA均未变化；旧失败数据、旧launch gate和所有原模型保留。首次独立审计见 `stage_a_01_independent_audit.json`，SHA256为 `b25a0e011d1df4868f420aec6bdded83fbd515fe9b11c0825c3a1f46503f8869`。修复后使用 `stage_a_02`、`stage_a_02_code` 和新的launch gate，不覆盖01证据。

### stage_a_02 原生结果（已结束：censored/invalid）

02在独立1331文件冻结快照中进入真实480 Hz物理子步，随后由操作者SIGINT结束（controller exit2，保护原件校验保持干净）。独立审计将结果标为 `censored/invalid evidence`，既不是A通过，也不是可计数的A失败或三指0/4试验：

- 481次逻辑控制；482个物理区间；3856个原始SDK子步；原始物理证据共30,840行。
- 8个首episode均为censored，0个完整episode，0次approach submission，0预算消耗；没有Student或PPO更新。
- 杆/腕未达到固定阶段所需静止条件；实际角漂移约0.0022–0.00328 rad，高于0.001 rad门槛，因此没有进入合格的几何恢复决策，geometry decisions=0。
- env4–7的Tripod pad只出现自然接触，阶段内没有进入 `REACQUIRING`，没有固定阶段恢复、terminal或success；因此不能将自然前缀接触算作右手恢复。
- SIGINT的flush顺序造成最后8个诊断行缺失；审计没有用预测值或补零伪造reducer。最终审计记录 `audit_integrity=false`、25个问题，其中包括末尾诊断缺失和最终FK中缺少SDK joints。A不能据此声称已测试或恢复右拇指/食指/中指接触。

A02的独立重算/保护证据：reconciled audit SHA前缀 `c309bb8...`，raw tail SHA前缀 `50a6f9...`，fixed-wrist quaternion audit SHA前缀 `ad2782...`。这些前缀只用于定位服务器原件，完整SHA和原始文件仍保留在服务器运行目录；没有将不完整证据纳入成功率或训练监督。

### 其他探针与离线候选（不属于原生A）

静态 source pregrasp candidate02 的离线全审计结果SHA前缀为 `d614...`，对应审计SHA前缀为 `4a7156...`；它只证明离线候选/几何审计路径，不能替代原生物理证据。Cook probe01 在 source-root 校验处 fail-closed；probe02 被过期gate preflight拒绝。probe03仅完成原生 reset 和一次RGB-D读取，实际mesh路径的精确getter对272项全为0，collector早于最终identity建立，`complete=false`、`live shape=false`，没有控制、PPO或可用原生RGB-D证据。probe04使用owner-first getter后每个存储hull count=1，但272个hull全部未通过独立closed-convex拓扑验证，仍为`complete=false/censored`、`live attachment=false`；没有recook或几何替换，仅一次RGB-D，无控制/物理/Teacher/PPO/Student。owner-first模块/测试SHA前缀为`ddf703d...`/`e55cb...`。探针问题不回写生产物理参数；完整CPU仍需覆盖最新owner/stage源码。

## 原生隔离与后续状态

所有新实验位于 `/mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z`，服务器候选源码位于 `/mnt/mydisk/xk/projects/dh116-tripod-recovery-20261107`。正式 A 使用新的 frozen-code目录、来源清单、exclusive输出及单Isaac锁；不覆盖旧失败实验、模型或原训练。

独立A先原样执行 Pressure 前缀0..436，再冻结当时实测双腕目标；A仅允许右拇指/食指/中指有界恢复，左指目标固定。实际腕和rod尚未静止时不授权几何接近。原生全11关节增量及17链接连续帧FK差异单独记录，零增量校准自洽不称原生运动学验证。

当前原生执行入口仅开放A。B/C/D判据和静态承托分配器已实现CPU验证，但B六指承载、C连续5 s保持、D共同Lift的串行执行与恢复/暂停联动仍须在A证据合格后实现及原生验证；没有将纯诊断gate称为已验证控制器。只有A真实独立恢复通过并经审计才允许独立B；B通过才运行C、C通过才运行D。E/Student、奖励消融/Adam新对照、柔性线缆泛化均未执行，不启动长PPO、不宣称退化解决、不转Ready、不合并。重量承托测量及受限联合腕优化也仍待实际证据推进。

当前原生物体保持原始刚性圆柱代理；本阶段没有将它替换为柔性线缆动力学。即使后续该代理的A恢复或D抓取通过，也只能限定在该原生场景，不能作为柔性线缆或真实DH116硬件的验证结果。所有接触力、接触深度、摩擦和力矩仍为仿真测量/估算，没有新增已确认接通的真实硬件传感通道。
