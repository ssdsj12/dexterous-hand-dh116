## 2026-11-13 — 双手物理 Episode 链路

双手 Reset、杆快照和 480/60 Hz 交换已接到同一 CPU 验证入口。保留初始净空拒绝、原保护终止与单样本时钟身份。 Ubuntu CPU 回归 96 项通过。[报告](docs/superpowers/reports/2026-11-13-dual-hand-physical-episode.md)

## 2026-11-13 — PhysX 到 DeformX 位姿交接

PhysX 前进后，读取新指腹位姿并校验 DeformX 仍处于前一边界；跳步、换 episode 或陈旧读取会拒绝交换。已接通 CPU 双手八步回归。 Ubuntu CPU 回归 92 项通过。[报告](docs/superpowers/reports/2026-11-13-dual-hand-clock-handoff.md)

## 2026-11-13 — 双手同杆原地 Reset 事务

核对同一根杆的 Reset 身份、双手实际关节与目标读回，并保留未知力学字段。四项既有 root pose fixture 失败已独立复现。实际 PhysX Reset 和 R0 仍待验证。 Ubuntu CPU 回归 312 项通过。[报告](docs/superpowers/reports/2026-11-13-dual-hand-reset-cpu.md)

## 2026-11-13 — 右手 Pinch 独立闭合筛查

450 个右侧候选完成，唯一严格端点候选在连续 finger13 近端净空门被拒，保留失败和断点证据。 Ubuntu CPU 回归 38 项通过。[报告](docs/superpowers/reports/2026-11-13-right-pinch-closure-screen.md)

## 2026-11-13 — DeformX 逐接触对 CUDA 校准 v2

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f7-deformx-pair-capture-v2-native.md)。

## 2026-11-13 — 柔性 24D 控制的同钟几何读回

用实测 link pose 和局部 COM 接入全手源包络净空，保持原两侧几何身份；SDK double 回归通过，原生尚未验证。 Ubuntu CPU 回归 43 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-live-geometry.md)

## 2026-11-13 — DeformX 接触对校准 worker 注册路径修复

从真实失败 intent 定位证据根层级，真实 CPU worker 子进程现通过授权到构造边界，并校验 solver 状态。旧失败保留。 Ubuntu CPU 回归 89 项通过。[报告](docs/superpowers/reports/2026-11-13-deformx-pair-worker-v2.md)

## 2026-11-13 — 柔性杆 Reset 与测量状态同钟适配

首动作前可验证杆节点、几何和材料顺序，未测接触保持 NaN 和 false mask。后端和物理时钟按实际输入核对。 Ubuntu CPU 回归 64 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-rod-reset-boundary.md)

## 2026-11-13 — Teacher 原策略 owner 动作与目标来源

按实际 TensorTask owner 和动作时序核对原目标与上一接纳 24D action，共 36 项独立字段。尚未接入柔性环境。 Ubuntu CPU 回归 118 项通过。[报告](docs/superpowers/reports/2026-11-13-teacher-policy-owner-source.md)

## 2026-11-13 — 逐指腹接触对与同子步加载诊断

对账逐对力、杆节点和 mesh 力矩，报告同时加载、材料节点和桌面 SDF。滑移、外部支撑与有效对向接触维持未知。 Ubuntu CPU 回归 73 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-pair-contact-diagnostics.md)

## 2026-11-13 — 柔性双手原 USD 场景装配与几何身份

复用原双 USD、腕驱动和桌面，离线核对 34 个碰撞体、10 个软指腹、12 个腕关节和两侧 SDF。实际环境启动与后验 cooked 校验仍待验证。 Ubuntu CPU 回归 71 项通过。[报告](docs/superpowers/reports/2026-11-13-dual-hand-scene-preparation-cpu.md)

## 2026-11-13 — DeformX PPO 与 Hold 窗口的单精度时钟校验

按真实 solver 单精度累计校验，同时保留 SDK 与杆整数步严格匹配。已覆盖 2401 步 CPU 重放。 Ubuntu CPU 回归 71 项通过。[报告](docs/superpowers/reports/2026-11-13-deformx-solver-clock.md)

## 2026-11-13 — Teacher 原生腕部实例路径绑定修复

按已安装 IsaacLab 的实际根路径发现规则区分 wrist_fixed 与 base_link 所在容器。仍为 66/371 CPU 来源覆盖。 Ubuntu CPU 回归 146 项通过。[报告](docs/superpowers/reports/2026-11-13-teacher-mounted-wrist-source.md)

## 2026-11-13 — DeformX CUDA 逐接触对读回单步校准

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f7-deformx-pair-capture-native.md)。

## 2026-11-13 — PhysX 构造 frame 释放 v2 单次原生

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-physx-lifetime-frame-v2-native.md)。

## 2026-11-13 — PhysX 构造 frame 引用释放与原生复验准备

保留同一物理初始化，仅清理构造帧持有的局部引用并使用全线程暂停取证。新单次原生指纹待批准。 Ubuntu CPU 回归 135 项通过。[报告](docs/superpowers/reports/2026-11-13-physx-lifetime-frame-v2.md)

## 2026-11-13 — Teacher 371D 原腕部状态来源

按原 WXYZ、COM 速度、内禀关节角和 reset seed 接入 32 个腕部通道。CPU 中两手共 66/371，未绑定通道仍有 305 个。 Ubuntu CPU 回归 139 项通过。[报告](docs/superpowers/reports/2026-11-13-teacher-mounted-wrist-source.md)

## 2026-11-13 — 60 Hz 双手控制与 480 Hz 柔性物理交换

单环境每控制步最多 8 个物理子步，接通原动作保护、真实 SDK 路由和同杆反力时差。CPU double 验证，场景装配和正式 R0 仍待资格。 Ubuntu CPU 回归 117 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-physical-exchange.md)

## 2026-11-13 — 柔性双手腕部原桌面安全投影

复用原 2.1 mm 净空、腕速度约束、内禀旋转及撤回不安全下压目标。场景全手净空与原生执行仍待接入。 Ubuntu CPU 回归 104 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-wrist-projection.md)

## 2026-11-13 — 双手 24D 控制边界与 PhysX 目标读回

使用原关节和腕部动作语义，核验左右实例、实际目标、SDK 和杆时钟。原桌面投影接入和原生执行仍待验证。 Ubuntu CPU 回归 174 项通过。[报告](docs/superpowers/reports/2026-11-13-deformable-bimanual-step-cpu.md)

## 2026-11-13 — 双手同杆运行时与 10 指腹反作用力路由

接通双手槽位和世界坐标反作用力路由，校验真实实例身份、单步时钟和 Reset。CUDA 用例在 CPU 回归中跳过，待原生标定。 Ubuntu CPU 回归 66 项通过。[报告](docs/superpowers/reports/2026-11-13-dual-hand-rod-runtime.md)

## 2026-11-13 — PhysX GDB 全线程暂停后取证

带后台线程的 CPU 进程复现旧 GDB 命令错误。候选捕获器在所有线程暂停后保留完整栈、映射和实际退出状态，待下一原生配方接入。 Ubuntu CPU 回归 42 项通过。[报告](docs/superpowers/reports/2026-11-13-physx-gdb-stopped-context.md)

## 2026-11-13 — 三抓法 Teacher rollout 准入与 GAE 边界审计

三抓法现有证据全部拒绝准入。复用 371D 来源门与原 GAE，终止不 bootstrap，截断只使用同回合末状态。2 环境 32 控制步预算仅作审计。 Ubuntu CPU 回归 107 项通过。[报告](docs/superpowers/reports/2026-11-13-deformable-rollout-admission.md)

## 2026-11-13 — PhysX 生命周期两格原生结果

两格 app.close 返回；GDB 全线程捕获失败且退出码 1，资源保留证据已复核。[报告](docs/superpowers/reports/2026-11-13-physx-lifetime-native-contrast.md)

## 2026-11-13 — PhysX 所有权生命周期 explicit 原生对照

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-physx-lifetime-explicit-native.md)。

## 2026-11-13 — PhysX 所有权生命周期 registry 原生对照

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-physx-lifetime-registry-native.md)。

## 2026-11-13 — PhysX Tensor View 两种清理路径对照准备

两种资源清理路径分别冻结，捕获 GDB 调用栈、寄存器和动态库映射，外层退出独立复核。原生对照待批准。 Ubuntu CPU 回归 113 项通过。[报告](docs/superpowers/reports/2026-11-13-physx-lifetime-probe-preparation.md)

## 2026-11-13 — 柔性 R0 控制步与提前终止结算

每个完整 8 子步窗口或已知提前终止只结算一次，未知来源保留空值。正式 R0 系数和训练奖励仍待原生证据。 Ubuntu CPU 回归 78 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-r0-control-settlement.md)

## 2026-11-13 — Tripod 与 Power 独立闭合可达性检查

Tripod 左右静态候选桌面净空各 2 mm，闭合仍有近端碰撞。Power 筛查保留未通过的必要可达性证据，未生成原生配方。 Ubuntu CPU 回归 44 项通过。[报告](docs/superpowers/reports/2026-11-13-tripod-power-geometry-preparation.md)

## 2026-11-13 — 柔性 Teacher 前动作 371D 来源审计

保留原 371D 合同，新增按实测名称绑定的双手 34 个通道适配。旧单手 Pinch 账本未取得完整的身份与关节顺序，原生可纳入 0/371。 Ubuntu CPU 回归 67 项通过。[报告](docs/superpowers/reports/2026-11-13-teacher-preaction-readback.md)

## 2026-11-13 — DeformX 逐接触对读回

独立 SM89 扩展已编译，62 项 CPU 回归通过；一次 CUDA 标定待批准。[报告](docs/superpowers/reports/2026-11-13-deformx-pair-readback.md)

## 2026-11-13 — Power 独立档案的柔性桌面与杆筛选

左右各16个历史握式种子通过初始避碰，但最差指腹到杆仍远，暂无柔性 Power 可达轨迹。 Ubuntu CPU 回归 20 项通过。[报告](docs/superpowers/reports/2026-11-13-flexible-power-initial-pose-preflight.md)

## 2026-11-13 — Pinch cooked hull 初态修正与 Tripod 独立筛选

修正左手 Pinch 独立候选并验证连续桌面净空，Tripod 自身旧种子未通过。原生配方继续等待接触对和生命周期证据。 Ubuntu CPU 回归 169 项通过。[报告](docs/superpowers/reports/2026-11-13-pinch-initial-geometry-audit.md)

## 2026-11-13 — PhysX Tensor View 退出栈与所有权审计

GDB 定位到 SimulationView 释放路径，区分管理器 view 清空、Python 所有者、CUDA 同步和外层退出；新增只读引用诊断。 Ubuntu CPU 回归 74 项通过。[报告](docs/superpowers/reports/2026-11-13-physx-tensor-lifetime-audit.md)

## 2026-11-13 — WireSeg depth18 精确配对与双任务恢复预检

完整分片达到 10/148，depth18 新增 360 对；分割与深度断点各 128 次历史更新，实际新增优化更新为 0。 Ubuntu CPU 回归 84 项通过。[报告](docs/superpowers/reports/2026-11-13-wireseg-paired-download-separate-resume-preflight.md)

## 2026-11-13 — 柔性杆力反馈来源与聚合证据校验

锁定 DeformX 的逐节点合力、逐 mesh 净力矩现可独立审计；contact pair、外部支撑和材料滑移保持未资格化。 [报告](docs/superpowers/reports/2026-11-13-deformable-force-evidence.md)

## 2026-11-12 — C5 与自由杆 Pinch 单次诊断

两次原生均未通过。C5 缺少目标 holder，Pinch 在第 3 SDK 步触发 mimic 保护。GDB 输出路径修复通过 47 项 CPU 回归。[结果与恢复点](docs/superpowers/reports/2026-11-12-c5-and-pinch-native-diagnostic.md)

## 2026-11-12 — Free rod left Pinch contact candidate only

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-free-pinch-left-candidate-native.md)。

## 2026-11-12 — Camera C5 telemetry holder lifecycle

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c5.md)。

## 2026-11-12 — 长时续任务第二批恢复点

六个独立源码工作包已完成发布，最终隔离回归458项通过、0失败/跳过。自由杆候选与C5均未原生执行；深度精度仍不合格，Teacher/Student未越过准入门槛。[完整状态和恢复计划](docs/superpowers/reports/2026-11-12-continuation-batch2-progress-and-recovery.md)。

## 2026-11-12 — WireSeg 完整固定清单监督域与误差复核

完成原16 train/128 heldout、288个视图的只读CPU复核，精确复用16个已完成视图。25.92秒/exit0、模型权重不变；128个持出中心视图全部差于训练常数基线，线缆ROI MAE 0.141680m 对比0.087620m。监督域差异与误差统计加权已区分，未假定唯一因果；无法重建的pooled P95保持NA。 发布前独立审查补齐helper/聚合实现/脚本SHA缓存绑定；原结果保持不变，另存事后身份核验凭据。 Ubuntu CPU 回归 77 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-full-manifest-loss-domains.md)

## 2026-11-12 — 自由杆单手 Pinch 实际运行入口与独立退出验收

新增真实自由杆、重力、有限桌面、五个独立原 pad SDF 和反作用力写回入口。240 SDK tick 候选包含全部初始化；异常持有链、worker 重入和独立多pad退出审计均有故障回归。材料为显式 P0 系数候选，非实测；完整371D/R0/抓取资格仍未成立，未启动原生。 首次集成回归因裸Python未加载pxr在收集阶段退出2，完整失败日志保留；改由现有scripts/python.sh加载USD路径后重新验证。 Ubuntu CPU 回归 156 项通过。[报告](docs/superpowers/reports/2026-11-12-free-pinch-physical-runtime.md)

## 2026-11-12 — WireSeg 旧深度模型只读误差归因

对原深度/分割模型做 4 train＋4 heldout 的只读 CPU 推理，32 条记录，外层 7.31 秒/exit 0，模型权重不变。深度模型仍差于训练常数基线；监督域差与预测范围压缩得到小 cohort 证据，未认定唯一因果。先继续 loss/标签域审计，不盲目追加训练。 Ubuntu CPU 回归 67 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-readonly-depth-error-attribution.md)

## 2026-11-12 — C5 原生 worker 保留独立采样入口

修复 C5 父进程设置丢失于后台 worker 的执行缺口；真实 CPU 子进程回归确认 worker 进入相同专用入口、保留 GDB 命令和证据路径。旧 94ffe… 指纹与未执行审批作废；没有消耗原生次数。 Ubuntu CPU 回归 45 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-c5-ownership-preflight.md)

## 2026-11-12 — WireSeg 旧断点独立诊断恢复与第九个官方数据块

旧断点可在独立诊断格式中恢复 model/Adam/Torch RNG 与真实 sampler，并支持后续完整当前 RNG 链式续点；未知历史信息仍为 unknown，strict v2 不接受。新增 360 个官方精确 RGB-depth 配对，完整块 9/148。本轮真实训练 0 更新、原生 0 次。 Ubuntu CPU 回归 62 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-legacy-diagnostic-import.md)

## 2026-11-12 — C5 telemetry hook 所有权对照与正常退出采样

新增单变量 C5：在 C4 配置下提前释放已存在的 telemetry hook pair，并准备自然 finalize/崩溃完整栈与 maps。根因假设未证实，原生尚未运行；旧 C0–C4 和调度源码不变。 Ubuntu CPU 回归 44 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-c5-ownership-preflight.md)

## 2026-11-12 — P0-P6 bounded continuation evidence and recovery point

本批四次原生调用：左右拇指预算合规、C3 自然退出，C4 SIGSEGV 保留栈和动态库映射且不重试。共 480 SDK 步/476 实测行/5712 杆子步，真实 PPO 与完整抓取回合均为 0。无相机 Teacher 结构化 CPU 更新/恢复已验证，WireSeg 8/148 入库；自由杆抓取、正式相机采集和真实训练仍按各自证据门控。原生批准额度已用完，已保存恢复点。 Ubuntu CPU 回归 357 项通过。[报告](docs/superpowers/reports/2026-11-12-continuation-p0-p6-batch-closeout.md)

## 2026-11-12 — P5 frozen-policy regression follows preserved float32 contract

首次合并回归 356 通过/1 失败已保留。CPU 同策略 2/4 批量复算的 logprob 舍入差为1.907e-6，新测试1e-6与原2e-5冻结检查不一致；改为调用原校验并验证策略参数和历史似然篡改均被拒绝。PPO源码、阈值和物理标准没有修改。 Ubuntu CPU 回归 9 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-ppo-frozen-likelihood-test.md)

## 2026-11-12 — P5 isolated structured PPO runtime and checkpoint CPU verification

无相机柔性 Teacher 使用独立 CPU 结构化 rollout/Checkpoint 入口，直接继承旧 PPO update、GAE、KL 和 Adam 回滚，旧源文件 SHA 不变。合成数据完成更新及恢复等价测试；不代表真实 Teacher 训练或抓取成功。Pinch/Tripod/Power 原生门槛保持关闭，当前相机失败不阻断这一准备工作。 Ubuntu CPU 回归 170 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-privileged-ppo-runtime-cpu.md)

## 2026-11-12 — P3 current C3 C4 native exit and stack audit

授权 C3/C4 各运行一次：C3 26.84 秒、outer/GDB 自然 0；C4 54.81 秒、outer255/SIGSEGV11。C4 默认窗口确实关闭，当前首故障为 _extensions+0x1e445，不能套用 C1 的 _appwindow 首帧。两格 0 物理步/0 PPO，不重试；渲染路径仍未资格化。 Ubuntu CPU 回归 30 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-c3-c4-native-stack-audit.md)

## 2026-11-12 — P4 measured Hold duration and immutable slip references

进一步关闭 CPU 成功候选边界：Hold 需真实时间跨度满 5 秒且连续 2400 间隔，4.999760 秒反例拒绝；左右材料与切向参考 UID/来源冻结至 Reset，缺测不延续 Hold。原生资格仍待实际自由杆测量。 Ubuntu CPU 回归 51 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-hold-clock-and-slip-reference-review-fix.md)

## 2026-11-12 — Camera C4 isolated lifecycle

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c4.md)。

## 2026-11-12 — Camera C3 isolated lifecycle

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c3.md)。

## 2026-11-12 — P5 independent Pinch Tripod Power privileged rod models

三种抓法使用独立模型身份与 Checkpoint namespace，原 371D common 加独立 masked 材料节点分支，复用 24D Tanh Gaussian；原 PPO/CLI 未改。尚未接原生或训练，不能称为 Teacher 成功；相机和 MPC BC 均不是永久前置。 Ubuntu CPU 回归 153 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-privileged-teacher-model-cpu.md)

## 2026-11-12 — P4 retain prohibited lift assistance until episode reset

交叉审查复现并修复 4 类误判：已知支撑违规不能被其他 NA 掩盖；外力、附着或非法外部支撑一旦出现，本 episode 不得取得 Lift/Hold 成功候选。正常接近阶段桌面接触不锁存，Reset 清理记录。原生与训练资格仍关闭。 Ubuntu CPU 回归 36 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-grasp-support-review-fix.md)

## 2026-11-12 — P2 WireSeg exact resume and verified paired subset

新增 strict v2 断点，校验模型/Adam/RNG/含跳过样本的 sampler/global step/数据与配置 SHA，旧断点原样保留并拒绝伪严格恢复。新增 2 个完整官方分片，360 个精确单目 RGB-depth 配对；缓存 8/148，未启动真实训练。 Ubuntu CPU 回归 30 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-strict-checkpoint-resume.md)

## 2026-11-12 — P3 AppWindow ownership and stereo raw-archive verification

旧 C0/C1/C2 栈定位到 IAppWindow Python 包装对象释放的虚调用，具体持有者仍待原生对照。新增 C4 单变量配方，并修复采集换轴与原始归档重核验；C3/C4 尚未授权或运行，相机资格仍关闭。 Ubuntu CPU 回归 83 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-appwindow-ownership-and-capture-audit.md)

## 2026-11-12 — P1/P4 free-rod task, reset and R0 source contract

新增自由杆独立任务合同，复用原 DH116/USD/24D/371D；逐物理步 Reset、材料锚滑移、连续 Hold 和十项 R0 统计可 CPU 验证。原生未知参数与奖励系数保留 NA，training_reward 关闭，未启动抓取或训练。 Ubuntu CPU 回归 172 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-dual-hand-task-r0-cpu.md)

## 2026-11-12 — 右拇指与柔性杆原生接触

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-right-contact-native.md)。

## 2026-11-12 — 左拇指与柔性杆原生接触

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-left-contact-native.md)。

## 2026-11-12 — P0 corrected thumb clock and contact proof

修正配方显式计入初始化与测量，独立重读 SDK/rod 时钟、接触节点和关闭记录；每侧 240 步、串行最多 480 步，尚未执行新的原生复验。历史超预算证据保留，Pinch/Tripod/Power 不获资格。 Ubuntu CPU 回归 78 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-thumb-independent-budget-proof.md)

## 2026-11-12 — 本批原生结果、127项集成回归及预算恢复点

127项最终集成CPU回归通过；原生左右接触与自然退出证据保留但484/480预算不合规，C0/C1/C2原生均失败。新预算修复仅CPU通过，未再执行；恢复点和剩余门槛已保存。 Ubuntu CPU 回归 127 项通过。[报告](docs/superpowers/reports/2026-11-12-eb31c79-batch-closeout.md)

## 2026-11-12 — 拇指初始化及收尾总物理步预算修复

修复总物理步漏计：240总额=2初始化+238测量，并在注销时钟前冻结物理推进。原484步失败预算证据保留，新指纹只通过CPU、待新冻结审批，未原生重跑。 Ubuntu CPU 回归 68 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-thumb-total-physics-budget.md)

## 2026-11-12 — Teacher与Hold预算显式计入初始化阶段

保留v1计数兼容，新增v2独立配置；Teacher 512行与评估2880行仅为测量阶段，setup及总额未验证时保留null并禁止scope晋级。5秒Hold仍至少2400物理步，所有草案均需独立批准。 Ubuntu CPU 回归 18 项通过。[报告](docs/superpowers/reports/2026-11-12-deformable-task-budget-v2.md)

## 2026-11-12 — 左右拇指接触账本与预算偏差复核

左右接触及自然exit0证据有效，但每侧另有2初始化PhysX步，总484超批准480共4步；已撤回预算合规资格并保留原回执。未重跑、未扩预算，不能算抓取或Teacher成功。 Ubuntu CPU 回归 59 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-thumb-v2-paired-native.md)

## 2026-11-12 — 相机C0到C2当前故障栈与原生对照

三次独立相机生命周期对照均在app.close返回后SIGSEGV，记录同次C++栈与动态库映射；审计完成但自然退出和相机资格未通过。未重复失败版本，C3未运行。 Ubuntu CPU 回归 19 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-c0-c2-current-native.md)

## 2026-11-12 — WireSeg第三个RGB分片完整核验

完整核验6/148 blobs；新增800条data_center只属test家族，精确配对深度0条，未训练或授予视觉资格。 Ubuntu CPU 回归 18 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-sixth-blob-verified.md)

## 2026-11-12 — 同渲染experience默认stage对照

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c2.md)。

## 2026-11-12 — 独立无stage渲染生命周期

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c1.md)。

## 2026-11-12 — 官方最小SimulationApp生命周期

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-c0.md)。

## 2026-11-12 — 右拇指与柔性杆原生接触

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-right-contact-native.md)。

## 2026-11-12 — 左拇指与柔性杆原生接触

原生，状态 `passed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f4-left-contact-native.md)。

## 2026-11-12 — 原生成功结果恢复与发布二次核验

恢复与发布均重新核对原始outer_exit、共享锁、日志及物理产物；外层139、ledger篡改、发布中断后篡改均拒绝，保留失败及已消耗指纹。 Ubuntu CPU 回归 57 项通过。[报告](docs/superpowers/reports/2026-11-12-native-receipt-binding.md)

## 2026-11-12 — 标定与Teacher回合预算边界

独立校验控制步、480Hz物理步、环境数、ledger行数与墙钟。2环境32控制步需512行，5秒Hold至少2400步/环境；零重力双端固定只能标定。示例未接执行、不扩大本轮480步授权。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-deformable-task-budget-boundary.md)

## 2026-11-12 — 相机官方最小App与stage单变量对照

新增独立C0最小App、C1无stage渲染、C2单变量默认stage对照；均同一GDB保留现场栈和库映射。C3仅备选配置，本轮不启动。无相机物理不依赖此链。 Ubuntu CPU 回归 66 项通过。[报告](docs/superpowers/reports/2026-11-12-camera-c0-c3-lifecycle-controls.md)

## 2026-11-12 — 左右拇指独立身份配方与单次取证

右侧锁定独立USD、SDF、link、slot和canonical action7；左右使用相同GDB取证，保持v2物理与释放逻辑。本轮预算左240后右240，两者均0update，右必须通过实际左回执资格门控。 Ubuntu CPU 回归 56 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-thumb-right-lifecycle-preflight.md)

## 2026-11-12 — 原生前置回执绑定与同次崩溃栈

右侧命令从真实已验证左侧回执解析路径和SHA，恢复复用前置；真实GDB对照保留外层255和inferior SIGSEGV，逐次捕获栈与映射，不改成成功。 Ubuntu CPU 回归 52 项通过。[报告](docs/superpowers/reports/2026-11-12-native-receipt-binding.md)

## 2026-11-12 — WireSeg 固定16样本分割与深度诊断

复用verified subset，独立完成128次分割和128次深度诊断；保留空mask与无深度失败证据。分别报告全视场、中央裁剪及线缆ROI，深度仍逊于训练常数基线，未授予Stereo/策略资格。 Ubuntu CPU 回归 25 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-fixed16-overfit-diagnostic.md)

## 2026-11-12 — 隔离 CPU 测试租约与原生执行

CPU fixture 改用私有锁。组合 OpenUSD CPU 回归 85 项通过，两次 V0 原生 SIGSEGV 继续归档。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-dh116-continuous-regression.md)

## 2026-11-12 — 空渲染 Kit 生命周期

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-v0.md)。

## 2026-11-12 — 相机退出隔离 SDK 默认场景

V0 首次运行仍由 SDK 隐式创建场景。隔离 create_new_stage 变量并读取实际 stage 状态。 Ubuntu CPU 回归 49 项通过。[报告](docs/superpowers/reports/2026-11-12-render-no-sdk-default-stage.md)

## 2026-11-12 — WireSeg 配对深度持出评估

新深度分片通过完整 SHA 校验。128 张独立持出图报告配对深度误差。 Ubuntu CPU 回归 18 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-paired-heldout-depth.md)

## 2026-11-12 — 左拇指资源释放变体与有界配方

保留首轮退出失败；原生复验只改变对象释放顺序，右侧入口核验左侧外层回执和账本。 Ubuntu CPU 回归 47 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-thumb-scheduler-preflight.md)

## 2026-11-12 — WireSeg 原始 RGB 子集续训与独立验证

两份原始 RGB LFS 复核通过。seg-only 续到 160 步，另完成 16 步精确配对深度诊断。 Ubuntu CPU 回归 17 项通过。[报告](docs/superpowers/reports/2026-11-12-wireseg-verified-rgb-subset.md)

## 2026-11-12 — 空渲染 Kit 生命周期

原生，状态 `failed`。详细结果见[报告](docs/superpowers/reports/dh116-managed-flex.f6-camera-v0.md)。

## 2026-11-12 — 任务相机 V0–V3 有界原生配方

四格按实际外层退出递进，V3 同一渲染读取全部模态。 Ubuntu CPU 回归 49 项通过。[报告](docs/superpowers/reports/2026-11-12-bounded-camera-recipes.md)

## 2026-11-12 — 带授权、预算和证据的原生调度

原生执行依赖可核验证据、批次授权、单实例资源锁和外层退出，恢复时保留既有回执。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-native-bounded-scheduler.md)

## 2026-11-12 — 任务相机单次渲染与自然退出诊断入口

同一渲染读回双侧模态，验证物理时钟与深度语义，保留独立退出判据。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-render-lifecycle-isolation-c.md)

## 2026-11-12 — 真实 DH116 左拇指接触账本与关闭诊断

左拇指 240 步真实物理账本记录 68 个接触步。外层退出 139，原生资格待修复，右侧保持待验证。 Ubuntu CPU 回归 7 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-left-thumb-contact-native.md)

## 2026-11-12 — 拆分柔性物理、双目采集与逐抓法 Teacher 资格

无相机 Teacher 分抓法依赖物理资格，双目采集和 WireSeg 单独推进。 Ubuntu CPU 回归 30 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-physics-stereo-dependencies.md)

## 2026-11-12 — 原 F3 刚体命令的集成原生复验

集成分支原命令 240 条物理样本、外层自然退出 0，账本与 E/E2 精确一致。 Ubuntu CPU 回归 6 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-integrated-rigid-entry.md)

## 2026-11-12 — Drop 与 hang 两种原生渲染的精简配置对照

Drop/hang 各有一帧完整原生 RGB、径向深度、Native ID 和 composed USD，精简配置下两轮仍外层 139。 Ubuntu CPU 回归 15 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-headless-experience-f6.md)

## 2026-11-12 — 将原生渲染与实际凸包 SDF 纳入柔性采集依赖

原生单指和同杆路径依赖实际凸包 SDF，F7 同步采集依赖 F6 原生关闭验收。 Ubuntu CPU 回归 29 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-f6-gate-wiring.md)

## 2026-11-12 — 10 个原生凸包 GPU SDF 烘焙与显式注册

10 个实际凸包使用 1 mm 体素烘焙并在同一 RopeBatch 注册；未推进物理或验证接触力。 Ubuntu CPU 回归 17 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-pad-sdf-registration.md)

## 2026-11-12 — F6 渲染自然退出崩溃 C++ 回溯

冻结 drop 入口的 GDB 首次 SIGSEGV 栈指向 AppWindow Python 绑定与 Carb，独立于 F3 PhysX Tensor。 Ubuntu CPU 回归 3 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-render-appwindow-backtrace.md)

## 2026-11-12 — 第二个 WireSeg 深度分片和实测速率

Ubuntu 完整 blob 现为 2/148。第二个 22.97 MB 官方分片在 Windows 用时 224.78 秒，正式预训练继续待完整数据。 Ubuntu CPU 回归 7 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-proxy-transport.md)

## 2026-11-12 — 实际 PhysX CPU/GPU 凸包逐指配对

两次独立原生退出 0，全部 10 个实际指腹有向多边形一致、顶点最大差值 0 m。 Ubuntu CPU 回归 15 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-device-parity.md)

## 2026-11-12 — F6 drop 原生渲染与关闭诊断

Drop 已产出 RGB、径向深度、Native ID 和 composed USD，外层退出仍为 139，hang 待实际验证。 Ubuntu CPU 回归 14 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-render-smoke-f6.md)

## 2026-11-12 — 原生 PhysX 读取 10 个 DH116 指腹凸包

实际 PhysX CPU 读回全部 10 个指腹的 cooked convex hull，外层自然退出 0。GPU 烘焙一致性仍待验证。 原生入口的生命周期调用与 F3 新入口分离，修复集成导入回归。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-native.md)

## 2026-11-12 — WireSeg 分块续传与首个完整深度分片

Windows 续传 6 块，Ubuntu 完整校验 22,541,803 字节官方分片并读取 360 个深度数组，未增加训练更新。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-resumable-relay.md)

## 2026-11-12 — F3 DeformX 与 PhysX 完整桥接自然关闭

A–E 均自然退出 0，两轮独立完整桥接各 240 步且与旧物理账本逐字节一致。原关闭路径的单一触发项尚未隔离。 Ubuntu CPU 回归 43 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-teardown-isolation.md)

## 2026-11-12 — 官方 drop/hang DomeLight 派生修复

四个派生 USD 只更新 DomeLight 纹理。76 个 USD 复审无未解析依赖，原始 109 个文件哈希一致。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-assets-f6.md)

## 2026-11-12 — 实际指腹烘焙设置和 PhysX 凸包读回入口

独立核对 10 个实际 USD 指腹的 cooking schema、坐标和离线凸包估计，确认已安装 PhysX 查询入口。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-audit.md)

## 2026-11-12 — 单 PhysX 刚体双向传力及关闭失败诊断

240 个原生步核对单刚体力和力矩交换。物理证据可复现，PhysX 插件在 Python 终结时报 SIGSEGV，F3 自然退出仍 blocked。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-rigid-bridge-native.md)

## 2026-11-12 — 将默认开发目标切换为 DeformX 柔性线缆

默认调度覆盖柔性 F0–F10 和离线依赖。原 solid_cylinder 仍可独立回归。 Ubuntu CPU 回归 28 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-execution-target.md)

## 2026-11-12 — DH116 实际指腹碰撞几何预检

从实际 USD 解析双手 10 个闭合指腹碰撞网格，复现上游跳过 CollisionAPI 所在 Xform 的问题。PhysX cooked hull 和动态加载仍待验证。 Ubuntu CPU 回归 8 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-dh116-colliders-f4.md)

## 2026-11-12 — 官方 DeformX drop/hang 资产和轨迹审计

Ubuntu 审计 109 个原始文件、72 个线缆 USD、4 个场景与 6 份轨迹；原生渲染待独立验证。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-assets-f6.md)

## 2026-11-12 — Standalone CUDA 静态、动力学与接触验证

RTX 4090 独立求解器 36 项通过，保留 1 项上游高刚度限制 xfail。静态平台读回双侧力和力矩。 Ubuntu CPU 回归 12 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f2-standalone-gpu.md)

## 2026-11-12 — 隔离 CUDA 12.8 与 cp310/SM89 扩展

固定 cp310 和 CUDA 12.8.61 两次构建成功，实际产物均有 SM89 设备代码。 Ubuntu CPU 回归 18 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f1-cuda-build.md)

## 2026-11-12 — DeformX 主仓库与递归源码锁定

柔性主线锁定上游主仓库与 CPU/CUDA 两个精确子模块，原刚性环境作为独立回归。 Ubuntu CPU 回归 9 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f0-deformx-source-lock.md)

## 2026-11-12 — WireSeg-36K 数据接入与单目诊断

柔性主线 F5 的数据读取及共享视觉骨干诊断已验证。现有证据为 4 张样本、2 对深度与 1 次 CPU 分割更新。 Ubuntu CPU 回归 33 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-ingestion.md)

## 2026-11-12 — 握式局部可达性有限差分复查

原碰撞验证后两手最大指腹间距改善至左 1.899、右 0.950 mm。12 个候选均未通过完整握式筛选。 Ubuntu CPU 回归 219 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-local-fd.md)

## 2026-11-12 — 原生 T5 碰撞实例与 Wire SDK 来源核验

核验 76 个启用碰撞体的身份与 256 条接触记录。该 Python SDK 写入路径未观测到调用，完整回合力学资格仍待验证。 Ubuntu CPU 回归 99 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-proxy-sdk-native-results.md)

## 2026-11-12 — 二指三指独立首终止原生配对

二指退出后继续观测三指。相同实测冷初态下，冻结 Teacher 的 sampled 和 mean 均出现右 finger12 合法目标下位置越限。 Ubuntu CPU 回归 46 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-per-slot-tripod-native.md)

## 2026-11-12 — 握式多腕姿原几何补充搜索

原有边界下每手 16 个不同起点，最接近的两手最大指腹间距仍高于原筛选线，握式保持待有效初态。 Ubuntu CPU 回归 219 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-diverse-wrist-search.md)

## 2026-11-12 — 握式 Teacher 整手包握几何预检

握式按双手各五指包握预检，8 个历史候选和有界 CPU 搜索未得到可用初态。 Ubuntu CPU 回归 234 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-geometry-preflight.md)

## 2026-11-12 — T5 原生 PPO Teacher 的对象侧接触验证

实际 R0 update-1 在 T5 的短窗完成 128 个物理步，双侧力闭合，端支架均持续承载。 Ubuntu CPU 回归 19 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-wire-tensor-native-results.md)

## 2026-11-12 — 冻结 Teacher 同初态原生 sampled／mean 复验

修复原生合同编码与回执。四次运行均退出 0，采样和 mean 均复现合法目标下的同一右 finger12 越限。 Ubuntu CPU 回归 56 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-native-v04.md)

## 2026-11-12 — 原 GPU 任务 Wire 对象侧接触与支撑读回

原 GPU 路径完成 128 个连续物理步，双侧接触力互相抵消。短窗的两个端支架持续承载，完整回合支撑资格仍未通过。 Ubuntu CPU 回归 80 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wire-tensor-native-results.md)

## 2026-11-12 — 原 T5 合同的特权 PPO Teacher 原生单更新

2×32 原生完成 1 次 R0 PPO 更新，Checkpoint 与 Adam 读回一致，完整成功回合 0。 Ubuntu CPU 回归 65 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-native-r0-update1.md)

## 2026-11-12 — 记录 GPU 任务早期接触读回失败并恢复默认探针

早期启用报告读回导致 GPU reset 索引与缓冲区设备不一致。保留失败证据并恢复原探针默认。 Ubuntu CPU 回归 74 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-task-contact-early-readback-failure.md)

## 2026-11-12 — 原 T5 MPC 八个首回合原生采集

原生二指 4/4、三指 0/4，10,559 条状态／动作配对，全部首回合已分区。正式 BC 准入为 0。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-t5-first-episode-native-evidence.md)

## 2026-11-12 — MPC 成功轨迹进入 Teacher 的原生证据预检

只读入口实测读取三类原生日志，分开记录任务成功、正式证据与专家装配资格。 Ubuntu CPU 回归 53 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-teacher-provenance-preflight.md)

## 2026-11-12 — MPC 内部控制状态与 reset 初始化来源

新增隔离的同钟内部状态来源旁路。真实 T5 输入预检通过，实际字段资格待新的原生采集。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-internal-state-source-preflight.md)

## 2026-11-12 — MPC 371D 原生同步账本持久化

真实旧 64 行回放逐位恢复 371D、24D 和掩码。新版可选账本供未来执行，现有运行沿用冻结源码。 Ubuntu CPU 回归 58 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-durable-371d-transition-journal.md)

## 2026-11-12 — 将 Teacher R0 绑定到原 T5 原生物理合同

完整 T5 的实际张量合同与历史成功 T5 相同。新增只读派生与原生回调严格身份检查。 Ubuntu CPU 回归 85 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-native-contract-binding.md)

## 2026-11-12 — MPC 完整上一动作窗口的腕部字段校验

62 条真实非 reset 行的 4 个腕部窗口字段逐值吻合，reset 行保留未测窗口。 Ubuntu CPU 回归 50 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-native-observation-value-proof.md)

## 2026-11-12 — 冻结 Teacher mean 配对的 sampled 完成证据

mean 启动前校验 sampled 回执、原生账本和动作身份。旧 Teacher Checkpoint 与历史 T5 场景仍不可互换。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-pair-runner.md)

## 2026-11-12 — MPC 371D 同钟数值与来源校验

旧短程原生的 64 行均实证 204/371 个字段，新增独立旁路，保留 167 个未资格字段。 Ubuntu CPU 回归 31 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-native-observation-value-proof.md)

## 2026-11-12 — Teacher 原生回调产物持久化

训练、采集和冻结 Teacher 评估在应用关闭前写入并同步回执。外层退出仍以独立监督记录为准。 Ubuntu CPU 回归 82 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-callback-durable-completion.md)

## 2026-11-12 — Teacher 原生成功专家数据装配器

装配器逐行核验完整 371D、实际 issued 24D 和原生证据。当前未产生合格 BC 专家包。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-expert-assembly.md)

## 2026-11-12 — 现有刚性 Teacher 任务逐接触读回探针

晚于首次 reset 的探针实测记录 256 个物理步，结构化报告仍为空。新版启用时点前移，待单实例原生复验。 Ubuntu CPU 回归 71 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-task-contact-probe.md)

## 2026-11-12 — Teacher Actor-only BC、R0 与冻结验收入口

新增 SHA 绑定的 Actor-only BC、R0 恢复与冻结验收预检，短程 MPC 归档通过拒绝测试。 Ubuntu CPU 回归 75 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-bc-r0-frozen-chain.md)

## 2026-11-12 — 冻结 Teacher sampled／mean 同初态配对入口

真实 update-0 与同一双抓法初态通过 CPU 预检。旧 MPC reset family 和 native readback 不匹配，三路资格仍待验证。 Ubuntu CPU 回归 53 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-pair-runner.md)

## 2026-11-12 — 原 T5 MPC 完整首回合采集预检

真实 T5 参数及 10 个来源 SHA 预检通过，归档按完整成功、失败与未结束回合分流。完整原生预算待独立批准。 Ubuntu CPU 回归 37 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-t5-first-episode-capture-preflight.md)

## 2026-11-12 — PhysX GPU 接触报告读回与左右拇指标定

原生左右隔离标定分别读出 37、43 个可识别接触 pair headers。任务逐指、外部承托及附着资格仍需单独验证。 Ubuntu CPU 回归 67 项通过。[报告](docs/superpowers/reports/2026-11-12-physx-gpu-contact-report-readback.md)

## 2026-11-12 — Cold8 状态标签与独立深度持出诊断

两种抓法各有 12 条 Center XYZ 和 12 条线速度监督；独立持出证据仍不足以授予动态泛化与双目精度资格。 Ubuntu CPU 回归 55 项通过。[报告](docs/superpowers/reports/2026-11-12-stereo-cold8-state-depth-diagnostic.md)

## 2026-11-12 — 右手 finger12 同初态三路配对预检

旧 Teacher 与新 Cold8 的实测初态交集为 0。预检记录了 MPC 近邻状态、测量缺口与三路配对预算。 Ubuntu CPU 回归 41 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-native-pairing-preflight.md)

## 2026-11-12 — 原 MPC 371D／24D 短程原生同步诊断

修复同钟记录器的 dtype 错误。原 MPC 复验取得 64 条状态／动作记录和 512 条环境级物理样本，完整成功回合为 0。 Ubuntu CPU 回归 30 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-sync-v02-native-evidence.md)

## 2026-11-12 — 8 初态双目同步采集

二指、三指各完成 4 个独立物理初态，合计 32 条同钟双目与状态动作候选样本。正式训练资格待独立验证。 Ubuntu CPU 回归 63 项通过。[报告](docs/superpowers/reports/2026-11-12-stereo-cold8-native-capture.md)

## 2026-11-12 — PhysX v3 原生自然退出与全新实例重启

原生两轮均完整记录 240 个 480 Hz 物理步，并自然退出码 0。该结果仅授予关闭与重启资格。 Ubuntu CPU 回归 65 项通过。[报告](docs/superpowers/reports/2026-11-12-physx-v3-native-shutdown-pass.md)

## 2026-11-12 — 复用双目深度辅助头的有界监督优化

71 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-depth-auxiliary-optimizer-diagnostic.md)

## 2026-11-12 — 双目分割按已验证的 Head 能力构建候选

68 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-segmentation-mask-policy.md)

## 2026-11-12 — 现有 solid_cylinder 同钟 SDK 线速度监督

72 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-rigid-sdk-state-supervision.md)

## 2026-11-12 — Teacher 右手 finger12 下限事件窗口

29 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-right-finger12-event-windows.md)

## 2026-11-12 — 现有刚性任务的密集和稀疏双目指标

56 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-reconstruction-metric-evidence.md)

## 2026-11-12 — 现有刚性任务完整闭环（复用原环境，不重新建模）

默认调度目标选择现有 `solid_cylinder` 闭环，柔性后端延期。27 项 Ubuntu CPU 回归通过。[报告](docs/superpowers/reports/2026-11-12-existing-solid-cylinder-execution-scope.md)

## 2026-11-12 — PhysX 标定句柄释放与退出崩溃诊断

v2 首轮 240 步完整，close 返回后子进程 SIGSEGV。隔离入口补齐句柄释放和崩溃诊断，65 项 CPU 回归通过。[软件报告](docs/superpowers/reports/2026-11-12-physx-sampling-handle-lifetime.md) · [原生证据](docs/superpowers/reports/2026-11-12-physx-v2-native-exit-crash.md)

## 2026-11-12 — PhysX 标定关闭模式与子进程读回

首轮 240 步完整，关闭返回标记缺失。隔离入口修复并通过 61 项 CPU 回归，原生资格待复验。[报告](docs/superpowers/reports/2026-11-12-physx-non-fast-shutdown-recovery.md)

## 2026-11-12 — Ubuntu 提交与 Windows 发布中转

Ubuntu 调度器只保存待发布提交，Windows 推送并读回后登记清单。22 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-12-windows-relay-publication-policy.md)

## 2026-11-11 — 双目 RGB 光轴深度辅助头

Ubuntu 65 项 CPU 回归通过。历史 VGA 复放通过左右 RGB 梯度检查，右图缺少独立标签。[报告](docs/superpowers/reports/2026-11-11-stereo-depth-auxiliary-diagnostic.md)

## 2026-11-11 — 独立排期刚性代理和柔性 Cosserat 验收

刚性代理与柔性 Cosserat 各有原生验收依赖。Ubuntu 19 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-11-rigid-deformable-scheduling-gates.md)

## 2026-11-11 — DeformX 固定源码实际载荷预检

CPU `passed`，通过 7 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp5.source-payload.md)。

## 2026-11-11 — 状态监督残差及不确定度完整性

CPU `passed`，通过 18 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp2.metric-integrity.md)。

## 2026-11-11 — 准备 MPC 371D 与实际 24D 同步采集

CPU `passed`，通过 50 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp3.mpc-sync-prepare.md)。

## 2026-11-11 — GitHub 传输失败时保留 Ubuntu 工作包

Ubuntu 调度器在 GitHub 连接异常时保留已测试提交。17 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-11-work-package-publication-defer.md)

## 2026-11-11 — 验证 PhysX 逐接触来源与条件方向

CPU `passed`，通过 50 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp4.source-audit.md)。

## 2026-11-11 — 双目候选置信度分组校准审计

CPU `passed`，通过 26 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp1.confidence-offline.md)。

## 2026-11-11 — Student 独立评估证据

Student 冷评估核对实际发出 24D、传感与推理时钟及逐样本任务身份。Ubuntu 92 项 CPU 回归通过，原生验收待资格与审批。[报告](docs/superpowers/reports/2026-11-11-stereo-student-independent-proof.md)

## 2026-11-11 — Ubuntu 状态合同审计

兼容旧 v1 配置的两个已审计字节表示，继续记录实际原始 SHA。集成 CPU 82 项通过，原合同与 Checkpoint 未修改。

## 2026-11-11 — 复核 Student 成功池与自有 rollout

CPU `passed`，通过 22 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp6.cpu-admission.md)。

## 2026-11-11 — 复核 DeformX CUDA 源码资格检查器

CPU `passed`，通过 2 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp5.cpu-source-check.md)。

## 2026-11-11 — 复核 PhysX 接触证据与退出合同

CPU `passed`，通过 41 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp4.cpu-contract.md)。

## 2026-11-11 — 复核 Teacher 动作合同与 PPO 回滚

CPU `passed`，通过 94 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp3.cpu-contract.md)。

## 2026-11-11 — 验证三种 Actor 的独立双目部署

CPU `passed`，通过 47 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp6.deployment-isolation.md)。

## 2026-11-11 — 区分状态监督的离线与原生资格模式

CPU `passed`，通过 49 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp2.mode-gates.md)。

## 2026-11-11 — 保留双目原色采样与候选分池

CPU `passed`，通过 36 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp1.sample-retention.md)。

## 2026-11-11 — Ubuntu 工作包调度

Ubuntu 工作区负责开发、CPU 验证与提交。本机只负责 GitHub 中转。调度器和中断恢复验证 14 项通过，原生任务与依赖安装按独立批准门槛执行。[执行计划](docs/superpowers/plans/DH116_AUTONOMOUS_EXECUTION_MASTER_PLAN.md)

## 2026-11-11 — PhysX 复验申请

既有冻结快照和服务器预检已重新核对。申请范围为单实例顺序两轮、每轮 240 步的自然关闭与重启，原生执行待批准。正式 BC、DAgger、Student PPO 及 Solver 构建继续阻断。[审批前复核](docs/superpowers/reports/2026-11-11-stereo-physx-shutdown-retest.md)

## 2026-11-11 — 双目匹配固定像素与采样归因

归档左右三类概率和固定尾部的双向代价。Roll20 软引导遗漏了仍有正确深度的原基线点，二指 83 个、三指 76 个。原色锚定补测仍低于原色基线，48 条固定尾部全数超 5 mm，候选保持 `blocked`。[复放报告](docs/superpowers/reports/2026-11-11-stereo-wire-point-traces.md)

## 2026-11-11 — Teacher128 同状态动作复盘

原 Checkpoint 对 256 条同一 371D 状态重算 mean 和 std。五次越限都发生在右 `finger12` 的实测位置，噪声没有一致方向。旧 MPC 缺同一状态的 371D，当前可配对数为 0，Actor-only BC 继续阻断。[复盘报告](docs/superpowers/reports/2026-11-11-teacher128-paired-action-replay.md)

## 2026-11-11 — 线缆点来源与时序身份

按点区分中心线、实测表面及未知槽，保持缺深度时的二维路径。历史 Roll20 正确点全部来自实测表面，未观测到稳定的 3D 中心线轨迹。新候选逐点旁路通过 CPU 回归，原生复验待批准。[验证报告](docs/superpowers/reports/2026-11-11-stereo-wire-geometry-sources.md)

## 2026-11-11 — 143D 同步证据与分阶段准入

同步候选绑定原生参考身份和左右实例 ID 的可用性。冷 8 只进入诊断，Velocity 需要同一已核验中心线 track 和连续物理时钟。历史数据仅有 6 行中心标签，单轮 CPU 诊断完成，正式状态训练仍阻断。[验证报告](docs/superpowers/reports/2026-11-11-stereo-state-stage-admission.md)

## 2026-11-11 — 正式训练准入复核

42 项 CPU 准入回归通过。Teacher 仍缺完整成功回合，双目正确点完整率尚未合格，正式 BC、DAgger 和 Student PPO 保持 `blocked`。[准入复核](docs/superpowers/reports/2026-11-11-stereo-formal-training-admission.md)

## 2026-11-11 — DeformX CUDA 来源复查

服务器 19 个候选目录仍无目标 CUDA 求解器独立仓库。新增只读候选核验工具，检查指定提交、子模块和实际源文件。构建与双向耦合继续 `blocked`。[来源复查](docs/superpowers/reports/2026-11-11-stereo-deformx-source-recheck.md)

## 2026-11-11 — PhysX 自然关闭预检

独立脚本验证 240 步完整采样、回调释放、所属 SimulationContext 清理、自然退出与顺序重启。41 项 CPU 测试和服务器预检通过，原生复验待批准。[执行计划](docs/superpowers/reports/2026-11-11-stereo-physx-shutdown-retest.md)

## 2026-11-11 — 双目线缆对应复放

原始左右 RGB 的完整匹配窗口保留纹理，分割概率只提供有界代价。两组历史捕获复放均恢复有效点，Roll20 软引导二指/三指正确点为 67/65，原色轮廓为 91/89。固定 48 条尾部仍全部超过 5 mm，历史捕获缺右侧原生 ID，候选保持 `blocked`。[验证报告](docs/superpowers/reports/2026-11-11-stereo-wire-correspondence.md)

## 2026-11-11 — Teacher128 关节动作审计

历史五次 `joint_limit` 均观察到右 `finger12` 实测角度跨过硬保护线，SDK 目标保持在名义限位附近或以内。新增有界诊断的逐 480 Hz 全关节动作证据，原生复验待批准。旧 MPC 专家数据缺少同一 pre-action 的 371D，特权 Actor 初始化需同步重采。[验证报告](docs/superpowers/reports/2026-11-11-teacher128-joint-limit-audit.md)

## 2026-11-11 — 双目状态监督多初态准入

现有状态监督入口可读取绑定 SHA256 的多来源清单，并按物理初态和采集族分配 train、validation、test。历史数据回读为 8 行、2 个初态，只有 6 行 Center XYZ 标签。正式训练继续阻断。[验证报告](docs/superpowers/reports/2026-11-11-stereo-state-multicohort.md)

## 2026-11-11 — 按工作包持续发布

每个完成验证的独立工作包自动提交并推送至 `codex/dh116-stereo-privileged-dagger`，随后核对 GitHub 远端提交和 Draft PR #2。失败工作包保留本地日志并发布必要的诊断与 `blocked` 状态。原生实验继续遵循独立批准门槛，8 初态同步采集仍待批准。[工作流程](AGENTS.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json)

## 2026-11-11 — 按明确指令发布WP增量与采集预检

本次按用户“更新到github”指令，将基于 `ae307a5` 的代码、独立配置、测试、处理后指标和预检报告发布到 `codex/dh116-stereo-privileged-dagger`，保留 Draft PR #2，不合并。新的8初态同步采集仍未获批准，本次发布不启动Isaac或训练。

发布前复验7份新增核心测试：**84 passed，11.56s，exit0**。1183份源码/脚本/配置/测试与已完成双平台1147项完整回归的冻结版本逐字节一致，528份旧保护文件不变。此次复验不重复计作新原生结果。原始图像、物理日志、模型和私有归档不加入Git。

本次包含真实RGB分割及point-source路径、三模式中心状态监督、Teacher KL事务与诊断、冷Reset同步证据准入和失败保存、PhysX来源/关闭诊断及768点尾部审计。Stereo精度、自然退出修复原生复验、逐contact来源与成功Teacher仍未通过；R2、Force Proxy、正式BC/DAgger保持关闭。最终测试和限制见[整合报告](docs/superpowers/reports/2026-11-11-stereo-wp-integration.md)、[readiness](docs/contracts/stereo_phase0_7_readiness.json)和[待批准采集计划](docs/superpowers/reports/2026-11-11-stereo-cold8-preflight-plan.md)。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_candidate_cold_trials.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_teacher_wp3.py
```

以下“未上传/不push”说明保留为预检阶段的历史快照；本条明确上传授权只改变发布范围，不改变实验授权或晋级状态。

## 2026-11-11 — WP1–WP6增量与8初态采集预检查（未上传）

以远端 `ae307a5` 为基线，保留已有架构。新增native-ID监督的双RGB分割、中心线/实测表面来源编码、右侧参考归档与逐点尾部审计；三种状态监督模式实际完成中心61–63的CPU诊断拟合。修复候选采集冷Reset读取、跨链初态身份、Checkpoint双检及失败样本保留。默认相机、371D/24D与原物理成功标准未变。

Roll20同cohort复放为二指91/96、三指89/96个正确点，固定分母2009/2054，完整率4.53%/4.33%。这是采样覆盖改进，尾部仍14–22mm；分割实际接入仍失败，历史片段ID未记录，不能授部署或跟踪资格。真实6行中心标签的训练内MAE为5.395/1.522/4.258mm，没有独立validation/test。

Teacher已完成本轮2env×128steps×1update原生R0诊断：256transition、5个joint_limit终止、成功0，受保护最终KL0.019917488。PhysX左拇指几何标定两次各240步完整，但退出需要采样后定向SIGTERM；修复代码经过CPU验证，尚待原生自然退出复验，pair来源未资格化。CUDA solver仍不可用，R2/Force Proxy、正式BC/DAgger/Student PPO保持关闭。

最终CPU：Windows **1147 passed / 1 skipped**（174.53s）；服务器现有Python3.10/Torch2.7 **1147 passed / 2 skipped**（152.2s），均exit0、无Isaac导入，统计不相加。528保护文件SHA不变。用户本轮明确暂不采集新的8初态、不push/merge，PR保持Draft。

[总报告](docs/superpowers/reports/2026-11-11-stereo-wp-integration.md) · [逐点预检](docs/superpowers/reports/2026-11-11-stereo-wp1-tail-preflight.md) · [待批准8初态计划](docs/superpowers/reports/2026-11-11-stereo-cold8-preflight-plan.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json) · [本轮文件清单](docs/stereo_wp_20261111_file_inventory.md)。以下保留历史阶段记录。

## 2026-11-11 — 双目点云、状态监督与 Teacher PPO 验证

本次更新以 085f6b8 为基线。Roll20 保存帧采用轮廓匹配和连通表面采样后，每帧二指 31/32 点、三指 30/32 点在 5 mm 内。正确点完整率分别为 1.54% 和 1.46%，三指光轴深度 P95 为 11.92 mm。该配置仍用于独立实验。

完成 143D 逐字段审计及三种 Actor 模式的 CPU 验证。新的 2 环境、4 步原生候选采集核验出 6 条相对中心 XYZ 标签。两个独立初态不足以完成三路划分，状态估计器训练仍待合格数据。两次独立的 2 环境、32 步、1 更新 Teacher 原生诊断均以 0 次抓取成功结束。在同一个 0.02 KL 目标下，回滚及学习率回退将最终 KL 从 5.84524 降至 0.01997，Checkpoint 回读一致。R2 和正式 BC/DAgger/Student PPO 的准入仍待验证。

本阶段完整 CPU 回归在 Windows 和服务器各通过 1063 项，分别跳过 1 项和 2 项。上传前执行以下 CPU 回归，93 项通过，用时 6.34 秒。

    python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py

发布范围包括源码、配置、测试、字段审计和报告。原始测量、模型及日志保存在私有证据目录。[总报告](docs/superpowers/reports/2026-11-11-stereo-reliability.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json)

## 2026-11-11 — Stereo 运行链路修复发布

发布基线为 `9bfee426363527190011dbca164c104a00cbc999`。修复 RGB 二维拓扑与独立可见片段、TaskStateAdapter 到 Actor、左右掌局部点编码、真实 Teacher 身份和动作标签；保留旧路径，新增独立 latent/task-state/hybrid 配置。高分辨率 PPO 改为唯一帧池与有界 minibatch，粗到细区分默认 audit 与显式 accelerated。

最终 CPU 回归：Windows **1030 passed / 1 skipped**（153.36s），服务器 **1030 passed / 2 skipped**（180.92s），均 exit0。完成 2/8/16/32 环境的 CPU/GPU 帧池测量、三模式 2env×4step 原生配对、Teacher 2env×1update 及 deterministic mean 采集诊断。原始二指0点失败被保留；逐分量拒绝修复后的同帧复放恢复4点，三指9点。详细原生复验见报告。

这些是运行链路证据：Teacher/Student 没有取得成功资格，三模式未训练，深度完整率仍待验证；tiny Teacher batch KL偏高，同一episode跨多update的checkpoint切换边界仍需补齐。rigid_proxy明确标识，deformable_rod仅严格状态接口，真实后端未接通。528保护文件与30mm/5s/0.5mm标准不变，本阶段训练范围为短时诊断，发布目标为 Draft PR #2 的开发分支。

上传前在 CPU 环境执行 pytest -q -p no:cacheprovider @tests，@tests 为 PowerShell 展开的全部 tests/test_dh116_stereo_runtime_*.py 和 tests/test_dh116_stereo_student_ppo.py。176 项通过，用时 23.78 秒。

[完整报告](docs/superpowers/reports/2026-11-11-stereo-runtime-repair.md) · [机器摘要](docs/contracts/stereo_runtime_repair_validation_summary.json) · [readiness](docs/contracts/stereo_phase0_7_readiness.json) · [文件清单](docs/stereo_phase0_7_file_inventory.md)。以下为历史记录。

## 2026-11-11 — 按用户授权上传J2-A/J2-B

本次发布现有开发分支的J2-A（`a5a15fb`）与J2-B（`3cc5c12`）代码、独立配置、测试、报告和readiness；沿用[Draft PR #2](https://github.com/ssdsj12/dexterous-hand-dh116/pull/2)，保持Draft且不合并11-10。最新完整CPU回归885/886与48解析案例为上一阶段已执行证据；本次上传前另执行63项cost/integration检查，63 passed（20.36s，exit0）。528保护文件和1131份服务器已测源码SHA复核一致，未运行新Isaac/GPU实验或训练。

原10run深度精度、物理抓取及部署资格仍未通过；下一工作包J2-C尚未执行。原始图像、运行日志、模型、缓存和私有归档不提交。推送后核对远端分支与Draft PR的HEAD SHA一致；下文保留各阶段当时未推送的历史记录。

## 2026-11-11 — J2-B：四种独立匹配候选

新增显式Census/Hamming、ZNCC、中心梯度L1与.25/.5/1粗到细候选；独立配置/SHA，完整RAW支持、同指标亚像素与全range审计，不覆盖旧默认SAD或模型。新增134项回归纳入最终相关组：Windows885 passed/1 skipped（149.97s）、server886 passed/1 skipped（140.38s），均exit0、无Isaac导入。48个固定解析case通过列出门槛；四候选标准coverage全100%、最大视差P95 .0250001px，Census存在极少坏尾，不能只看P95。

这只是软件/解析证据。原10run native accuracy、物理与部署资格不变，GPU/完整吞吐未测；粗到细全range审计不宣称加速。无新Isaac、训练、push或merge。下一工作包J2-C，J2-F同原始数据比较与J3仍待执行。[完整报告/实际命令/失败记录](docs/superpowers/reports/2026-11-11-stereo-j2b-matching-candidates.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json)。以下保留阶段历史。

## 2026-11-11 — J2-A：matcher合同与可靠亚像素候选

本轮按任务书第十六节完成J2-A：旧默认SAD五Tensor通过全图逐值回归；SGBM的0/1兼容confidence明确为有效性标志，真实遮挡和不可观测的内部拒绝项标为NA。新增显式启用的`reliable_sad`、五成本验证、独立图像空间分数验证及单频重复纹理拒绝；原控制、物理参数、模型和Checkpoint不变。

新增J2-A组合 **156 passed（29.86s）**；最终相关全组Windows **751 passed / 1 skipped（115.60s）**，服务器现有Python3.10/Torch2.7cu128 **752 passed / 1 skipped（78.03s）**，均exit0。实际命令、源码冻结SHA和失败证据见[独立验证报告](docs/superpowers/reports/2026-11-11-stereo-j2a-matcher-contract.md)。旧J1 Pixel A仅做保存图像的离线crop检查，深度误差7.218→2.681mm；不能代表全场景或新原生精度。

解析遮挡边缘仍有27个错误有效点，一般重复纹理鲁棒性也未证明。**本轮无新Isaac/GPU实验、无训练、无push/merge**；原10run深度精度、物理任务和部署资格继续未通过，正式BC/DAgger仍blocked。下一工作包J2-B待执行，不能以本次软件通过提前晋级。[机器摘要](docs/contracts/stereo_j2a_validation_summary.json) · [readiness](docs/contracts/stereo_phase0_7_readiness.json) · [文件清单](docs/stereo_phase0_7_file_inventory.md)。下文保留历史阶段结果。

## 2026-11-11 — 发布J0/J1诊断版本

按用户明确上传指令，将本地`33d73d3`的J0/J1代码、测试、报告及readiness发布到现有开发分支，继续保持[PR #2](https://github.com/ssdsj12/dexterous-hand-dh116/pull/2)为Draft，不合并`11-10`。本次发布复核：31项诊断/几何CPU测试通过（9.75s，exit0）；528份原保护文件不变，1111份源码/测试与此前服务器完整回归冻结版本一致，52份派生图SHA匹配。

先前完整回归595/596及10run原始数据重分析结果仍见下文和报告，不冒充本次新原生实验。深度精度、物理任务与部署资格继续未通过；下一步J2按已提交的代码/测试/资源计划实施。原始图像、运行日志、模型和私有归档保留本地，不上传。

## 2026-11-10 — J0/J1：最新分支审计与深度根因实证

按新786行任务书第11节，本轮只执行J0/J1：以GitHub最新`7c80878`为基线复核readiness/保护合同，新增评估专用分区诊断CLI、局部几何分桶和31项测试，重读已有10次native capture；没有重建Phase0～7、改变matcher/物理/控制/成功门槛或开始训练。

最终CPU：Windows **595 passed / 1 skipped，80.43s**；服务器现有Python3.10/Torch2.7cu128 **596 passed / 1 skipped，70.27s**，均exit0。已有40 batch frames/80 env-frame重分析生成52张真实图；saved/replay validity mismatch=0，158条原输入/来源绑定SHA末尾复核均未变。新增native次数0，accuracy/物理/部署资格仍false，未选最终分辨率，正式BC/DAgger仍blocked。

关键定位：Block9大量正确整数winner被负向亚像素偏移推过5mm；SGBM坏尾包含LR=0的自洽错配，confidence=1实际只是范围标志；当前金色fixture颜色阈值准确，但实际WirePointExtractor在12个环境观测均0/128 token，严格数字骨架拓扑另有阻断。K/T、光轴Z与半像素约定内部一致，未用改标定/缩ROI降低误差。

[J0审计/gap matrix](docs/superpowers/reports/2026-11-10-stereo-j0-latest-source-audit.md) · [J1完整诊断/命令/限制](docs/superpowers/reports/2026-11-10-stereo-j1-error-diagnostics.md) · [逐像素根因](docs/superpowers/reports/2026-11-10-stereo-j1-matcher-rootcause.md) · [J2代码与测试计划](docs/superpowers/plans/2026-11-10-stereo-j2-evidence-driven-perception.md)。本轮不自动push/merge，Draft PR保持；下文为先前阶段快照。

## 2026-11-10 — 双目Phase0～7隔离实现与原生分辨率实测

已落地371D/24D/奖励审计、独立Teacher PPO、同步双RGB/亚像素深度、时序Actor与状态估计、BC、在线DAgger、Student PPO/actor-only推理、冷初态与原始证据验收软件。**这是实现及测试进度，尚未完成新Teacher/Student抓取验收。** 原生产源码、控制器、物理参数和历史Checkpoint不变。

最终相关CPU回归：Windows **564 passed / 1 skipped，75.44s**；服务器现有Python3.10.12/Torch2.7cu128 **565 passed / 1 skipped，67.68s**，均exit0。之后执行10次独立原生相机配置采集：四分辨率与front/roll20对照，2环境、每次4计量帧，480Hz/60Hz不变，无模型训练。采集及重读验证通过，**深度精度资格全部未通过，未选最终分辨率**。640×480 roll20/block9的wire MAE7.978mm、正确3D点完整率46.05%，有效覆盖93.87%不能当作正确完整率；Actor延迟与完整训练吞吐仍未测。

[完整实施/命令/限制](docs/superpowers/reports/2026-11-10-stereo-phase0-7-integration.md) · [原生分辨率结果](docs/superpowers/reports/2026-11-10-stereo-native-resolution-final.md) · [修改文件清单](docs/stereo_phase0_7_file_inventory.md) · [readiness账本](docs/contracts/stereo_phase0_7_readiness.json)。下一步是深度精度与force/support真实来源接线，再做新Teacher/Student独立原生验收；BC/DAgger正式晋级保持关闭。独立分支 `codex/dh116-stereo-privileged-dagger`，PR保持Draft，不合并。下列同日条目保留各阶段当时的测试快照，以本条和最终报告为最新结果。

## 2026-11-10 — R1/R2隔离因子与可用监督补全

实现四类R1几何势能改善及R2承载/恢复/负载/执行目标变化因子；候选默认关闭，缺几何承载份额不补50/50，原R0不改。native runtime增加逐维核验的24个SDK关节监督，以及当前实测FK下的左右掌局部可见点；force、隐藏几何与未确认硬件仍无资格。

完整冻结回归：Windows CPU528 passed/1 skipped（73.57s，exit0）；服务器现有Python3.10.12/Torch2.7cu128 CPU529 passed/1 skipped（71.81s，exit0），未导入Isaac、未安装依赖。实际R1/R2 measurement factory与独立geometry资格仍待接通，不能把纯因子算术当原生奖励验证。随后才启动独立2环境相机分辨率实验；不训练模型、不合并PR。

## 2026-11-10 — 双目分辨率设计：640×480精度基线

支持224×96、320×240、640×360、640×480及资源范围内自定义尺寸；新主精度配置为640×480，旧224文件完整保留。改变尺寸重建/缩放K，严格同步双RGB，使用流式亚像素匹配及图像分批，不降低480Hz/60Hz物理和控制精度。

[分辨率实现报告](docs/superpowers/reports/2026-11-10-stereo-resolution-v2.md)记录21项分辨率测试与115项相关CPU通过组合。服务器现有OpenCV数值路径已执行，但首轮整体CPU仍有6项快照/跨Python审计问题待重跑；尚未执行原生分辨率测量或选定最终分辨率。先补齐所有隔离软件与测试，再按实际深度MAE/RMSE/P95、线缆覆盖和吞吐/显存选择，不能用640×480默认值冒充测量结论。

## 2026-11-10 — Stereo Phase 7：冷初态评估与原始证据晋级

新增每种抓法至少100实际独立初态的冷试验runner、逐480Hz回合重算、Teacher/Student分别统计、独立heldout最佳模型选择及公共晋级门槛。原生terminal在reset前保留；不接受summary成功flag、clone峰值或手填训练初态清单授信。缺force/support来源、外力/附件审计或完整训练来源登记时继续阻断正式DAgger/部署。

修复新bootstrap对原launcher成功SystemExit的处理，恢复callback返回与完成报告；18项真实runpy CPU回归及关联164项通过（16.23s，exit0），原launcher/单实例锁不变。[Phase7报告](docs/superpowers/reports/2026-11-10-stereo-phase7.md)、[冷试验报告](docs/superpowers/reports/2026-11-10-stereo-cold-trials.md)逐项列出pending；当前尚无本新架构原生资格。下一步补完可独立的奖励因子并重新冻结验证，再测试分辨率；PR保持Draft。

## 2026-11-10 — Stereo Phase 6：独立Student PPO与RGB推理

新增双目Actor/371D privileged Critic的独立asymmetric PPO、完整优化断点、actor-only导出与raw左右RGB独立推理。BC→DAgger→PPO保留实际数据/模型文件SHA和预处理合同；换matcher/分辨率不能静默套用旧模型。训练来源登记在raw文件关闭后进行，不把导出或BC loss视为部署资格。

本轮CPU集成463 passed、1 optional OpenCV module skipped（72.90s，exit0）；部署/来源/优化相关44项组合通过。服务器首轮CPU458通过、6失败已定位为快照路径/文档和跨Python AST审计问题，正在新冻结快照重跑；未升级环境、未开始native训练。详见[Phase6报告](docs/superpowers/reports/2026-11-10-stereo-phase6.md)。新raw RGB路径尚无原生深度/抓取资格，原3844D路径不变，PR保持Draft。

## 2026-11-10 — Stereo Phase 5：在线DAgger

新增Student访问状态上的实时Teacher查询、逐环境整24D owner切换、append-only聚合和独立BC更新。D4为Student执行，Teacher仅标注；D0参考及失败状态不能冒充成功BC示范，也不写入PPO on-policy storage。

本轮8项DAgger协议测试纳入450项CPU集成并通过（exit0）；[Phase5报告](docs/superpowers/reports/2026-11-10-stereo-phase5.md)记录接口与边界。正式DAgger仍需要新Teacher和双目原生资格，尚未运行；后续继续Student PPO与独立推理接线，PR保持Draft。

## 2026-11-10 — Stereo Phase 4：同步采集与独立BC

新增严格pre-action对齐采集、带来源/有效性掩码的数据集、按实际初态分组的holdout及独立Adam BC。成功示范准入需要本次完整原生回合证据，不能用历史Teacher成功给新轨迹授信；BC保持原PPO storage隔离。加载推理模型后的BC会临时启用训练/图像分块反传，并恢复原模式。

本次全链CPU回归450 passed（61.80s，exit0）；随后BC/运行集成19 passed、1 optional OpenCV module skipped（18.02s，exit0）。这些是软件fixture与合同测试，没有启动Isaac或取得抓取资格。详见[Phase4报告](docs/superpowers/reports/2026-11-10-stereo-phase4.md)。下一步在线DAgger与native验证仍独立推进，PR保持Draft。

## 2026-11-10 — Stereo Phase 3：共享视觉编码与时序状态估计

新增共享GN-ResNet18、双目cross-attention、四帧Transformer、geometry/contact/force-proxy/slip/371D估计heads与24D Actor。Actor只接收严格的双目传感历史；状态监督按逐维来源mask排除NA，force须重读原始资格证据，默认无效。

[Phase3报告](docs/superpowers/reports/2026-11-10-stereo-phase3.md)记录27项CPU单测；随后含collection/DAgger/BC/runtime的集成组合 **51 passed，exit0**。这些是软件/梯度验证，原生估计精度和Student抓取未验证。图像处理可分批，224×96不作为上限；640×480主精度配置与亚像素扩展独立进行。

## 2026-11-10 — Stereo Phase 2：同步RGB、深度与坐标软件路径

新增两台真实RGB相机的独立采集入口、标定读回、纯Torch视差基线/可选SGBM、显式无效掩码、world→实际palm FK变换和仅RGB可见线缆骨架提取。CPU组合 **43 passed，exit0**；相机原生采集和实际精度尚未验证。

224×96/80mm基线下整数视差在0.5～0.8m距离约21～55mm/px，不能称为精细抓取精度。遮挡/分叉/无纹理不填仿真真值；SGBM及原生标定须单独验证。[Phase2报告](docs/superpowers/reports/2026-11-10-stereo-phase2.md)与`probe_dh116_stereo_camera.py`、`evaluate_dh116_stereo_depth.py`明确区分CPU检查、native probe和离线参考误差。

## 2026-11-10 — Stereo Phase 1：独立 privileged PPO Teacher

新增371D→256→256 Teacher Actor/Critic、24D saved-latent Tanh Gaussian PPO、独立Actor BC、独占Checkpoint及三份CLI。R0逐值复用原任务奖励，不改旧控制器/模型。集成CPU **49 passed，exit0**，包括合成更新、保存/恢复与terminal保留；不是原生训练或抓取成功。

[Phase1集成报告](docs/superpowers/reports/2026-11-10-stereo-phase1-integration.md)列出命令、文件及原生入口限制。所有native执行仍须显式opt-in，Teacher成功资格、R1/R2物理来源及正式DAgger门槛未通过。继续开发隔离的双目/估计/蒸馏模块，PR保持Draft。

## 2026-11-10 — 双目 RGB Teacher–Student：Phase 0 审计

在独立分支 `codex/dh116-stereo-privileged-dagger` 完成[371D逐维状态](docs/stereo_privileged_state_contract.md)、[24D动作](docs/stereo_action_contract.md)和[奖励](docs/privileged_teacher_reward_audit.md)审计。当前371D为153+106+92+20；旧3844D RGB-D输入不等于双目RGB输入，历史261D/26D Teacher合同不能直接作为新Teacher资格。

本次CPU组合覆盖真实observe/控制器算术、奖励dt/覆盖与成功门槛，没有启动Isaac或训练。528个原有源码、脚本和配置逐文件SHA不变，`11-10`仍指向`6cee7418`，没有读写历史Checkpoint。最终测试数、完整命令、来源SHA及尚未资格的support/feedback见[审计报告](docs/superpowers/reports/2026-11-10-stereo-phase0-audit.md)。

用户已授权按[Phase0～7计划](docs/superpowers/plans/2026-11-10-stereo-privileged-dagger-implementation.md)持续实施。下一步新增独立privileged PPO Teacher和R0奖励透传；R1/R2支撑资格、Teacher原生成功及正式DAgger晋级仍须独立验证。PR保持Draft。

## 2026-11-10 — CUDA 12.8 最终目标与独立安装计划

按用户最新要求，最终扩展固定以当前 Python 3.10.12 / Torch 2.7.0+cu128 / **CUDA 12.8 / SM89** 重新编译，不用现有 12.2/12.4 代替。只读复核仍未找到 12.8 Toolkit；拟新增用户目录 `/mnt/mydisk/xk/toolchains/cuda-12.8.0`，不替换 Driver、原 Runtime、Python/Torch/Isaac 或已有 Toolkit。

已制定[独立安装与源码准入计划](docs/superpowers/plans/2026-11-10-cuda128-isolated-toolkit.md)和[官方组件计划锁文件](docs/source/deformx_cuda128_toolchain_plan.lock.json)：选择 12.8.0（nvcc 12.8.61），33个用户态归档逐包SHA锁定，排除6个Driver/内核/服务组件。仅使用新目录，避免全局CUDA链接和loader变更；若需管理员操作，先给出命令及影响，等待确认。

**未安装、未构建、未原生验证。** 源码准入仍阻塞：现有Windows GitHub身份已确认是 `ssdsj12`，需维护者开放指定CUDA提交及嵌套依赖或提供完整可核验源码。未再次重试已返回404的后端，未擅自修改认证或原训练；PR保持Draft。

本次发布前静态核验：`git diff --check`、6个拟发布文件的凭据模式扫描与本次新增相对链接检查、33项计划锁和官方manifest逐项对照通过；这些检查不等于安装、编译或Isaac验证。原始审计产物、运行日志、缓存和模型不上传。

## 2026-11-10 — DeformX 服务器补查：源码身份与安装来源

按当前服务器 `/mnt/mydisk/xk/dh116` 补查真实源码、Git 根、gitlink、子模块和认证：找到 19 份旧 PyElastica/co_sim vendored 源码，没有找到可核验的目标 CUDA 后端；主项目没有 gitlink 或 `.gitmodules`，不能把空的 submodule status 输出当作安装成功。服务器 `gh` 未登录、Git HTTPS 无可用凭据、GitHub SSH 22 端口超时；与此前 Windows 凭据的 404 分开记录，没有重复下载已拒绝的后端。

实测解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`（3.10.12），由 `activate.sh` 指定并通过原 wrapper 显式验证。发现 **Isaac Lab distribution=0.54.4，而源码根 VERSION=2.3.2**；源码无 Git 元数据，不能据此认定官方未修改版本。包安装器为 uv，Lab 为本地 editable；Sim 部分包明确来自本地 wheel，其余历史下载来源未记录。

该阶段只记录实测环境，不升级或替换；后续用户已明确最终编译使用独立 CUDA 12.8，现有12.2/12.4仅作为基线保留。没有修改原 vendor、控制器、模型或训练，没有构建/原生 GPU/PhysX/抓取验收。详见[服务器补查报告](docs/superpowers/reports/2026-11-10-deformx-server-source-access.md)，其中列出解除源码阻塞所需材料。

## 2026-11-10 — DeformX GPU 原环境适配：源码与工具链审计

已锁定指定上游 `8f32ad7`、CUDA gitlink `96b8514` 与官方资产 revision `9ef440fa`；629 个已取得源码文件通过 Git blob SHA 重验。经服务器现有 `scripts/python.sh`，426 个 Python 文件在实际 Python 3.10.12 上通过语法编译检查；3 个最小资产文件通过 SHA 校验，USD/NPZ 可读取。没有启动 Isaac 或将这些检查计为 GPU/物理验证。

**CUDA 后端源码不可访问仍是明确阻塞**：上游后端处于需权限的 release preview，Windows 现有凭据访问返回 404；服务器现有认证另行核验，不混用身份。现有 Torch 为 2.7.0+cu128，找到的 Toolkit 为 12.2/12.4，仅作为保留基线；最终构建按顶部计划使用独立 CUDA12.8，源码及SDK验收前不编译。完整 `ropesim.isaac_bridge`、cp310 CUDA 构建、双向 PhysX 接触及 DH116 抓取均未完成。没有升级环境、修改控制器/策略/checkpoint 或覆盖原 `vendor/DeformX`。

实际命令、版本差异、接口与单位风险、资产清单及下一步见[兼容性与验证报告](docs/superpowers/reports/2026-11-10-deformx-gpu-compatibility.md)。PR 保持 Draft，不合并。

## 2026-11-10 — Stage 7 D：承托转移八阶段请求状态机

新增未接SDK的APPROACH/CLOSE/STABILIZE/LOAD_TRANSFER/LIFT/AIRBORNE_HOLD/RECOVERY/FAILED状态机。0.2s实测双侧对向稳定及有源object-following后，才建议0.5mm/s、累计请求曝光上限2mm的转移。

本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 C：局部力驱动响应与批量FK数学预览

新增完整17D局部implicit capped force-PD数学响应及批量mounted FK，左右手按各自SDK行、关节轴和单位计算。NONE/零cap驱动与缺参数分开，未知量不补0，原始M不对称化、不加单位惯量或完美mimic。

本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 B：H8原命令桥接与约束诊断

新增未接控制的64子步命令假设与固定13项约束报告，复用原共享kernel和真实左右SDK命名映射。每手12个直接目标不冒充17D实际响应；反馈冻结/外推必须显式声明。

本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 A：三路被动接触观测侧车（默认关闭）

复用现有Wire→桌面/两端支架视图，新增默认关闭的只读采集接线；保留实际路径、每getter时钟/来源和独立normal/friction原值及NA。不创建World/view，不改变原控制或support字段。

本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 双手接触状态与原预算委托

新增独立、未接线的双手接触状态wrapper，区分真实接触与几何候选、未知接触与LOST，显式支持pinch/tripod必需指。原1mm/1s/0.5s恢复预算与reserve/commit顺序委托既有ledger，接触闪断不补预算。

独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元27项是其中子集，不另相加。本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。详见[实现与验证边界](docs/superpowers/reports/2026-11-10-stage7-bimanual-contact-runtime-cpu.md)。

## 2026-11-10 — Stage 7 H8 完整17D运动学外推

新增独立、未接线的双手完整17D运动学核心：8个60Hz控制周期展开为64个480Hz子步，显式速度预测与当前q/qd、来源、候选摘要分开。常速外推与候选无关，不能提供动作响应、物体/接触/驱动预测或控制准入。

独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元39项是其中子集，不另相加。本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。详见[实现与验证边界](docs/superpowers/reports/2026-11-10-stage7-h8-kinematic-cpu.md)。

## 2026-11-10 — Stage 7 原生 descriptor 失败定位与 USD ABI 修复

首次 descriptor 原生短测自然退出，但8帧、64条样本全部结构不可用；封存完整性审计通过，不能称为FK/J验证通过。已定位到当前USD绑定要求 `GetPropertyStack(Usd.TimeCode.Default())`，仅修补Attribute/Relationship调用，原失败数据保留。独立source16 CPU实跑1364 passed、3 skipped、238 subtests；修复版原生复测仅已准备，尚未执行。固定腕双侧恢复实验仍单独运行，MPC/Student无新增成功。命令、SHA与边界见[失败与修复报告](docs/superpowers/reports/2026-11-10-stage7-descriptor-usd-abi.md)。

## 2026-11-10 — Stage 7 mounted descriptor 与17D运动学预览

新增默认关闭的只读descriptor接口与左右独立完整17D FK/J假设，保留各自真实关节轴、被动状态、来源和NA。独立source12 CPU实跑769 passed、1 skip、238 subtests；本单元新增Native作业0，实际FK/J校准仍待单实例原生验证，不能用于控制准入。原Teacher/物理/3844D/371D/24D Tanh和成功标准不变，PR继续Draft。详见[实现及验证边界](docs/superpowers/reports/2026-11-10-stage7-mounted-model-cpu.md)。

## 2026-11-10 — Stage 7 八项MPC代价算术

新增独立CPU八项cost计算，逐项记录raw/weighted、单位、来源、mask和NA；严格区分当前已测常量与候选相关预测。独立实跑73 passed、41 subtests，尚未接入原生控制，不是完整H8 MPC或抓取成功。原Teacher180步接触观测继续独立运行；默认控制、物理、3844D/371D/24D Tanh和成功标准不变。FK/J、动力学、求解/约束及正式抓取配对仍待完成，PR继续Draft。命令与边界见[算术验证报告](docs/superpowers/reports/2026-11-10-stage7-mpc-objective-arithmetic.md)。

## 2026-11-10 — Stage 7 接触数值与首回合退役观测

新增默认关闭的v2首回合退役ledger与F1 normal/friction数值归档，复用原34 sensors、真实env/行/时钟与issued24，保留首terminal、独立NA和失败partial；旧v1/控制器/物理/Student输入保持。

公开范围独立CPU实跑639 passed、1 CUDA-unavailable skip、238 subtests。epoch1、values1与values16三次原生短测均自然退出；8/8/128帧封存后审通过，数值两通道分别2176/34816 valid、NA为0，但非零观测均为0，不能当成接触承托或成功。全部回合截断，无真实terminal，异步退役完整原生资格仍待验证。180步接触建立观测窗口独立运行中，尚无结果。

M0、默认Teacher、3844D/371D/24D Tanh与30mm/5s/0.5mm不变，不启动B3/PPO，PR仍Draft。H8、新Teacher100初态配对与新专家/Student尚未完成；实际命令、SHA及边界见[验证报告](docs/superpowers/reports/2026-11-10-stage7-filtered-contact-values.md)。

## 2026-11-10 — Stage 7 新鲜接触路径配对验证

新增默认关闭的F0元数据观测，读取原34个hand view实际sensor/filter路径、SDK/source/stage/view身份和逐getter时钟，按真实env/行绑定Wire，不用配置表达式补值。原控制器、Actor/Critic/24D动作与物理参数不变。

独立1/16控制步Isaac短测均正常退出，8/128个物理帧及2176/34816条配对记录的闭合SHA、CRC、原Core/Aux动作/时钟关联与raw重算通过。所有回合均截断，无新增Teacher/Student成功。公开范围CPU实跑582 passed、1 CUDA-unavailable skip、223 subtests；正常量/摩擦数值、作用侧、承托、未来动力学和实时资格尚未取得。

严格F0 v1在任一环境首terminal后拒绝继续，不能把短测当完整异步首回合资格；独立逐env退役/显式generation收据与数值接触观测仍在开发。默认Teacher、B3/PPO、3844D/371D/24D Tanh及30mm/5s/0.5mm保持，PR仍Draft。实际命令、SHA、失败边界和下一步见[验证报告](docs/superpowers/reports/2026-11-10-stage7-filtered-contact-metadata.md)。

## 2026-11-10 — Stage 7 完整关节读回与接触交付检查

新增默认关闭的完整 articulation 读回与独立 480Hz 归档，保留实际关节/目标/驱动参数、当前 M/C/G 和逐字段来源/NA。修复实际 SDK enum wrapper“等值但不是同一对象”的识别问题后，左右手各17 DOF/23 links/22 joints已在独立原生采集中读回；Model16的128帧/1024条状态完整后审通过，44个当前字段各1024 valid，实际motor torque、隐式PD估算和未来候选动力学共6个字段继续NA。

两个独立接触交付实验正常退出且归档检查通过：post-step callback128次但三向量全空；full-report subscription未收到callback，同步poll仍空。空报告不证明无接触或无桌/支架反力。保留source08 Model字段绑定失败及原始矩阵，未将旧NA回改为有效值。所有本次回合均为短预算截断，未增加Teacher或Student成功。

本公开观测单元独立CPU回归526 passed、1 CUDA-unavailable skip、147 subtests。原M0、PPO v2/Tanh、3844D/371D/24D、四个原控制器、物理/材料/执行器与30mm/5s/0.5mm保持。采集为低速诊断，不授H8动力学、60Hz实时、合格接触/承托或专家资格；不启动B3/PPO，不替换默认Teacher，PR保持Draft。命令、SHA、失败和后续门槛见[原生读回报告](docs/superpowers/reports/2026-11-10-stage7-articulation-contact-delivery.md)。

## 2026-11-10 — Stage 7 场景/接触来源与完整首回合证据

新增默认关闭的辅助测量：实际 composed Cylinder、COM/惯量、Wire→桌面/端支架的三个低层 view、PhysX contact report 来源检查，以及与原状态归档共享 issued24/时钟的只读 480 Hz 数据。仿真特权字段不进入 Student，原控制器和物理参数不变。

原 Teacher 完整采集已自然结束：1800 控制步、14,400 物理帧、84,461 条首回合状态、8 个真实 terminal；二指 4/4，三指 0/4（两次掉落、两次超时）。关闭后的 raw/DTO/动作/时钟和 reset/no-op 隔离检查通过。仍是标称来源的副本，不是新 100 初态、新 MPC 或 Student 成功。

独立 CPU 组合验证 336 passed、1 CUDA-unavailable skip、93 subtests。保留 source05 的真实 Shadow 初始化失败；修复来源明确的 Sequence→aux→state 链后，source06 的 16 步原生冒烟 exit0，128 帧/1024 状态及关闭归档完整性通过，8 个回合均为截断。端支架 raw view 有非零观测，但全局 contact report 为空，作用侧/符号/完整性和合格支撑仍未验证，不能补造 partner ID 或推断无反力。

M0、旧 Teacher、3844D Actor/371D Critic/24D Tanh、原材料/执行器和30mm/5s/0.5mm不变。H8/恢复/载荷转移、正式配对和新 Student 训练仍受后续门槛约束；无默认 Teacher 替换或实时资格，PR保持Draft。实际命令、失败、SHA与后续缺口见[原生测量扩展报告](docs/superpowers/reports/2026-11-10-stage7-mpc-contact-source-audit.md)。

## 2026-11-10 — Stage 7 MPC 状态与原生只读测量

统一 Teacher `MPCState`、实际测量桥和默认关闭的 `--vision-mpc-state-capture` 已实现。逐字段保存来源、单位、时钟、有效性及 env/episode；实际 backend 跟踪误差、COM 原点换算、raw 接触归属和首 terminal 隔离均单独校验，仿真特权字段不进入 Student。

本次独立 CPU 验证 185 passed、1 CUDA-unavailable skip、58 subtests。实际 Isaac 短时采集已自然 exit 0：8 个混合环境、180 控制步、1440 物理帧和 11520 条状态，逐帧时钟/动作/raw 对齐。全部回合为截断、terminal 0，不能当作完整任务成功率。合格支撑、桌面/端支架反力等能力缺口仍为 NA；两次启动准备失败及修复也保留。

原预算 1900 控制步的独立旧 Teacher 首回合采集正在执行，用于实际 terminal/no-op/reset 验证。新 H8 MPC、逐指恢复、载荷转移及正式二指/三指各 100 个不同初态仍待后续门禁；不启动新 Student PPO。M0、旧 Teacher、3844D/371D/24D Tanh、原物理和 30 mm/5 s/0.5 mm 标准保持，PR 继续 Draft。命令、SHA 和已知限制见[测量验证报告](docs/superpowers/reports/2026-11-10-stage7-mpc-state-measurement.md)。

## 2026-11-10 — Stage 7 首次物理分叉与目标选择

第一证据单元完成：E0/B0/B1/B2 的真实来源首回合均 0/4、未达 30 mm、合格 Hold 为 0。两次 Teacher 重放各 4/4 单列，不计 Student。64 项逐次冷重置短切换及 64 项六时点后处理全部完成，共 256 个有界窗口、41,472 个 Student 实际物理子步，缺失 0；全部正常截断，不能当完整任务成功率或失败率。四来源仍是一个标称布局的副本，不是新 100 初态验收。

当前没有可晋级的闭环目标赢家：B0 保留为控制组、B1 为事件采样诊断基线、B2 为条件命令损失消融假设；不按离线拟合或继承 Teacher 高度选模，B3/PPO 继续关闭。HOLD/60 虽各臂均聚合接触 1 s，B1/B2 四环境原生 Hold 计时均中断；3 mm 诊断锚漂移与原生 30 mm/5 s 窗口锚分开。下一步独立实现真实接触状态和 Teacher/MPC 闭环，尚未更换默认 Teacher、生成新合格专家池或修复 Student 抓取成功率。

实际独立 CPU 组合回归 310 passed；命令、最终执行/交接/汇总 SHA、完整失败、逐环境保持比较及 256 行附件见[Stage 7 报告](docs/superpowers/reports/2026-11-09-stage7-teacher-student-divergence.md)。保持 3844D/371D/24D Tanh、M0、原物理/材料/执行器及 30 mm/5 s/0.5 mm 标准，PR 继续 Draft，不合并。

## 2026-11-09 — Stage 6 连续视觉监督与条件命令展开

已实现16步连续数据、事件采样、共享原控制计算的可微命令展开，以及独立B0/B1/B2训练。三组现已分别完成3000次更新、192,000个有效样本曝光，padding为0，每250次保存实际模型与固定视觉探针；未按训练MSE选模。主模型继续使用ReLU、hemisphere和单次Tanh，旧BC/PPO路径保留。

新版采集02已自然结束：二指Teacher/MPC 4/4，三指0/4；4698条成功视觉记录、37,584个live物理子步的原目标、SDK cast、PhysX backend及cache分别误差0，4638个16步窗口也通过累计目标一致性。533项冻结聚焦CPU回归通过，本次训练/激活及旧BC相关检查另为101 passed。新的纯Student来源/短分支原生评估仍在运行，尚不宣称抓取改善；没有新PPO。实际命令、三个最终模型SHA及未通过门禁见[Stage6实施记录](docs/superpowers/reports/2026-11-09-stage6-sequence-control.md)。3844D/371D/24D、M0、原Teacher及30mm/5s/0.5mm标准保留，PR仍Draft。

五命令入口`python scripts/run_dh116_stage6.py --help`已接入：准备真实序列、训练独立候选、来源纯Student、Teacher前缀分支和独立heldout。实际Teacher tape冷重置重放刚完成4/4、37584物理样本，属于Teacher重复基线。物理实验仍逐项重新reset，缓存只复用SHA绑定的离线观测比较。新的纯Student完整结果及首次物理分叉尚待作业结束后报告，不能用训练MSE或前缀成功替代。

新增独立Tier1 Teacher扰动manifest入口，限制物体XY±0.5mm/yaw±1°，保持原关节、手根、速度及double命令seed；原生reset/相机读回通过后才使用第一帧。成功二指首回合、失败/未完成池和保守来源lineage分别保存，目前仅完成CPU与真实输入清单验证，扰动原生次数0。有限评估驱动也已以`scripts/run_dh116_stage6_evaluations.py`发布；完整命令、125/200项独立范围CPU结果和未通过门禁见[实施记录](docs/superpowers/reports/2026-11-09-stage6-sequence-control.md)。

共同暂停双腕现为默认关闭的独立完整回合实验：原Actor、单侧失败对照、共同暂停分开记录。阈值绑定真实Teacher同钟PhysX backend减post-physics q的P99.9；98项本次相关CPU检查通过，尚未完成这三个新原生回合。原Teacher/MPC强制卸载接口与移动腕资格仍未获准，不把新hybrid保护默认装入Student。

本次四来源纯Student的E0基线已实际结束：0/4成功，3掉落、1超力，四个首terminal及原始物理轨迹保留；B0/B1/B2继续按相同初态串行对照。Teacher重放4/4与E0纯Student0/4分别计数，尚不认定衰退已修复。

固定腕三指恢复已支持显式left/both与N×2×3逐指状态，原A/right仍为默认范围；每episode累计1mm/1s及0.5s无进展预算不因接触闪断重置。新的双侧/旧恢复及有限驱动CPU检查79 passed。Stage B只代表固定腕资格，移动腕真实几何门禁、C/D/E及三指Student训练仍未开放；五项原生跟踪/恢复计划需单实例逐项验证。

本次上传新增只读物理分叉分析与真实成功池合并工具，并修复共同暂停的CPU读回/CUDA掩码索引、Tripod已结束slot被旧目标覆写两项接口问题。最新检查197 passed、2 CUDA skip，另真实Torch CUDA2执行2 passed（SDK格式仍为合成fixture，非Isaac验证）。E0及B0纯Student来源均已完整0/4；B1进行中、B2待运行。Teacher/MPC成功继续单列，未宣称Lift修复或启动新PPO，PR仍Draft。

## 2026-11-09 — Student原生执行链复核完成

同模型、同seed、同200个初态的诊断复测已自然结束，二指/三指仍各0/100成功。逐480Hz真实backend/cache、掌位、动作、首terminal和原始物体遥测完成跨流核验；额外自动reset记录不计评估次数。二指两掌最高上移50 mm时物体仍接近原高度，掌位峰值不代表抓取。完整逐手分位、失败原因、峰值env/UID/时刻和同环境轨迹见[执行链审计](docs/superpowers/reports/2026-11-09-stage5-student-execution-audit.md)。

二指Teacher原路径及Adapter路径各4/4仍是Teacher/MPC结果；三指和视觉Student尚未通过成功标准。没有新PPO、三指BC或DAgger，PR继续Draft。原生作业已结束，保留全部失败实验、模型和冻结源码。

本次收口核验了证据SHA、样本/失败计数、文档链接及`git diff --check`；未修改源码或重跑模型拟合。最近源码聚焦检查为522 passed、1 skipped，完整命令/退出原件及全仓旧失败限制见报告，不能将文档检查当作新增CPU或原生通过次数。

## 2026-11-08 — 第五阶段物理映射与双手协同验证

目标来源更正：旧T1/TeacherAudit的命令/缓存不能当PhysX后端读回。新独立原生补证已确认两次finger12事件的后端目标合法且与cache一致，实际仍跟踪越界；仅覆盖所存3个watch环境和0.533秒窗口。原生Teacher成功仍有效，历史模型/数据和来源标签不回写。

新增逐 480 Hz 的关节跟踪审计、左右独立 USD FK/Jacobian 适配器和独立 Z0～Z3 协调实验。T1 重放再次复现右 `finger12` 两次越限，合法目标与实际关节运动之间出现 0.052～0.055 rad 跟踪误差；隐式执行器力矩为仿真估算。T2 完成 53 组、8 环境微扰，左右指腹源 FK 对 SDK 位置最大误差 0.000354 / 0.000399 mm，右 `finger12` 正负方向各 8/8 正确。详见[关节审计](docs/superpowers/reports/2026-11-08-stage5-joint-tracking.md)和[物理映射报告](docs/superpowers/reports/2026-11-08-stage5-physical-mapping.md)。

Z0～Z3 已完成同一模型、初态和随机种子的二指/三指各100例配对：800个首回合全部有效，原生成功全部为0。置零右Z消除了约4 mm下沉，却未建立合格抓取；Z3三指关节越限由M0的12/100升至66/100，该协调候选不具备晋级资格。逐环境掌位峰值身份、接触、载荷、动作与跟踪统计见[双手干预报告](docs/superpowers/reports/2026-11-08-stage5-bimanual-interventions.md)。原生短窗中共同暂停双腕消除了两次旧越限，但其0.533秒结果不能替代完整回合安全证据。

新增显式V4 Adapter模型绑定与推理前资产/源码/SDK合同核验，旧模型仍走原路径。Teacher专用480Hz审计分别保留请求、执行、实测关节与首终止，并保留重锚前的真实接触锚滑移；三指协调候选必须先通过二指Teacher回归门禁，修改后的三指动作不进入BC监督池。未来配对路径已修复结束环境被协调器重新驱动的问题；现有冻结Z3按首终止截断统计，未将后续尾段算作合格执行。

本次Teacher原路径和Adapter对照均已完成二指4/4成功、最高31.627704 mm、Hold5 s，T5门禁通过；两臂10559条动作和主要物理状态一致。三指两臂仍各0/4；独立协调候选也已完整结束，二指4/4、三指0/4，右侧接触提高伴左侧退化，不能晋级，详见[T6对照报告](docs/superpowers/reports/2026-11-08-stage5-teacher-coordination.md)。

只读探针发现源模型PointNet在已测实际输入上特征为0，原run未做Teacher视觉蒸馏。本次二指BC1000次离线更新恢复坐标敏感性，MSE降至0.000195265；旧模型不改写，Critic/PPO状态保留。首轮200初态原生评估完成，两抓法均0/100，最高净Lift二指25.235939 mm、三指0.651073 mm，尚未通过30mm/5s/0.5mm标准。本次BC没有三指专家标签（源PPO曾混合训练）。标准评估缺失败码和完整逐手SDK链，后续独立补采现已完成；不回填原件，同模型、同200初态重复不算400个新初态，也不调参，详见[视觉迁移报告](docs/superpowers/reports/2026-11-08-stage5-visual-transfer.md)及上方执行链审计。

本次200条原生冻结初始观测的只读前向中，旧模型点云敏感0/200、新模型200/200；这只证明初始坐标敏感性，不代表SDK运动或抓取成功。

完整CPU冻结快照为6136 passed、113 failed、48 skipped、25 errors，另91 subtests passed；该快照相对`3f7fd69`的195个新增节点全通过、无新增回归，原有失败和依赖问题单列。后续真后端/完整日志修正的统一聚焦回归为522 passed、1 skipped，控制/奖励及reset parity通过。移动腕逐指恢复仍缺少几何资格，未绕过碰撞检查。M0默认、M2独立、3844D/371D/24D和30mm/5s/0.5mm标准保持，PR继续Draft。见[总验证报告](docs/superpowers/reports/2026-11-08-stage5-native-validation.md)。

## 2026-11-08 — 右掌动作偏置与原生关节越限

原生回放复现了三指环境 932 和 799 的右手 `finger12` 越限。两条轨迹的 SDK 主动关节目标均在限内，实测角度在物理步 252 和 256 穿过下界保护线。五个回放组使用同一物理初态。固定输入重推三个实际 M0 Checkpoint 时，第 1024 轮已存在右掌 Z 负偏置。详见[物理终止记录](docs/superpowers/reports/2026-11-08-vision1024-stage4-native-terminal.md)和[分阶段 PPO 统计](docs/superpowers/reports/2026-11-08-vision1024-stage4-ppo-bias.md)。

确定性评估入口使用 `--vision-evaluation-checkpoint`、冻结的 200 例 manifest 和 `--paired-bias-arm`。M0 与 M2 逐个核对真实初态，右 Z 干预回放 M0 的其余 23 维动作。动作、物理子步和首个终止逐环境存档。详见[原生配对评估](docs/superpowers/reports/2026-11-08-vision1024-stage4-native-pair.md)和[关节越限的五组对照](docs/superpowers/reports/2026-11-08-vision1024-stage4-joint-limit.md)。

四个确定性对照臂共用二指、三指各 100 个冷初态。M2 三指净抬升 P95 达到 0.401537 mm，同组 M0 为 0.074721 mm；三指越限由 12/100 升至 81/100，两种策略、两种抓法的原生成功数均为 0。保留 M0 默认合同。将 M0 的右掌 Z 置零使右掌末位移中位数由约 −4 mm 接近 0 mm，两组均未出现合格 Hold。

## 2026-11-08 — 视觉 Student 右掌不对称 P0 诊断

本阶段先做逐环境动作链审计，不修改右手奖励、动作幅度、Actor结构或物理参数。`--vision-action-diagnostics` 为默认关闭的只读诊断开关，保存24D动作合同（`finger12|palm_translation6|rotation6`）、左右手请求/投影/最终SDK目标、实际掌位与速度、reset-relative坐标、env/episode/physics step/时间戳、净空、safe_z、接触载荷和有效接触。

CPU镜像回环与动作切片测试通过；独立最终Checkpoint单更新回放记录32768条逐环境样本，左右安全投影前后差值均为0、投影比例均为0，右掌Z动作均值−0.289而左掌−0.047。实际掌位改用每个环境物理 reset 后的实测掌位作为固定基准，控制器动态 lift offset 另行记录。该证据支持“原始左右初始化/策略缺少镜像等变性”是当前最强方向，尚不足以认定唯一根因。第513轮Checkpoint与旧峰值的逐环境轨迹不存在，不能补造或冒充对比结果。原始Checkpoint、物理参数、30 mm/5 s/0.5 mm标准、3844D/371D/24D合同均保留，PR继续Draft。

详见右掌不对称审计报告。

## 2026-11-08 — 右掌偏置第二阶段双层诊断

新增默认关闭的60Hz policy/PPO和480Hz physics双层诊断。短原生回放保留每环境每policy step的24D动作、latent、log_std、log_prob、value、advantage、returns和全部reward terms，并按每个480Hz物理子步记录掌位、速度、接触载荷、clearance、safe_z、投影原因和最终SDK目标。terminal样本立即保存，后续no-op不会覆盖。

离线M0/M1/M2镜像消融结果为 L2 0.754994/0.806492/0.296581，M2为`offline_only`，没有接入旧模型。独立短原生验证产生32768条60Hz记录和262144条480Hz记录，未启动长PPO。详见[第二阶段诊断报告](docs/superpowers/reports/2026-11-08-vision1024-stage2-diagnostics.md)。

## 2026-11-08 — 右掌偏置第三阶段

第三阶段完成32768条PPO诊断的按抓法/左右手偏置分析，并新增独立M2原生反事实路径。M2只在独立输出目录中将镜像观测送入Actor、将动作逆变换回物理坐标，M0默认路径不变。M2短回放二指/三指净Lift为0.003487/0.014168mm、成功0，出现2次joint_limit，因此没有修改默认部署合同。诊断完整性增加reset epoch、唯一episode_uid、时间/物理步单调性和terminal覆盖测试。

## 2026-11-07 — 右三指独立恢复与固定手腕原生 A 实验

新增独立的逐指腹恢复状态机、双手全源几何非线性候选审核和每480 Hz实测诊断。每个物理episode/指腹保留1 mm接近命令与1 s尝试预算，0.5 s无实际进展停止本轮；接触闪烁、求解失败或恢复均不补充预算。实际post-guard SDK提交后才记账，控制步中途接触建立会撤销剩余接近动作，真实丢失也不能让旧压力速度变成无预算接近。压力保持按每个实测子步重建，不依赖旧MPC的永久停止计数。

新增默认关闭的 `--tripod-recovery-stage A`，先重放原Pressure前缀，再冻结实测双腕和左指目标，仅测试右拇指/食指/中指。最新完整源码回归为5917 passed，原96 failed/28 errors/49 skipped保持，91个subtests通过；5786个基线节点状态无变化，304个新增全部通过，6090个节点的最终源码清单保持不变。此前A02快照的5883/270结果已被该完整源码回归取代；这些均不是原生抓取证据。

首次原生启动 `stage_a_01` 在进入 A 控制前因桌面父Xform与子碰撞体的读取不匹配退出，**A试验数为0**。修复仅汇总启用的真实子碰撞体，保留原世界边界与碰撞限制；3项定向和2项独立USD测试通过，失败目录与冻结源码保留。

独立 `stage_a_02` 随后进入真实480 Hz物理子步，但操作者以SIGINT结束，结果被独立审计标为 **censored/invalid evidence**，不能记作A通过、失败或三指0/4：481次逻辑控制、482个物理区间、3856个原始SDK子步，30,840条物理行，8个首episode均censored、0个完整episode、0次接近提交、0预算消耗，0 Student/PPO。env4–7只出现自然Tripod接触，阶段内0次REACQUIRING、0次固定阶段恢复、0 terminal/success。终止前杆/腕没有达到固定阶段的静止条件，实测角漂移约0.0022–0.00328 rad（门槛0.001），几何决策0次；SIGINT flush顺序导致末8个诊断缺失，未伪造汇总。A仍未取得资格，B/C/D及Student训练不启动。当前原生对象沿用原刚性圆柱代理，离线候选和探针不能替代柔性线缆或真实DH116硬件验证。

后续 cooked readback 探针也只验证读取链路，不构成A或抓取结果：03在原生reset-only下272个mesh getter均为0；04使用owner-first读取后每个存储hull getter count=1，但272个hull全部未通过独立closed-convex复合核验，因而仍为complete=false/censored、live attachment=false，未recook或替换几何。两次均无控制、物理推进、Teacher/PPO/Student；04仅有一次RGB-D读取。对应owner-first模块/测试SHA前缀为 `ddf703d...` / `e55cb...`，完整CPU仍需在这批最新源码后重跑。

所有历史模型/失败实验、二指成功路径、原物理/材料/执行器、3844D Actor/371D Critic/24D动作与30mm/5秒/0.5mm成功标准保持；不启动长PPO，PR #1保持Draft、不合并。完整来源、原生结果及限制见[本阶段恢复验证报告](docs/superpowers/reports/2026-11-07-tripod-recovery-budget-validation.md)。下文为先前阶段结果，不能冒充本次验证。

## 2026-11-07 — 三指接触获取与约束MPC实验

新增显式 `pressure_geometry_acquire` 和 `pressure_geometry_constrained` Teacher 专家实验。前者使用实测 body pose、源 USD 指腹和当前圆柱表面，为尚未承载的三指指腹生成有界几何目标；已有承载 pad 保留压力参考。后者只对违反原约束的 soft 提案求解可行替代轨迹，已可行解逐值保留。两者均优化12个指关节，Teacher腕部动作保持；局部几何预测不代表真实接触、力预测或抓取成功。

第一轮完整原生实验 `geometry_acquisition_pilot_02` 已自然结束并经独立审计：二指保持 **4/4 成功**、最大净抬升 **31.627704 mm**；三指仍 **0/4**，最大净抬升仅 **0.006431 mm**，四个首回合均因 `object_dropped` 结束。实际480 Hz记录中，三指双手六个所需pad从未同时承载，接触保持仍未解决；失败数据不进入成功监督池。固定来源四环境回放不计作100个独立初态，也不是Student验证。

对实际失败决策的逐步重放表明，六次永久几何停止对应的soft提案被拒，但相同约束下存在可行替代解；因此增加独立约束求解profile，保留原腕部、目标/权重、预算、永久停止、执行器和物理限制。新增默认关闭的 `--vision-teacher-hold-tail-control-steps`，显式记录来源结束后的零参考、实际MPC issued动作、RGB-D及首terminal；默认0沿用历史行为。尾段不延长原30秒episode超时、不重置失败回合，也不能把未记录动作计入合格标签。

第二轮 `geometry_constrained_pilot_03` 使用同物理初态、seed42、原相机与源轨迹，已在独立目录完成1800控制步及8个完整首回合，独立审计exit0、无截断：二指仍 **4/4成功、最高31.627704 mm**；三指仍 **0/4、最高仅0.019719 mm**，两个在1092/1169步掉落，两个在原30秒限制下超时。1056个尾段样本全部核对零参考、实际动作和首终止，未产生尾段成功或三指合格标签。三指双手六pad加对向承载仅在两个环境各出现1个物理子步（2.083 ms），远未形成稳定保持；实际载荷和深度峰值增加，不能称抓取或Lift退化已解决。

新求解器已消除早期误拒，完整审计核对524次求解调用、实际编码动作和45504个准入子步，执行限位违规为0。但后段仍存在固定法向点Jacobian模型的H4约束冲突；这只是局部模型的数值证据，不表示物理抓取不可达。CAD中心点间隙也不能等同于实际指腹接触状态，后续需改进接触与运动预测。已完成177项最终factory/tail/相关回归、261项求解器相关回归及六个真实失败快照检查；最终完整CPU5613 passed、原96 failed/28 errors/49 skipped、91个subtests通过，原5670节点不变，本次116新增全通过。未训练新Student，不能用CPU或专家接触片段替代Student成功。

3844D Actor、371D Critic、24D动作、Tanh Gaussian/default v2、原物理/执行器和净30 mm、连续5 s、漂移0.5 mm成功标准保持。所有源模型和历史数据保留，没有启动长PPO或将仿真接触追加到Student。实现、命令、证据与未完成项见[本次验证报告](docs/superpowers/reports/2026-11-07-tripod-reacquisition-validation.md)。PR #1继续Draft，不合并。以下为此前阶段记录。

## 2026-11-06 — 接触MPC验证与视觉Student输入、启动拟合修复

新增显式 `absolute_reference` / `pressure_reference` Teacher MPC 实验，以及 Actor 内部的 `wrist_quaternion_hemisphere_v1` 和独立 BC v3 启动采样选项。3844D Actor、371D Critic、24D动作、Tanh Gaussian PPO、原物理/执行器限制及30mm、连续5秒、0.5mm成功标准保持；默认v2与历史raw模型不静默切换预处理。仿真接触仅供训练专家使用，没有新增已接通实机反馈的声明。

BC100纯Student已完成固定清单的二指/三指各100个冷初态、1800控制步；独立重放1,503,649条原生物理样本，0无效测量，但两组均0/100成功、0双手有效对向接触，净抬升仅为数值沉降量级。实际输入审计确认等价腕部quaternion的符号变化会改变旧Actor动作，且原BC100在Teacher自身启动状态上也明显欠拟合；这两项证据不能用总体离线MSE或PointNet有梯度替代。

`absolute_reference_pilot_v2` 已完成并通过独立二进制/时序/原生成功观察器审计：二指恢复4/4成功、最高净抬升31.627704mm；三指仍0/4，保持原 `excess_force` 失败判据。独立 `pressure_reference` 实验也已完整审计：二指仍4/4，三指全部越过旧约419步的过载失败时刻，但在684/938/783/864步因 `object_dropped` 结束，最高净抬升仅1.07533mm，仍0/4成功。接触建立/保持不足，没有合格三指标签；两项均为8环境固定来源回放，不是每抓法100个独立初态或Student成功。

此前三份1000步离线拟合只有一个二指来源、无三指标签和独立heldout。显式将25%的采样混合权重分配给前4控制步后，启动MSE由均匀1000步的0.00449366降至0.0000122437，但全来源MSE增加3.97%，阶段切换误差仍存在；132项独立保护/采样核对通过。随后新Student候选已使用绝对参考MPC的合格4698帧（3355帧动作经MPC修正）完成1000步canonical BC及25%启动混合采样，101项独立检查与73项保护状态检查通过。来源仍只有一个二指轨迹、无三指标签/heldout；Critic、噪声、PPO Adam/LR/计数/RNG及原件保持，离线拟合不代表新Student成功。

最终完整CPU结果为5497 passed、原96 failed/28 errors/49 skipped、91个subtests通过；5529个旧节点无缺失/状态变化，新增141项全通过，并非整个旧套件全绿。新候选纯Student评估（`student_mpc_canonical1000_100_per_style`）已完成200/200回合，独立审计exit0，原生执行证书与条件遥测完整性均核验通过，但二指/三指成功率仍均0/100，30mm达成与合格Hold均0。二指Lift P50/P90/P95/max为0.00388034/0.0111964/0.102051/24.537986mm；按全部5个pad统计的双侧对向接触在100回合均曾出现，但总时间占比仅0.7462%，100回合均丢失，共5765次。三指对应Lift为0.0000558165/0.0000843213/0.0001004292/0.0008006615mm，完全无双侧对向接触，滑移为NA。输入修复和启动拟合尚未解决稳定抓取；首批200个原生动作与CPU最大差1.19e-7及quaternion符号不变性复验只验证推理一致性，不另计原生试验。实际命令、来源、审计和限制见[本次接触MPC与Student验证报告](docs/superpowers/reports/2026-11-06-contact-aware-mpc-validation.md)。不启动长PPO，模型与原始运行数据不进入Git，PR #1继续Draft、未合并。以下历史记录保持原阶段语境。

## 2026-11-06 — Teacher参考与MPC监督的视觉Student

新增独立的“Teacher参考动作 → 当前原生状态的局部MPC → 同步视觉示范 → Actor BC → 显式v3 warm-start”路径。保留Tanh Gaussian PPO、3844D感知Actor、371D训练Critic、24D动作、默认v2及原物理/执行器/30mm、5秒、0.5mm成功标准。MPC的仿真接触/Jacobian仅供训练专家生成标签，不追加到Student输入。

MPC实际优化未来4步的12个手指命令，腕部12维跟随Teacher；它是局部pad运动学模型，尚无物体/摩擦/力动力学或抓取获取保证。数据保存pre-action真实RGB-D、四帧Actor输入、原参考和实际issued动作、frame/env/episode/物理时序与来源SHA。只有当前真实原生首回合成功、无recovery并通过抓法接触审计才可进入成功监督池；失败数据独立保留，CPU注入相机/环境不能取得原生资格。

Student以单次`tanh(mu)`拟合实际issued24D，独立Adam只更新PointNet/Actor MLP；Critic、噪声、PPO Adam/LR/计数/RNG和历史模型保留，新BC及warm-start使用独立文件。Teacher/MPC动作不进入Student的PPO rollout，不混用Student log probability。四份历史Teacher回放只有两个原始轨迹来源，不能按相邻帧随机拆分宣称独立验证。

本次Teacher二指4/4原生成功（最高31.620mm），三指0/4；原局部MPC二指0/4（最高12.489mm），失败标签被隔离。独立审计还发现来源1439的PointNet全死，初版BC只拟合本体信息；现新增显式`reset_pointnet`及独立BC v2，默认拒失活分支，必须验证真实梯度和点云对动作的影响。新BC100使6个PointNet tensor全部有梯度，MSE0.043277→0.001974，清点云动作差最大0.1392；不是抓取成功或独立验证集选模。

新模型关闭Teacher/MPC后完成8环境180步原生推理，exit0、无无效测量，但0完整/8未完成episode。CPU回归、原生Teacher/MPC对照、两版实际BC及明确限制见[本次验证报告](docs/superpowers/reports/2026-11-06-visual-teacher-mpc-supervision-validation.md)。不启动长PPO，不宣称视觉Student可靠抓取或Lift退化已解决；PR #1继续Draft。下文为此前各阶段记录。

最终完整CPU5356项及91个subtests通过；原96失败/28错误/49跳过逐节点保持，原5416节点无缺失/变化，新增113项全部通过。BC v2专项32项和独立复审通过；模型和原始运行数据不进入Git。

## 2026-11-06 — P0 接触诊断与真实源CAD几何筛选

新增默认关闭的逐指腹原生接触诊断：左右10个pad的建立、持续、丢失与未知/终止censor事件，绑定env、episode、实际物理子步和时间，保存原始PhysX点/法向、缓冲归属和真实body pose。分别统计二指/三指、左右手和双手有效对向接触，区分7种原因；力矩明确为模拟估计，未知滑移不填0。保留Tanh Gaussian PPO、默认v2、3844D Actor、371D Critic、24D动作、原物理/限制和30mm/5秒/0.5mm标准。

实际129/1439模型的同初态配对评估已完成：每模型每抓法100次，0 invalid，真实readback逐字节相同，全部原生成功均0。129三指62/100曾形成对向接触、1248次观测丢失；1439二指23/100、568次；其余两组从未形成有效对向接触。完整Lift P50/P90/P95、保持、接触物理时间占比及模拟力矩饱和见[本次验证报告](docs/superpowers/reports/2026-11-06-grasp-contact-geometry-validation.md)。

沿中心线的候选搜索使用每手17个原USD hull、各自5个CAD指腹、原mimic/主动限位、完整线缆及self/interhand/桌面/原支架筛选和静态承托LP。02及原腕限位内扩大姿态的03搜索均为每组24候选、最多200次几何优化，二指/三指均0 accepted；几何前置条件不满足，未进入LP，未放宽阈值或把失败项送入闭环。所有几何结果native准入固定False，候选到原生source/scene/reset门禁仍待实现。当前原生对象仍为刚性圆柱代理，弯曲线仅离线proxy。

完整CPU回归5243项及91个subtests通过，旧96失败/28错误/49跳过不变，原5314节点无缺失/状态变化，新增102项全通过。8环境120控制步原生历史观察冒烟已完成，逐pad接触记录有效；0个完整episode、不执行PPO、不计100次验证或抓取成功。长PPO排队已取消；课程训练、R0-R5/Adam对照及泛化仍未执行。PR #1继续保持Draft，尚未证明可靠抓取或Lift退化已解决。

## 2026-11-06 — T2 基础设施与新版 PPO 原生冒烟（09:44 UTC快照）

补充完善：新增默认关闭的原生训练初态审计，在相机前后保存全部环境真实 q/qd、手根与物体位姿/速度及三个物理时钟；Adam 对比须在第一次 PPO 更新前与同源 reference 匹配。修复真实1439轮合同中已知 tuple 字段经 JSON 保存后被选模误拒的问题，完整模型与 typed 合同哈希仍严格。评估诊断改为整批 blocking CPU 快照，CPU/CUDA 的终止记录和原始 JSON 对照一致；这些新增路径尚未取得原生对比结果，不能计入100次统计。

面向 `11-5` 的 PR #1 保持 Draft。新增冷重置视觉评估、终止前物理记录、原生证据校验、独立验证集选模、完整 Checkpoint 周期保留、显式保留/重置 Adam warm-start 和独立 R0～R5 奖励实验合同。Actor3844D / Critic371D / 动作24D、物理参数和30mm/5秒/0.5mm成功标准保持；没有把未确认的实机传感反馈加入 Student。

新版 `tanh_gaussian` 已在真实 Isaac 的16环境中完成1440、1441更新，保存后恢复完成1442；RGB-D有效率100%，log-probability重建最大误差约3.34e-6。另一次原生无更新恢复逐项保持策略、Adam、LR、CPU/CUDA RNG、计数和完整合同相等。原生successes仍0，这些结果只证明动作、采样、更新及保存恢复链路。

最新完整CPU回归5141项及91个subtests通过；原96失败、28错误、49跳过逐节点保持，4801个基线和5079个上阶段节点均无缺失/状态变化，本阶段新增235节点全通过（比d796917再新增40个全部通过）。规格和质量复审已完成；不能将这些CPU结果当作原生抓取改善。

用户后续明确授权原进程退出：先保留1750轮完整原模型（44897回合，successes0，SHA837791335e5f1c2153db763ed419381c708ec8b1f8ff2d4abfc6fa025aab249d），再正常SIGINT并核实GPU3释放；原模型仍保留，没有把1750称为完成2048。所有派生任务按单Isaac实例、独立代码快照及输出串行运行。实际129模型已完成每抓法100个原生冷初态并经完整raw重放核验：两组成功均0%，二指/三指最大净lift分别0.005627mm/1.033019mm；1439模型继续相同条件评估，奖励消融/Adam有效性/验证选模结果尚待原生验证；不能宣称Lift退化已经解决。详见[本次原生进度与限制](docs/superpowers/reports/2026-11-06-native-stage2-progress.md)、[固定实验协议](docs/superpowers/plans/2026-11-06-native-experiment-protocol.md)。

## 2026-11-06 — Lift 修复：T1 已实现并通过 CPU 验证，原生评估待执行

独立分支 `codex/dh116-lift-repair-20261106` 已接入 `--vision-action-distribution tanh_gaussian`：保存 latent 和实际有界命令，使用正确变换概率、有界熵/KL与微批PPO，命令和动作历史不再额外clip。新模型使用独立v3合同；旧v2默认路径保留，只能通过显式离线工具派生warm start。完整权重、Adam、LR、已有Torch RNG和已知静态配置差异在写入/启动Kit前检查，原始文件及输出受保护，拒绝会反转PPO优化方向的maximize模式。

最终定向CPU回归 **327项通过**（根代理独立复跑22.52s），规格复审和质量复审均PASS。完整CPU回归4906项及91个subtests通过；保留原96失败、28错误和49跳过，基线4801个节点状态逐项一致，新增278个节点全部通过。真实冻结1439轮断点离线派生保留原字段，四组历史视觉输入的确定性动作差为0；原件SHA不变。命令、文件差异及限制见[集成验证报告](docs/superpowers/reports/2026-11-06-lift-repair-t1-integration-validation.md)，不能把CPU测试当作原生抓取成功。

Actor仍3844D、Critic371D。按[硬件审计](docs/superpowers/reports/2026-11-06-dh116-hardware-feedback-audit.md)，当前没有已证实接通的实机反馈通道，20D仅作带来源/单位/时序/有效性的可选接口设计；PhysX接触和成功监视器不追加到Student。原1207个源码/资源文件与冻结清单一致；2026-11-06 06:03 UTC原GPU3训练已完成1548/2048次更新，原生successes0，继续运行，未停止、重启或切换。

同时准备了T2冷重置manifest和首terminal统计基础设施；生产schema每抓法至少100个独立float32状态，缺测值保留null和coverage，始终标为authored_unverified，未进行每抓法100次原生验证。完整视觉评估adapter、真实初态读回、验证选模/周期保留、奖励消融及修复版重新训练仍按后续门槛推进。没有证据表明Lift下降或稳定抓取已经解决。

## 2026-11-06 — 独立 Lift 修复：T1 概率核心通过 CPU 验证

独立 `tanh_gaussian` 核心保存 latent `z`、实际 `tanh(z)` 和变换后的 log probability，使用稳定 Jacobian、当前策略有界熵及精确 latent Normal KL。3844D actor / 371D critic / 24D 动作保持；新 actor 仅移除输出层的参数无关 Tanh，由分布执行一次 squash。CPU float32 rollout 保存 latent，微批按样本数加权，每个逻辑 minibatch 只执行一次梯度裁剪与 Adam，正常配置仍为每轮 40 次 Adam 更新。新指标与原三项 loss 分开，记录动作饱和、各维均值/标准差、概率一致性、KL、熵和裁剪前梯度范数。

隔离服务器上新增及相关旧测试 **107 项通过**（根代理独立复跑 10.40s），规格审查与代码质量审查均通过。命令、边界与限制见 [T1 核心验证记录](docs/superpowers/reports/2026-11-06-lift-repair-t1-core-validation.md)。全仓库 CPU 测试收集得到 4831 项，另有 11 个旧测试文件因缺少 `open3d` / `elastica` / `co_sim` 无法加载，尚不能宣称全仓库通过。

训练入口、v3 合同及显式派生恢复仍在集成；默认旧路径未切换，GPU3 原训练继续运行。尚未运行修复版 Isaac 或获得修复版抓取成功；固定模型原生回归等待现有训练自然完成第 2048 次更新。保持原物理参数、0.2 Nm 执行器限制与原生成功判据，尚未修改奖励、控制或 Actor 反馈输入。

## 2026-11-06 — 独立 Lift 修复分支：T0 与硬件审计

本工作分支 `codex/dh116-lift-repair-20261106` 从 `11-5` 的 b42f1aa 派生。按用户实施文档，先冻结历史、审计真实传感来源，再升级PPO有界动作合同；当前GPU3训练继续运行，原始源码、物理参数、执行器限制和旧checkpoint保持。Actor仍为3844D、Critic371D；20D仅预留有来源/单位/时间/有效性的可选接口，不将native接触或成功监视器追加到Student。

T0已冻结第1439次完整PPO更新：47644672 transitions、37680 episodes、原生successes0，checkpoint SHA256为 `d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9`，80个tensor在CPU上均有限。完整策略链日志5+124+1310行、原始Git源码和运行服务器1207个源码/资源文件均保留校验副本，本机副本设为只读。第269/482/1328轮权重未发现，不能由日志补造。

[硬件反馈审计](docs/superpowers/reports/2026-11-06-dh116-hardware-feedback-audit.md)已交付：厂家SDK有部分读数能力，但当前项目未证实实机数据接通；触觉/腕F/T/独立从动角均未确认，PhysX数据只属于仿真通路。设计与执行顺序见[设计](docs/superpowers/specs/2026-11-06-lift-repair-design.md)、[P0计划](docs/superpowers/plans/2026-11-06-lift-repair-p0.md)。

本阶段在隔离服务器目录实际执行以下CPU基线，81项通过（13.63s）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
 tests/test_vision_student.py tests/test_vision_ppo.py tests/test_vision_training.py tests/test_vision_resume.py \
 tests/test_vision_palm_lift.py tests/test_palm_lift_summary.py tests/test_palm_lift_telemetry.py -q --tb=short
```

T1独立Squashed Gaussian合同正在实现；尚未运行修复版Isaac或宣称抓取成功。原生评估须等当前训练自然完成2048后串行执行。后续奖励逐因素消融、反馈接入、双手控制和重新训练均按验证门槛推进。以下为此前已发布阶段的记录。

## 2026-11-06 — `11-5`：定期高度汇报与当前视觉训练进度

定期监控现在每次成功连接后都会汇报二指、三指的本轮物体净抬升、全视觉续训链历史最高值与 PPO 轮次，同时保留左右手掌的本轮实际位移及各自历史最高值。正常推进也发送高度表；单次 SSH 失败继续等下次确认，训练状态检查仍只读。字段、历史边界和汇报规则见[当前视觉训练监控说明](docs/dh116_visual_training_monitor.md)。定时任务保存在本机 Codex 中，仓库记录规则，拉取代码本身不会创建定时任务。

截至 **2026-11-06 03:21:14 UTC**，已完成 **1328 / 2048** 次 PPO 更新，累计 **44007424** transitions、**34985** episodes，最慢环境 **27** 回合，原生 PPO successes 仍为 **0**。1024 个固定相机视觉环境继续运行，二指、三指各 512，3844D 感知 actor / 371D 特权 critic；当前 GPU3 训练进程和 tmux 均在，采样正在推进。

| 抓法 | 第 1328 轮物体净抬升峰值 | 全视觉链最高净抬升 | 当轮左掌上移峰值 | 当轮右掌上移峰值 |
|---|---:|---:|---:|---:|
| 二指 | 4.524309 mm | 26.163260 mm（第 269 轮） | 0.122219 mm | 0.057757 mm |
| 三指 | 2.919395 mm | 31.813391 mm（第 482 轮） | 0.616342 mm | 9.403050 mm |

物体净抬升排除支架预抬的 60 mm；三指历史越过 30 mm 只达到高度门槛，真实成功仍要求原生完整成功判据。左右掌峰值可来自不同环境与时刻，手掌历史从第 130 轮开始记录。本轮发布核对运行源码 SHA256（699 个运行源文件一致）、完整迭代日志、相机/手掌字段和 Git diff；上一轮 1429 项 CPU 回归结果保留在原验证报告中。本次同步监控规则及进度文档，继续现有训练至第 2048 次完整 PPO 更新，真实稳定抓取仍待验证。以下为此前阶段的记录。

## 2026-11-05 — `11-5`：实测左右手掌上移与物体净抬升

新增训练诊断：分别记录二指、三指组的物体净抬升及左、右手掌实际上移量。手掌位移使用原生物理 body-link 的实测世界 Z，减去该环境本回合物理 reset 后的实测 Z；逐物理子步采集，终止前的峰值不会被自动 reset 清除。日志同时提供每手峰值、最后有效样本的组均值、峰值环境 ID 和有效环境数；左右峰值可以来自不同环境与时刻。负值表示向下移动，缺失值保持 null，旧日志不能补算为零。

已保存第 129 次完整 PPO 更新的视觉断点，并从同一 1024 环境配置严格恢复；只是重载训练进程以加入诊断。第 130、131 次完整更新实测 **39.77 / 44.61 s**，累计 4784128 transitions，2669 episodes，最慢环境 2 回合，原生 successes=0。两组各 512 环境的左右掌读数全部有效，图像/物体有效率均为 100%。第 131 轮二指物体/左掌/右掌峰值为 **0.001228 / 0.502139 / 0.542551 mm**，三指为 **0.170358 / 0.569493 / 0.861406 mm**。它们是独立测量峰值，不表示同一对手共同完成了抓取。

TensorBoard 新增 `lift_mm/{抓法}/rod_peak`、`left_palm_peak`、`right_palm_peak` 与左右 `*_last_mean` 曲线，单位直接显示毫米。`monitor.json`、`iterations.jsonl` 同步保留米制字段与测量定义。诊断不进入 3844D 感知 actor、371D critic 输入、奖励或断点合同；原生成功判据仍为物体净升 30 mm、连续保持 5 s、固定锚滑移不超过 0.5 mm，当前尚未出现真实 PPO 成功。

本次相关 CPU 回归 **1429 项及 83 个 subtests 通过**，CPU 模式跳过 2 项 GPU 测试。原生恢复、断点有限性、Adam 延续、实际执行命令和限制见[左右掌抬升验证报告](docs/superpowers/reports/2026-11-05-palm-lift-telemetry-validation.md)。当前 run 为 `/mnt/mydisk/xk/dh116/runs/dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z`，代码为 `/mnt/mydisk/xk/projects/dh116-palm-lift-20261105`；保持二指/三指各 512、224×96 固定 RGB-D 与 support_deficit，继续至累计第 2048 次 PPO 更新。以下为此前阶段的验证记录。

## 2026-11-05 — `11-5`：1024 固定相机，优先训练吞吐

按最新要求将运行改为 1024 环境（二指、三指各 512），每环境一台固定相机，一次采集覆盖全部环境。224×96 图像保留原水平视场并减少上下背景；320×240 在 2048/1024 相机下均出现显存分配失败，160×120 虽可容纳但细杆有效点不足。最终配置保持原像素反投影、3×3 深度筛选和 3 mm 初始几何审计门槛，1024 环境全部通过，初始每环境 55–80 个有效点，最差环境 P95 距离 0.338320 mm。

从原 4096 视觉策略第 5 次更新显式缩容续训，保留模型、19 组 Adam 状态、LR、RNG、绝对更新和累计采样步数，并按两种抓法分别投影环境计数。新增第 6、7、8 次完整 PPO 更新用时 **39.29 / 42.38 / 40.22 s**，每步实际 1 次 render、1 个采集批次；原方案约 400 s/更新。两者环境数和分辨率不同，不能把约十倍的每轮提速当作相同样本数下的纯算法加速。新配置约 800 transitions/s，原 4096 容量测试约 309 transitions/s。

1 秒间隔采样记录的显存峰值为 21776 MiB（21.266 GiB）。第 8 轮断点的 80 个 tensor 全部有限，Adam 各组 step=320，累计 753664 条 transition；图像/物体有效率均为 100%，原生 PPO successes 仍为 0。最终相关回归 **1194 项与 83 个 subtests 通过**，CPU 模式跳过 2 项 GPU 测试。实现、失败容量档位、命令和限制见[本轮验证报告](docs/superpowers/reports/2026-11-05-fixed-vision-throughput-validation.md)。

当前 run 为 `/mnt/mydisk/xk/dh116/runs/dh116_gpu3_fixed1024_224x96_parse_mixed2048_20261105T104905Z`，代码隔离目录为 `/mnt/mydisk/xk/projects/dh116-vision-fixed-20261105`，继续训练至累计第 2048 次 PPO 更新。使用原生 USD 逐环境加载及显式碰撞组，绕开本机观测到的 PhysX 快速复制初始化等待/崩溃。旧模型和运行原件保留；真实抓取与两组净抬升继续独立监控。启动和显式缩容说明见[视觉模式文档](docs/dh116_vision_student.md)。

## 2026-11-05 — 历史容量验证：4096 环境视觉 Student

视觉 actor 使用前方俯视 RGB-D 相机的真实表面点，经过相机→世界→左右掌心坐标变换后，与编码器、腕部 FK、已发动作和传感时间/有效性组成四帧 3844D 输入。371D 特权状态供训练 critic 使用；actor 不读取物体真值、接触力或任务成功标志。当前没有教师动作蒸馏，旧示范缺少同步相机标签。

4096 个物理环境中，2048 个二指与 2048 个三指始终共同采样、共享一个策略。128 台真实渲染相机分批覆盖全部环境；采用 320×240、AA Off、Kit 32 工作线程。CPU float32 pinned rollout、PPO 微批梯度累积和精确 CAD 高度分块控制显存，保留每轮 32 步、5 epochs × 8 逻辑 minibatches、40 次 Adam 更新。详细输入合同与启动命令见[视觉模式说明](docs/dh116_vision_student.md)。

本次原生容量验证结果：完成两次完整 32 步 PPO 更新，用时 423.29 / 424.65 s，共 262144 条 transition；两秒间隔的显存采样峰值为 19058 MiB（18.611 GiB），无 OOM，训练已继续第三轮。全部 4096 环境初始点云审计通过，每个环境 128 个有效点，最差环境 P95 表面误差为 0.390714 mm。最终相关回归通过 1132 项测试和 83 个 subtests，2 项 GPU 测试在 CPU 回归中跳过；独立 GPU 数值等价验证另见[本次验证报告](docs/superpowers/reports/2026-11-05-vision4096-validation.md)。

当前视觉策略从零开始，运行目录为 `/mnt/mydisk/xk/dh116/runs/dh116_gpu3_vision4096_all_kit32_mixed2048_20261105T094038Z`，目标是累计完成第 2048 次 PPO 更新。原生 PPO 成功数仍为 0，短期容量验证不能说明已学会可靠抓取。成功沿用净升 30 mm、连续保持 5 s、固定锚滑移不超过 0.5 mm；二指和三指净抬升分别记录，历史教师和旧特权策略结果不并入本视觉策略。

训练代码位于服务器隔离目录 `/mnt/mydisk/xk/projects/dh116-vision4096-20261105`。`monitor.json` 记录完成的 PPO 更新，`collection_status.json` 记录尚在采样的进度，`vision_student_ppo.pt` 原子保存断点。下一步继续本 run 的 2048 次更新并监测原生成功与两种抓法的净抬升。模型、相机捕获和原始运行日志保留在服务器，未随源码上传。

以下为历史阶段记录。

## 2026-11-04 — `11-3` 从零训练启动器快照

归档当前训练使用的 `scripts/fresh_multistyle.py`。该脚本校验现有多抓法训练入口的 SHA256，允许随机初始化，并拒绝从命令行载入旧断点或 Student actor。快照保留部署主机上的 `/root/hand_reward_11_3_validation` 路径，使用前需有对应的训练源码与环境。

本次读取的 Isaac 原生训练状态为二指第 983 次 PPO 更新、累计 85,753 个 episode，最慢的环境完成 17 个，真实 Student PPO 成功数为 0。该 run 从零启动，使用 4096 环境、371D 观测和 `support_deficit` 奖励配置，预算为每环境累计 2048 个 episode。每个抓法块以所有环境均完成至少 128 个回合为切换条件。训练仍在进行，下一步继续监测首次真实成功和抓法切换。

本次离线核对确认 7 个训练代码与配置文件已与 GitHub 基线逐字节一致。启动器通过 Python 编译检查，`--help` 返回 0，`--resume unused.pt` 与 `--student-actor unused.pt` 均按预期退出 1。以下定向 CPU 回归通过 62 项，本次发布验证未启动额外的 Isaac 实例。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/conda/envs/hand/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_contact_geometry.py tests/test_dh116_policy_control.py \
  tests/test_surface_contract.py tests/test_surface_reset_resume.py \
  tests/test_surface_strict_training_integration.py
```

## 2026-11-03 — `11-3` 二指与三指续训

从第 902 轮的只读断点派生 64 环境续训分支，继承模型、PPO 与 BC 优化器、随机数状态和累计步数，回合统计按新批次单独起算。预检完成二指第 903 轮、三指第 904 轮，两次均为原生 PPO 更新，原生成功计数为 0。修复已验证 reset 家族之间切换时的 371D 合同摘要更新。执行 `python -m pytest -q tests/test_surface_reset_resume.py tests/test_surface_contract.py tests/test_surface_strict_training_integration.py`，36 个测试通过。原第 902 轮断点的 SHA256 校验通过。

后续使用已有 `support_deficit` 奖励配置，串行从第 904 轮续训，目标为二指第 968 轮、三指第 1032 轮。运行目录为 `/root/hand/runs/deployment/11_3_ppo_64env_multistyle_20261103T044735Z`，后续学习效果以原生成功为准。

## 2026-11-03 — `11-3` 奖励边界与完整回合验证

原生成功的最后一个有效物理子步可以发放当步达到的 Hold 里程碑。失败和已终止的子步不发放，重置会清除已发放标记。滑移使用独立的固定锚，在测量有效且对向指腹接触持续时读数，不受 Hold 的 0.03 m/s 速度门槛限制。失去接触或测量无效时丢弃锚并停止计分。承托因子计入向下摩擦，只有可归属到有效指腹的向上摩擦才获计分，结果受原生整手有符号净力约束。已有 371D 观测与 baseline 使用原来计算，30 mm、5 s、0.5 mm 原生成功判据保持原定义。

同一第 902 轮策略在两种既有 reset 家族中确定性运行。每种抓法的 baseline、reference 和八个单因素 Profile 在同一原生轨迹上独立计分，记录每个环境的首回合与每步奖励。完整验证入口为 `scripts/audit_dh116_surface_reward_episodes.py`。原生证据位于 `/root/hand/runs/deployment/11_3_reward_full_episodes_20261103`。

同一组边界用例从 5 失败、13 通过变为 18 通过。使用下方相同的 `python -m pytest -q` 定向命令，最新结果为 296 个测试和 24 个子测试通过。二指和三指各完成两个首回合，原生成功均为 0/2。二指最高净抬升 29.49 mm、最大有效固定锚漂移 1.013 mm，三指分别为 11.43 mm 和 0.840 mm。四个首回合的最长原生 Hold 都是 0 s。两种抓法前 180 步的观测及动作与旧版逐值一致。旧断点的 SHA256 前后一致。此轮结果用于验证计分和固定策略表现，PPO 学习效果仍待验证。

## 2026-11-03 — `11-2` 奖励隔离与原生验证

`baseline` 保留历史计分。`v3_reference` 使用相同计分作为有合同记录的实验起点，每个 `v3_*_only` 分支只改变一个奖励因素。高度组用合格物理高度高水位替换 `dex_lift` 和旧高度里程碑计分，旧观测状态照常更新。滑移按 60 Hz 完成窗口的有效固定锚漂移计分，Hold 失效的窗口仍可扣分。断触在两个完整窗口后触发，重新稳定 100 ms 后可再次触发。Early Hold 每回合最多 10 分，承托不足使用可归属的有符号接触力及原生物重，平滑项使用关节、手腕实际增量及安全投影后的命令偏差。旧递进 Profile 标记为 cumulative。

定向回归 293 个测试和 24 个子测试通过。以下命令在隔离的 Isaac Python 环境执行。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. /opt/conda/envs/hand/bin/python -m pytest -q \
  tests/test_surface_reward_runtime.py tests/test_surface_baseline_compatibility.py \
  tests/test_surface_stable_hold_rewards.py tests/test_surface_success.py \
  tests/test_surface_contract.py tests/test_surface_feedback.py tests/test_surface_checkpoint.py \
  tests/test_surface_student_evaluation.py tests/test_surface_exploration.py \
  tests/test_free_wrist_grasp.py tests/test_surface_strict_training_integration.py
```

离线重查三指 Teacher 的 10,176 个 Isaac 原始子步。两次原生成功的跟踪输出逐值一致，三次实测越过 0.5 mm 的滑移窗口均被扣分。

第 902 轮纯 Student 原生对照用相同权重和种子，每组 2 环境、180 控制步。二指比较 reference、高度和平滑，三指比较 reference、滑移和承托。各抓法内的观测、动作和杆轨迹逐值一致。两种抓法净抬升均低于 0.001 mm，连续 Hold 为 0 s，观察期内完成回合和成功均为 0。原生证据在 `/root/hand/runs/deployment/11_2_reward_native_20261103`，各组独立记录配置、断点及合同 SHA。实际物理、24D 动作及 351D/371D 观测合同保持原定义，真实成功仍要求净升 30 mm、连续保持 5 s、固定锚相对滑移不超过 0.5 mm。短回放确认奖励接入，完整回合成功率与训练收益仍待小规模 PPO 对照。本轮未启动 4096 环境长训练。

## 2026-11-02 — `11-2`：371D contract、派生checkpoint与确定性A/B验证

本阶段先锁定24D动作合同：12个关节速度、6个世界平移速度、6个本体旋转速度，保留原尺度、60 Hz控制/480 Hz物理执行和原执行器限制。新增20维真实物理反馈时，371D只追加到原351D输入尾部；反馈按控制周期完成时间窗，遇到无效或未对齐数据直接让rollout失效。原351D checkpoint `tensor_ppo.pt` 的SHA256为`55d2a041cab933542f7c8f49b060e30d7f009467937396da1c4566c0b471c58a`，派生371D checkpoint单独保存，SHA256为`e096b0f4d76e84db281674b9cfd91f3d3770ca09f8cd70cc5bda0a4a70a30033`；原文件未覆盖，actor/critic、PPO/Adam状态、RNG和计数均保留。

新主机原生确定性A/B各运行2环境、1800控制步。A使用原351D checkpoint并以只读反馈采集telemetry，B使用独立371D派生checkpoint并把20维反馈接入Student输入。两边nominal/executed 24D动作逐值一致，371D反馈形状为`(1800,2,20)`且全有限；351D观测前缀与B的前351维数值一致（最大误差`4.44e-16`）。两边均为0/2成功、hold 0s，说明这一步证明了输入扩展和动作合同保持，不能当作Student已经稳定抓取的证据。证据目录为远端`/root/hand/runs/deployment/11_2_contract/evidence/student_a_351_v5`和`student_b_371_v1`。

371D Teacher又在普通TensorEnv独立回放通过2/2原生成功，1175控制步；与351D示范动作逐值一致，输入前缀最大误差2.22e-16。示范合同SHA最终化另存派生文件，原生样本和原件保留；Teacher结果不算Student成功。

代码已加入逐项奖励消融配置：`baseline`、`lift_progress`、`early_hold`、`contact_break`、`slip_relative`、`support_deficit`。动作平滑继续沿用原有奖励和硬安全约束，不重复添加等价平方项。独立审查后修正奖励注册、baseline合同兼容、BC保留、真实物重/摩擦支撑、滑移、合同SHA和DAgger状态绑定；每次只替换一个奖励因子，原BC Adam保留。DAgger原生入口读取Student失败前真实状态和Jacobian，仅输出24D纠正候选；候选须另经原生纠正验证才可进入辅助BC。本轮直接相关回归190项及24个参数化子测试通过。完整状态单轮续训371→372已完成，新checkpoint全部tensor有限、PPO/BC Adam均更新，累计28147个episode、success仍0。第372轮确定性Student/纠正候选采集另行验证，尚未进入长训练或完成奖励原生消融；见[本次证据、命令与限制](docs/superpowers/reports/2026-11-02-371d-contract-validation.md)。

## 2026-11-01 — `11-1`：teacher → student 迁移修复与续训

已确认teacher在新服仍可2/2成功；原475轮student去噪对照16环境均抬过30mm，但最长只连续保持0.25s，0/16成功。原BC的低MSE不代表能闭环抓稳，不能把教师成功写成student成功。

本次补齐真实原生诊断、独立示范辅助Adam、保持样本加权、小噪声探索，以及仅对有效指腹抓取给lift奖励、连续保持奖励和真实滑移软惩罚。实际成功仍为净升30mm、5s、固定锚漂移≤0.5mm，物体和执行器物理参数不变。401维接触历史仅为可选实验，正式续训保留351维。

原4096环境的第475轮完整断点、53533个新episode和每环境最低8回合已保存，显式派生断点保留actor/critic、已有PPO状态及回合计数，目标仍每环境累计2048个新episode，二指/三指跨块继续。已实际恢复4096环境并完成第476、477轮PPO更新，全部checkpoint tensor有限；相关回归1373项、85个subtests通过，当前native student successes仍为0。TensorBoard只读当前run；首次真实PPO success另行监控。实现、原生对照、恢复过程与全库测试限制详见[修复记录](docs/superpowers/reports/2026-11-01-student-transfer-repair.md)。下文为历史进度，不能当作当前运行状态。

## 2026-10-31 — 已补齐二指原生成功teacher

每手拇指+食指的AM示范已独立通过：真实净升31.536 mm，在30 mm以上保持5.017 s，最大相对漂移0.00367 mm，其它三指有效力为0。原三指成功示范同时保留，原物理与执行器限制不变。普通TensorEnv补采二指/三指各2个实际成功回合，共4891组351/24样本；按抓法均衡预训练后，64环境跨初态4次PPO更新已通过。AQ正在各追加64次更新，尚未证明student独立lift或MPC实际收益。见[多抓法记录](docs/superpowers/reports/2026-10-31-multistyle-teacher-student-validation.md)。

## 2026-10-31 — PPO入口已实跑验证

16环境完成2次PPO更新，共512条transition，模型、优化器和TensorBoard保存正常。surface模式由策略连续探索，分面/分指均力保留为软奖励，成功按真实净升30 mm及5 s防滑保持判定；回合预算30 s。AK原生示范逐帧回放得到相同5.017 s成功窗口。当前从静态预抓姿开始训练，没有加载教师策略；短验证成功数0，不代表已经学会lift，也没有启动长训练。见[PPO验证记录](docs/superpowers/reports/2026-10-31-ppo-readiness-validation.md)。

## 2026-10-31 — 已找到真实lift成功样例（31.5 mm / 5.02 s）

原生双手教师AK第二例通过独立复核：圆杆最高净升31.534 mm，在30 mm以上连续稳定保持5.017 s，保持窗口最大相对滑移0.00263 mm；普通碰撞载荷、adapter异常和reset均为0。保持时左右总法向夹持力约1.807/2.664 N，尚未都达到3 N参考。只保持手指关节指令，圆杆仍由真实接触/摩擦承力，没有物体固定、传送或外加抬升力。该次找到一个成功样例，另一个测试参数例失败，不代表通用成功率。

最新要求：最终以真实lift成功为准，分指力允许偏差。每手3 N、每面1.5 N及同面分指均分保留为控制参考，0.3 N误差另存诊断，不再阻止提升或否决成功。法向目标不含摩擦力。成功仍检查相对支撑初态净升30 mm、有效对向指腹承力、连续5 s稳定保持及保持窗口内相对滑移≤0.5 mm；保留原执行器、原生深度和异常隔离，不把支架架高或手腕运动算作杆已抬起。

用户已明确力误差按每个60 Hz控制周期内8个480 Hz原生子步的真实平均力记录，不另用EMA冒充测量。瞬时力保留作诊断，实际高度、滑移、碰撞与深度仍逐物理步检查。AA曾真实净升约6.98 mm后滑落；AB按当时严格均值口径仅连续约0.217 s满足分力误差，提升后仍掉落，尚无成功教师。

新增显式教师选项`--pad-centering --batched-contacts --batched-native-sensors`。耗时采样定位传感器重复读回/解包与逐形状几何检查；两秒原生对照68.09→48.50 s，原生物理和控制记录逐值一致、reset 0，尚非实时。采样缓存修复PhysX contact/friction共用count/start造成的索引覆盖。三指源候选仍只是静态可行，不把性能优化当作抓取成功。

成功入口为`scripts/probe_dh116_fixed_force_lift.py`的显式`--retain-airborne-pose`，配合三指初态和原生批量读回。成功原始轨迹、独立审计与`successful_teacher_demo_env1.pt`保存在`/root/hand/runs/native_pose_carry_20261031_ak/`；这是原生物理示范数据，不是已训练策略模型。正式PPO模式仍连续探索，本轮没有启动长训练或推送GitHub。坐标/Jacobian核查、中心抓姿搜索、复现命令和逐次原生结果见[继续拾起验证](docs/superpowers/reports/2026-10-31-native-pickup-continuation.md)。以下为同日较早记录。

## 2026-10-31 — 有效指腹、按抓取面分力与端支架自由探索

新增显式24维 `--surface-force-sharing` 模式：所有手指、抓点和双腕姿态可探索，去除新模式固定目标力区间，双手本体仍12主动+10从动。每手先给两个对向抓取面各50%，再在各面实际承力指间均分；1对2为50%/25%/25%。按各指源CAD局部指腹和原生法向筛选，普通碰撞单列，不能冒充有效抓取或获得正向指腹受力奖励。

允许显式60 mm端支架，中部净跨度476 mm；净提升从支架初始杆底面计，架高本身不算lift。静态重力/力矩LP筛出4组候选，只是预测可行，尚无合格动态教师。最终定向回归905项与8个subtests通过；原生短PPO完成预算，成功0。修复将有限负分离项误判非法导致的reset，并将普通碰撞排除于抓取锚点/正向奖励。最终L两环境各2 s实测reset和adapter异常均0，圆杆最大净升仅0.0101 mm，仍未形成双手有效对向抓取。完整验证与限制见[本轮记录](docs/superpowers/reports/2026-10-31-surface-sharing-validation.md)。

可用 `DH116_NUM_THREADS=4 bash scripts/smoke_dh116_surface_grasp.sh <唯一标签>` 复现2环境×1更新冒烟；没有启动长训练或推送GitHub。新模式需自己的24维动作合同，不能静默沿用旧26维模型。质量、重力、原材料和0.2 Nm/0.3 rad/s执行器限制保留。

## 2026-10-30 — 继续实测2–3 N多指抓取（未成功，作用方向已核对）

本轮完成逐物理步受力核对、960 Hz/TGS/刚性mimic诊断、原生指垫柔度控制补偿及新预抓姿搜索。动量账目一致；补偿改善部分静态力偏差，尚未消除脉冲，普通PPO默认物理设置未改。501项定向回归与2个subtests通过。

最后K2实测4环境2745步：3组已真实执行lift，双腕最高上移约0.32–0.49 mm，杆底面最高仅0.00501 mm，随后因相对偏差暂停。实际法向竖直合力仍向下1.2–2.4 N，向上摩擦不足抵消1.18 N重力；每手2–3 N也未稳定跟上。reset/成功/教师/PPO均0。下一步优先筛选能提供向上支撑的实际抓点并完善接触纠偏和重试。

测试已结束，原始数据及源码冻结保留并另存本地；TensorBoard仍为`http://127.0.0.1:16006/`。新补偿显式开关默认关闭，探索提升harness没有接入正式训练。本轮修改尚未推送GitHub。复现命令、全部对照、原始与均值区别及限制见[本轮详细记录](docs/superpowers/reports/2026-10-30-force23-momentum-compliance-validation.md)。以下是同日较早记录。

## 2026-10-30 — 新服务器与每手总力2–3 N测试（尚未抓稳）

已迁移至 `a21693232428085248691341:/root/hand`，现有hand环境和资产通过核验，项目TensorBoard服务已恢复，本地入口为 `http://127.0.0.1:16006/`。迁移前修改另存归档及stash，运行数据保留。

用户确认本轮2–3 N是每手总法向力。增加已承力指的预算平摊、独立寻触、包含手掌的原始总力窗口与逐指最大/最小力比≤1.5检查；滑移撤销排队上升目标。保持原电机及物理参数，未启动PPO。实际A/C/D均未抓稳或完成提升，成功0；已确认快速接触/联动波动，增至64速度迭代未一致改善，均值接近2.5 N不能算稳定。最终审查修复已部署，483项定向回归通过，修复后的完整物理验收仍未完成。详见[本轮记录](docs/superpowers/reports/2026-10-30-force23-new-host-validation.md)。本轮修改在本地和新服务器，GitHub既有`10-30`基线不代表这些新修改已上传。

以下为此前阶段记录，涉及“不固定总力”的描述已由本轮2–3 N测试要求覆盖。

## 10-30 分支 — 多指均衡与防滑提升（当前开发快照，尚未成功）

按最新要求允许所有手指参与并较均衡分担，增加原始承力稳定窗口、杆相对双手的固定参考和防滑提升保持诊断。双手仍为12主动+10从动，不固定总夹持力，普通PPO入口保留。

保存4组多指几何预抓姿及搜索脚本。214项定向回归、搜索自测和编译检查通过；实际4环境×6s短测均未达到稳定条件，未进入提升，成功/教师均为0。当前代码包含仍待完善的滑移时手腕目标撤销及独立5s验收，不将其描述为已完成的稳定抓取控制。

入口为 `scripts/run_dh116_multifinger_lift.py`，配置为 `configs/dh116_training/multifinger_pregrasp_v1.json`；完整复现命令、结果和未完成项见[当前验证记录](docs/superpowers/reports/2026-10-30-multifinger-nonslip-lift-validation.md)。以下为此前阶段记录。

## 2026-10-30 — 持续承力复核（接触改善，实际提升仍失败）

继续在右手式预抓姿上实测。源几何退让判据与原生 manifold 间隙不一致；改用原生深度并按最近8个物理步的最小力判断持续接触后，静止四指同时承力从约2ms增加到3.09s，但右拇指仍有明显力波动，尚未抓稳。凸包简化只能解释部分几何差异，完整碰撞等价性仍待核查。

实际提升I中，双腕均上移46.92mm，圆杆底面最高仅离桌0.00688mm，随后失联；reset=0、成功=0、教师=0。lift指令已执行，当前夹持没有留住圆杆。新增显式`--depth-feedback native --contact-statistic minimum`和`--hold-only`诊断选项，默认仍保留旧source/mean行为，没有固定目标力或新的lift门槛，未启动PPO。112项定向测试通过；原始数据、力曲线、独立审计和运行源码保留。详见[持续承力与提升验证](docs/superpowers/reports/2026-10-30-pinch-force-stability-validation.md)。

## 2026-10-30 — 双手参照右手末节对捏（预抓姿已适配，尚未稳定提升）

按用户截图，以右手实际接触几何适配双手各自的关节和手掌朝向。新增`right_pinch_pregrasp_v1.json`：独立CPU采样检查手物穿透、自碰均0，最小桌面间隙0.459mm，双手AABB在Y方向分离310mm。左右手角度分别拟合，没有直接复制关节角或镜像四元数。

实际固定微闭合A及逐指寻触B均未抬起圆杆。已保存原始状态与失败结果，合格教师仍为0。`run_dh116_pinch_contact_seek.py`仅为显式诊断：缺接触的拇指/食指补闭合，已接触保持，源顶点采样深度过大时退让；它不保证完整连续碰撞深度，也不代表稳定抓取控制已修复。没有目标力或额外力矩，普通PPO不变，未启动训练。

该阶段42文件回归1043项通过。此前Isaac窗口曾展示该轮记录回放；回放不算新的实时物理验证，目前该窗口已关闭。命令、原始结果和限制见[右手参考验证记录](docs/superpowers/reports/2026-10-30-right-pinch-reference-validation.md)。

## 2026-10-30 — 当前要求：目标力不限，以真实 lift 为准

用户最新要求覆盖此前的目标力跟踪和两指分配门槛。新增 `scripts/run_dh116_grip_lift.py`：在既有安全初始位姿上搜索闭合幅度，保持闭合关节目标，通过物理手腕驱动提升；不使用目标力、力误差、四指接触、力均分或轴下位置作为 lift 门槛。正常有限联动偏差不打断运动，实际关节角度、速度和0.2Nm力矩上限保留。

高度/保持验收使用原重力/桌面条件下实测圆杆底部离桌至少30mm、真实手物接触、线速度小于0.03m/s并连续保持0.5s；中途达到与结束保持分别记录。逐480Hz保存原始状态、接触、故障和reset证据，再独立重放验收。41文件940项回归通过。

实际40组搜索A及同布局重跑C均有1组达到高度/保持条件：最高49.6405mm、在30mm以上连续4.4542s，最后1s左/右法向力均值4.3723/1.4632N。原始物体、关节、接触与故障字段逐步相同；单环境原点重跑B失败。独立源凸包检查发现闭合过程最大穿透5.119mm（保持窗口约0.900mm），所以**合格教师仍为0**，不自动用于PPO。新入口导出的lift记录必须继续接受几何审查。

`configs/dh116_training/grip_lift_pregrasp_v1.json`保存四个可复现预抓姿；默认40组布局用于诊断上述现象，不是已通过硬件标定的成功控制器。完整结果和限制见[验证记录](docs/superpowers/reports/2026-10-30-surface-teacher-validation.md)。下面各节为历史模式及诊断记录。

## 2026-10-30 — 多点搜索、持续力跟踪与捏紧保持（实现及实际诊断）

- 双手12主动+10从动=22手指关节；新增原生USD多点几何、穿透/自碰检查及多抓姿搜索。全指节搜索经CPU复核得到192个几何组合；几何可行不等于稳定抓取。
- 夹持力在2–12N/手内探索，逐480Hz物理步持续反馈；法向Z/摩擦Z/总Fz分开记录。修正接触模型刷新、失联旧力矩和PI请求上限；前馈与原驱动共享0.2Nm电机预算。
- 按最后要求新增`--pinch-hold`：先在2–3N内选取初始夹持力并保持，手腕目标不变，仅用原关节驱动闭合补力，关闭额外接触前馈及自动lift。允许手指自然关节弧线，原PPO自由探索入口不变。
- 保持模式继续由原生mimic约束联动，取消有限联动误差造成的拇指/食指主动目标退回；桌面软件间隙与初始审计统一为0.3mm，必要避让仍保留。补力按真实10Nm/rad关节驱动和接触力臂缩放，反馈用上一物理区间均值，原始子步仍完整验收。实际执行器限幅保留，40文件最终回归863项通过。
- 实际B/C/D均未抬起圆杆，D原始子步最高仅0.011153mm；合格教师为0。本次未启动长PPO。最终保持测试、回归、原始证据和复现命令见[本次验证记录](docs/superpowers/reports/2026-10-30-surface-teacher-validation.md)。
- 教师发布需实际离桌10mm/0.3s、四指承力/平衡及逐子步力跟踪，误差<=max(0.1N,5%指令)；带前馈的数据必须配套相同低层执行器，不能直接作为普通26维执行器的BC样本。

## 2026-10-30 — 自由姿态抓取控制（实现及实际试验，尚未抬起圆杆）

- 已实现显式 `--free-wrist-grasp`：手指、抓点和手掌朝向可变；双手本体仍是12主动+10从动=22手指自由度。26是策略命令维数，另含12个外部手腕运动指令和2个法向目标参数，不是26个手指主动自由度。
- 新增六轴物理手腕、完整积分状态观测、真实几何/逐指承力反馈及实际离桌奖励；保留有符号向上奖/向下罚、连续运动与异常隔离。旧20维及手动入口保留；新合同不能直接加载旧模型。
- 本次最终定向回归33文件、668 passed in 12.64s；原生Isaac验证实际旋转与选择复位，随后完成1次PPO更新。3组短PPO分别完成4/8/8次更新，另2组自由位姿接触搜索各360步。
- 最高实际离桌仅0.133572mm，10mm保持和正式成功均为0；存在失载及瞬时过载，不能称为抓取成功。手指力矩、物体物理参数未改；外部手腕参数未经硬件标定。
- DexGrasp-Anything已核对论文和代码，当前仅调研；可借鉴多点指腹几何、穿透代价及多抓姿筛选，不把无重力扰动测试当成本项目lift成功。未安装其环境、未启动长训练。
- 命令、逐项结果、来源和限制见[自由姿态验证记录](docs/superpowers/reports/2026-10-30-free-wrist-validation.md)。后续仍需解决稳定双手承力与实际提升。

## 2026-10-30 — 持续提升与异常隔离（实现及实际验证）

- 新增 `--continuous-motion`，标准手动及 policy PPO 脚本默认启用。有限联动误差记录并有界纠偏，不再单独触发双手暂停、mimic失败或复位。
- 保留逐指力分配、原始接触读数和现有执行器限幅；无效状态先隔离 reset，完整验证复位及首次复发，无法恢复才停止。手动恢复保留 Run/lift 意图，显式 Reset+Pause 保留。
- 本次最终定向回归：27文件、493 passed（11.66s），含169个新增控制、状态与PPO采集恢复用例；shell语法检查通过。关闭模式的历史语义仍有回归覆盖。
- 实际Isaac：单环境5N/手控制下150控制步未复位，双手完成约10mm上移；圆杆未离桌，不能视为抓取成功。两环境异常读数注入只复位故障环境；额外80物理步的真实CUDA采集验证启动校验、坏动作/状态的两次隔离与完整重采，失败记录完整，权重和优化器未更新；PPO更新0。
- 代码和既有驱动/环境未完成硬件能力标定；后续优先解决夹持和抓取保持。未启动长训练，GraspXL A的25维逐指观测/奖励扩展尚未实施。
- 详细命令、对照结果、回执位置和限制见[验证记录](docs/superpowers/reports/2026-10-30-continuous-motion-validation.md)。

## 2026-10-30 — GraspXL 借鉴方案 A（设计阶段）

用户选择保留自由探索、渐进补齐逐指反馈、有限幅度几何引导和抓取提升期间的持续协调。
设计见 [方案 A](docs/superpowers/specs/2026-10-30-dh116-graspxl-option-a-design.md)，待文档审阅后制定实施计划。

- 已核对：PPO 已有逐指负载；新增重点为逐指真实触点信息、统一受限控制及逐指质量评价，避免重复叠加已有奖励。
- 设计保留：固定掌朝向、每手 2–12 N 法向目标、四指轴下、两指力差 <=50%、双拇指稳定及无阶段自由探索；原向上奖/向下罚和独立物理保护保持。
- 当前实现状态：本次仅新增设计和进度记录；联合闭环仍未通过合格抓取与真实 10 mm 提升验收，不将历史测试计为本方案验证。
- 本次验证：源码/约束核对、git diff --check、占位内容扫描及文档链接检查通过；未运行新的 Isaac 或 PPO。下一步先审阅设计，再完成定向测试、单环境物理验收和最小 PPO 冒烟。
- 本地设计提交不代表已经上传 GitHub、部署新控制器或恢复长训练。

## 10-29 故障恢复快照（2026-10-29 UTC）

当前阶段：保存服务器故障前留在本地的源码与配置。以 `10-28` 的 `bd805c6570eb14fdfcbb3beb8d4afd890c2e9415` 为基线，叠加本地可恢复修改；不是服务器磁盘或虚拟环境的完整镜像。

- 已保存：连续抓取/抬升与奖励、掉落及关节异常 reset、手动 Isaac 控制与四指原始接触诊断、拇指稳定判据、逐指力差控制、实验性的接触位置/力联合闭环、相关测试与环境说明。
- 联合闭环目前仅在诊断入口显式创建，尚未接入默认手动窗口，也未通过稳定抓取及真实 10 mm 物体抬升验收。手掌朝向保持，原始测量与物理保护保留。
- 本次上传验证：73 个 Python 文件 AST 解析、8 个配置 JSON 解析、相对模块与测试辅助模块存在性检查、源文件哈希核对通过；本地没有 Torch/Isaac，服务器不可达，未重跑运行时测试。
- 故障前历史验证：联合控制模块最后 10 项定向测试通过；188 项较早的相关回归通过发生在最后几项审查修复之前，不代表此快照的完整回归结果。物理对照仍失败，不能视为成功抓取。
- 未取得：仅存服务器上的 PPO 权重、TensorBoard 数据、完整 Conda 环境和未同步文件。最后已知 PPO 按用户要求停在第 259 轮，本次没有重启训练。

恢复范围、未完成项和历史物理结果见 [恢复说明](docs/RECOVERY_10-29.md)，逐文件校验见 [恢复清单](docs/recovery-10-29-files.json)。

## 10-28 上传快照（2026-10-28）

当前阶段：策略直控与抓取控制改进已实现并完成部分离线验证；接触反馈探索闭环尚在开发，未完成真实物理抓取抬升验收。本分支是当前代码快照，不是全部测试通过的发布版。

- 改进：包含策略直控动作接口、物理驱动手腕、接触力/抓姿保持与恢复、物理诊断、教师/学生验证及训练观测相关工作。策略直控20维动作采用12关节位置、6手腕位置、2驱动刚度，不能直接续训旧力目标模型。
- 本次上传前实际检查：下列针对性命令结果为 **204 passed, 5 failed in 4.88s**。`git diff --check` 与 `bash -n scripts/python.sh scripts/train_dh116_tensor_ppo.sh` 均通过。未运行完整仓库回归。
- 5项失败均位于 `tests/test_dh116_contact_exploration.py`：4项因 ContactExplorer 尚未实现而导入失败，1项因策略任务尚未实现取消旧稳定等待的即时提升转换而失败；保留这些测试以记录待实现要求。
- 实际Isaac状态：此前新策略接口的固定指令演示手腕上移约2.6–2.9mm，物体未稳定抬起，回合失败复位；不是自主接触探索或抓取成功。本次上传操作未重新启动仿真，历史结果不作为本次验证。
- 下一步：完成接触反馈探索与实测抓稳后直接提升闭环；定位物理步进、接触传感器读取和执行循环耗时，再完成单实例实际验证。OakInk仅进行了源码调查，尚未集成。
- 不上传 runs/、archive/、运行日志、缓存及密钥；本地原件保留。依赖本地归档/运行产物的验证不保证仅凭GitHub克隆即可复现。

```bash
DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_policy_control.py tests/test_dh116_contact_exploration.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_tensor_world.py tests/test_dh116_coordinated_grasp.py tests/test_dh116_lift_grasp_hold.py tests/test_dh116_grasp_shaping.py
```

# DH116 独立工作目录 · 10-25

## 2026-10-28 — 策略直接探索抓取和提升修正（离线候选）

新增 `--grasp-controller policy`，训练入口 `scripts/train_dh116_tensor_ppo.sh` 默认选用；历史协调控制可用 `DH116_GRASP_CONTROLLER=coordinated` 显式选择。新模式在 close/lift 均接受策略动作，不使用保持抓姿覆盖、自动补力、自动提升或普通接触恢复暂停。20维动作明确为12个主动关节位置、左右手各3轴手腕位置、左右手驱动刚度；与历史力目标动作不兼容，旧检查点不能直接续训。

手指目标使用现有驱动0.3 rad/s上限，手腕使用0.05 m/s上限；保留机械范围、严重联动/超力处理、桌面间隙恢复及现有运行时硬保护。参数仍属仿真假设。手腕Z探索范围为复位基准下移4mm至上移50mm；实际位移仍经过物理驱动与保护。物体实际离桌的成功判据未修改。新模式要求 `--palm-exploration-axis xyz`，标准启动脚本已接线；不支持旧教师脚本/旧回放动作混用。

本次相关离线回归 **204 passed in 5.00s**，shell语法与diff空白检查通过。本次尚未启动新模式Isaac或训练，不宣称稳定抓取、提升成功或实际提速。实现计划与验证记录见 [policy-owned-grasp](docs/superpowers/plans/2026-10-28-policy-owned-grasp.md)。

## 2026-10-26 — 修正控制后从零训练

- 用户要求从零开始：停止v154，保留第618轮60751872样本检查点及完整旧目录，另存`runs/control_restart_20261026_v155/previous_tensor_ppo.pt`并核对SHA256。
- v155新建训练目录，4096环境、24步更新、每环境1024回合；沿用协调抓握与物理驱动手腕，奖励及物理保护不变。启动不传resume，training_start实测策略未加载旧检查点、初始iteration/timesteps/episodes/successes/Adam状态数均为0。
- 已完成第2轮实际PPO更新，成功0；后台只读观察器启动，TensorBoard运行名`v155_fresh_coordinated4096`。此处验证启动与更新，不代表已学会抓取。

## 2026-10-26 — 抓取控制联调与物理手腕提升

- 修复固定根位姿更新时“手掌位置变化、物理速度为零”的控制路径：增加三轴物理驱动手腕，显式USD DriveAPI；只在复位时写根位姿，正常提升由驱动执行。启动预检与GPU批量路径共用实测手掌观测及跟踪保护。
- 协调控制改用实测手指位移补偿手掌；政策动作下限不再覆盖受力状态下的位移限制，拇指接近限位时释放对侧食指并暂停移动。手指力矩、奖励及硬物理保护不变；手腕支持参数为仿真假设，不是厂家标定值。
- 本次验证：空手16只手均完成约29.9mm行程，最大跟踪误差约0.103mm；最终8回合抓取探针全部完成、无硬保护故障，但3mm与30mm成功均为0，最大实际离桌0.531mm。不能称为稳定抓取成功。
- 相关CPU回归105通过、2跳过（CUDA测试未运行）；shell语法及diff检查通过。详见[联调结果](docs/superpowers/specs/2026-10-26-integrated-grasp-control-results.md)。
- v154在新目录恢复4096环境、每环境24步更新，从保留的第583轮策略/Adam/计数继续；合同仅增加协调控制与物理手腕。已验证第586轮实际更新，成功0。训练准入与抓取成功分开；继续由PPO学习稳定抓握。后台观察器只记录异常/趋势，不自动修改代码。

## 2026-10-26 — 当前奖励配置从零重训

- 按用户要求停止v133，保留第409轮40206336样本检查点及日志；v134不传resume/initialize-from，全新策略、Adam和计数启动，4096环境、24步更新、每环境1024回合预算。
- 保留受力势差、3/10mm一次性离桌里程碑、close→lift向下−3及持续下压罚，物理参数/保护和156维观测不变。
- 实际training_start验证resume_source为空，初始iteration/timesteps/episodes/successes/optimizer_state_entries全为0；已完成第2轮196608样本，策略/Adam有限，新奖励及切换力事件TensorBoard标签可读，成功0。只读观察器已启动，旧记录未删除。

## 2026-10-26 — 受力方向势函数与实际离桌里程碑

- 用户批准新增两项学习信号，保留切换向下−3、持续下压罚、现有向上/高度/成功奖励与物理保护。新增`--grasp-shaping`仅允许当前close-direct-lift奖励配置。
- 将手对物体有符号竖直合力除以运行时mg，按0.05秒滤波；势函数Phi=2*tanh(filtered_ratio)。每个PPO控制步仅计算一次0.995*Phi(next)-Phi(previous)，与算法gamma相同。成功/失败终止下一势置0，rollout分段不清势，防止通过终止或来回变化增加折扣总收益；这是势差项，终止补偿不能解释为抓取成功奖励。
- 实际物体最低点分别离桌3mm/10mm，lift阶段且实测载荷ready、速度<0.03m/s、滑移≤0.02m持续0.1秒，一次+2/+3；中断稳定重新累计，已领取不重发，复位才清除。最终30mm/0.5秒+25不变，无新增hold阶段。
- 观测追加滤波力势、两项dwell和领取位，151→156。精确单向合同迁移，旧网络列与Adam矩保留，新5列及对应矩补零；实际第402轮检查点CPU加载验证通过。动作仍20维。
- 新增lift_gates遥测，统计lift物理子步中联动/跟踪/载荷/滑移/压力阻塞、实际命令推进与高度上限比例（阻塞可重叠）；不放宽控制条件。TensorBoard新增reward/force_progress、reward/lift_milestone、lift_gates各项；切换瞬时力曲线保留。
- 163项相关CPU/GPU回归通过，覆盖势差终止折扣抵消、60/480Hz滤波/里程碑一致性、选择复位、失败隔离、任务终止快照、151→156权重/Adam零扩展及各提升门槛。shell/diff检查通过。
- v132保留第403轮39616512样本检查点，策略/Adam有限；旧Isaac退出后v133从该检查点继续4096环境。training_start确认第403轮/13项Adam状态保留，迁移报告仅force_progress_lift_milestones_v1及151→156零扩展。实际第406轮39911424样本，首批force_progress产生非零值，lift_milestone仍为0；第405轮检查点策略/Adam有限，TensorBoard新增奖励与门禁/事件标签读回通过。第406轮仅46个lift物理采样中命令推进0，载荷阻塞73.91%、压力阻塞80.43%、联动阻塞13.04%（可重叠）；这是早期小样本诊断，不是完整物理根因结论。物体底部最大0.30000590m，成功0，尚未离桌。独立复核59项CPU通过，另TensorEnv自动复位终止势结算检查通过；观察器无故障。

## 2026-10-26 — close结束向下瞬时扣3分，并显示切换瞬间受力

- 用户确认叠加惩罚：有效close→lift切换物理子步，双手对物体的有符号竖直合力Fz<0时，`close_downward=-3`一次性计入；Fz≥0不扣，既不乘dt也不按力大小缩放。原close中向上奖励与close后持续下压罚分保留。`--close-downward-penalty`仅允许当前直接lift/连续有向力配置。
- 新合同仅允许精确单向增加事件奖励，网络151维、Adam及计数保留，不修改物理参数、阶段准入和成功标准。独立reward/close_downward用于查看事件扣分。
- 新增lift_entry_force遥测，只在有效close→lift同一物理子步、复位前采样。TensorBoard `lift_entry_force_N/竖直合力_向上为正`、`向上力_平均`、`向下力_平均`、方向峰值及左右手合力；`lift_entry_force/事件数`记录本rollout事件个数。对事件求均值，不混入其他时间的受力；无事件时force=null且不写力标量点。原force_N全程曲线保留。
- 144项相关CPU/GPU回归通过，含60/480Hz边界符号、重复/选择复位、非法/失败/inactive隔离、精确合同、环境中切换采样与无事件显示。事件惩罚独立复核16项CPU通过（4项CUDA未运行）；shell/diff检查通过。
- v131第369轮检查点（36274176样本、13项Adam状态）有限，旧Isaac中断不退出，验证检查点后终止；v132单实例从该检查点恢复4096环境。training_start确认第369轮/13项Adam状态保留；第373轮完成36667392样本，策略/Adam有限。该轮实际发生1次close→lift，切换合力−2.16610 N（左−0.75417、右−1.41193），close_downward批量均值×4096×24恰为−3，持续下压罚同时为负。TensorBoard瞬时力8项及事件数、两项罚分实际event读回通过；观察器无故障，成功0。新增遥测独立复核30项CPU通过（2项CUDA未运行）。旧训练记录保留。

## 2026-10-26 — 取消hold，close完成直接lift

- 用户明确要求删除额外hold等待。默认训练脚本启用 `--skip-hold`：双手各1–15.02 N、物体速度<0.02 m/s、滑移≤3 mm连续满足0.2秒后直接从close进入lift，不再等待0.3秒hold；原阶段编号/151维观察保留，未启用该选项的历史流程不变。
- close阶段持续向上奖励；首次close→lift后的下一子步开启向下罚分并锁存至reset。提升仍检查实际载荷、滑移、压力、跟踪/联动误差等，阶段进入lift不代表实际离桌。保留实际离桌30 mm并稳定0.5秒的成功标准。
- 新readiness合同明确hold时长0与close_to_lift，精确单向迁移保留策略/Adam/计数，拒绝反向或额外参数变化。116项相关CPU/GPU回归通过；最终补充普通丢失接触而非异常状态断言后，新5项CPU/GPU测试再次通过。shell与diff检查通过。
- v130第28轮检查点（2752512样本、13项Adam状态）有限；旧Isaac退出后启动v131，以4096环境、24步更新继续。实际已完成第31轮，首次新增close→lift直接转换1次，当前rollout的hold占用0；历史hold计数随检查点保留，不代表新流程仍经过hold。第29轮全在close，向上奖励均值2.00290e-5/控制步、向下罚分0，确认close已获得向上奖励。实际TensorBoard正负力和向下罚分启用比例可读、策略/Adam有限，独立复核4项CPU通过（1项CUDA未运行）。当前成功0，尚未达到离桌验收。只读观察器无报错。

## 2026-10-26 — close阶段向上奖励，完成close后向下惩罚

- 新模式 `--force-reward-start downward_after_close`：close开始即按双手对物体竖直合力 `+0.5*clamp(Fz,0,10)*dt` 持续给向上奖励；向下惩罚 `-0.5*max(-Fz,0)*dt` 从首次进入hold后的下一子步开启。掉回close不重新豁免，只有回合reset清除标记。没有1秒窗口或每回合5分上限。
- close仍要求双手各1–15.02 N、物体速度<0.02 m/s、滑移≤3 mm连续0.2秒；hold再稳定0.3秒进入lift。这是接触稳定条件，不是向上承重或离桌成功证明。物理保护及实际lift验收不变。
- 151维观测保持不变，计分位表示向下惩罚是否启用；新增TensorBoard `force_reward/downward_penalty_enabled_env_fraction`。精确单向v3→v4恢复保留策略/Adam，其他合同差异仍拒绝恢复。
- 验证：close_upward_reward、after_close_force_reward、continuous_vertical_reward、vertical_reward、tensor_training、tensor_task、tensor_env、vertical_force共100项CPU/GPU测试通过；独立复核48项CPU通过（5项GPU未运行），shell语法及diff空白检查通过。真实v129第24轮检查点策略/Adam有限，迁移仅含close_upward_reward_v4；旧Isaac已退出，v130单实例从该检查点继续。

## 2026-10-26 — after_close配置从零训练

- 按用户要求结束v128并保留第117轮检查点，另起v129全新4096环境；移除resume/initialize-from，不恢复策略、Adam或回合计数。
- 保持close完成后计分、连续向上奖励/向下惩罚、151维观测及中文受力曲线。training_start确认恢复来源为空、初始轮数/步数/回合/Adam状态均为0；实际首轮98304样本完成，策略与Adam有限、TensorBoard力标签存在。
- 当前成功0。只读监控继续运行，旧训练记录保留。


## 2026-10-26 — close完成后才计竖直力奖惩

- 按用户最新要求，以首次完成close进入hold为起点；close→hold边界子步不计力，下一个子步开始，之后本回合持续发放向上奖励/向下惩罚。掉回close不会重新豁免，只有本回合reset清除计分标记。没有采用此前讨论的0.5秒方案。
- `--force-reward-start after_close`仅与连续有向力奖励一起使用。力测量、中文TensorBoard曲线及物理保护全阶段保留；时间、失败、实际lift/保持奖励保持原逻辑。TensorBoard新增force_reward/enabled_env_fraction用于区分未开始计分与没有力。
- 追加计分状态观测（150→151），精确单向迁移保留旧网络列和Adam状态，新列及对应动量补零；策略能够观察计分是否已开始。新旧奖励合同被区分，非指定变更拒绝恢复。
- 118项相关CPU/GPU回归测试通过；独立复核54项CPU通过（4项GPU未运行）。真实v127检查点CPU迁移验证旧网络列未变、Adam状态保留。
- v127第110轮10813440样本检查点保留。v128单实例恢复当前策略/优化器/计数，启用after_close；实际已恢复至第113轮，151维观察、13项Adam状态均有限；第111–113轮全部环境仍在close，有实测受力但两项方向力奖惩均为0。TensorBoard计分启用比例和原中文力曲线均正常写入。hold后的正负计分由阶段边界CPU/GPU测试验证，当前恢复首批尚未进入hold，不能据此声称已经抓稳。


## 2026-10-26 — TensorBoard向上/向下力标注

- 新增每个有效物理采样的双手竖直合力正负分量归约，先分向上/向下再求均值，避免正负样本被平均抵消。记录向上/向下平均大小和瞬时峰值，均为N；原有带符号合力及分手数据保留。
- TensorBoard新增`force_N/向上力_平均`、`向下力_平均`、`向上力_瞬时峰值`、`向下力_瞬时峰值`、`竖直合力_向上为正`、`物体重量_参考`。同步英文vertical_force/upward_mean_n等标量和monitor字段。
- 79项相关测试通过，另实际TensorBoard临时event文件读回验证10个新标量（含中文标签）。仅新增遥测，不改变奖励、控制、观测或检查点合同。
- v126保留第25轮检查点（2457600样本，策略与Adam有限）；v127使用同一检查点恢复策略、Adam及计数，避免为记录指标重新学。旧Isaac在回调中吞掉中断，验证检查点后终止，未同时启动第二实例。已恢复至第27轮，training_start确认初始第25轮/2457600样本/4082回合/13项Adam状态；新中文曲线实际event读回及localhost:6006服务可见性通过。第27轮向上平均0.006236N、向下平均0.111738N，差值等于带符号合力−0.105502N。


## 2026-10-26 — 全回合持续向上奖励

- 用户确认取消向上力奖励的1秒窗口与每回合5分上限。新增`--vertical-force-reward-mode continuous`：每个有效非失败物理子步`contact=0.5*clamp(Fz,0,10)*dt`，整个回合持续发放；向下`-0.5*max(-Fz,0)*dt`保持不变。实际高度、保持、成功及有向夹稳一次性奖励不变。
- 保留历史window模式默认供旧配置复现，默认训练启动脚本显式启用continuous。150维观测保留旧时钟以维持形状，但它不再限制continuous奖励；新合同禁用contact_reward_window，监控不再把连续奖励标为已过期。新旧奖励合同不同，不静默加载旧检查点。
- 验证：96项相关CPU/GPU测试通过，包括两种采样频率、超过1秒/5分、晚接触、NaN/失败/inactive隔离、CUDA选择复位、新旧合同拒绝混用；shell语法与diff空白检查通过；独立复核95项CPU测试通过（4项GPU未运行）。
- v125中断并保留第299轮检查点（29392896样本，成功0）。按此前不续训要求，v126启动4096环境、24步/轮、每环境1024回合预算全新训练，不传resume/initialize-from；实际已完成第4轮393216样本，确认resume为空、初始计数0/Adam空，更新后策略与Adam有限。第4轮覆盖仿真1.2–1.6秒且没有回合复位，contact仍为正（均值2.96574e-6/控制步），downward_force为负（−0.0101464/控制步）；证明1秒后仍持续计分。该轮19.36秒，成功0。
- 持续力奖励仍可能鼓励低高度托举，不能代替真实离桌验收；保留原物理守卫，未宣称修复此前超速物理根因。


## 2026-10-26 — 向上奖励、向下惩罚，从零重训

- 使用物体过滤的法向力+摩擦力得到双手对物体的有符号竖直合力Fz。新开关`--vertical-force-reward`替换无方向contact：close开始前1秒内`0.5*clamp(Fz,0,10)*dt`，每回合最多5分；所有live子步（含终止）施加`-0.5*max(-Fz,0)*dt`，向下罚分不设1秒限制和饱和。
- 夹稳+3只在已准入lift、仍ready且实际Fz>0时一次发放；向下或零力不发，允许后续实际托起后领取。实际高度、保持和成功奖励保留。修复lift使用旧力目标、压力释放使用新目标的不一致，复用当前pressure_limited；未放宽硬保护。
- 108项相关CPU/GPU测试通过；独立复核首轮73项CPU、增量14项CPU通过。新奖励与时序合同不静默接受旧模型。默认启动脚本同步当前软指垫/联动400/实测夹持/新奖励配置。
- 用户要求旧训练先备份后删除：`/home/xk/coding/load/dh116_before_fresh_20261026_095257/training_history.tar`，3,419,822,080字节，107145个文件逐项SHA256与归档读取校验通过；旧runs原件已删除。SHA256：`88dbb517bb8bce4e286500bdb861fbe9e469bcc7e357a1a600fe3098d3d5ff5d`。上次10月25日备份保留。
- v125以4096环境、24步/轮、每环境1024回合预算全新启动；不传resume或initialize-from。已完成第1轮98304样本（23.13秒），初始resume为空、Adam状态0、计数0，更新后13项Adam状态及策略有限。真实首轮contact平均每步+8.8144e-6，downward_force平均每步−0.00172565；TensorBoard两项奖励均已写入。成功0，尚未离桌。
- 限制：v123第799轮后的被动关节超速物理根因尚未确定，新奖励和时序修复不等于解决该求解器冲击；原守卫继续生效。历史训练的成功数仍为0，阶段进入lift不等于物体离桌。


- **2026-10-26：双手向上作用力遥测与从零训练**：新增同一Wire过滤器的法向力＋摩擦力，取反后按世界Z方向有符号合计，排除桌面接触。记录双手分量、合力、实际质量×重力、超过重量的采样比例与瞬时峰值；统计为单次rollout的live物理子步，不作为成功判据，也不改奖励或提升门槛。89项相关测试通过；独立GPU加载校验向上摩擦0.4000002 N、动量反算0.3999784 N，最大误差6.23e-5 N。用户要求重新从零：旧v122第554轮保留，v123不加载模型启动4096环境、每环境1024回合；已实测完成前2轮更新（196608样本），training_start确认初始策略无加载、计数0、Adam空；第2轮策略/Adam有限。首轮18.39秒，第2轮双手竖直力平均−0.8631 N（向下）、瞬时最大+0.5130 N，重量1.1772 N，超过重量采样比例0；成功0。

- **2026-10-26：探索噪声失控修复候选**：v121未触发物理致命保护，但第400轮高斯标准差已达9.26–17.33，归一化动作范围[-1,1]下理论最低截断比例93.78%；未受限log_std在高斯熵奖励下持续增大，截断后难以细调。保留第400轮检查点，新增每次Adam小批次后及续训加载时将std投影至[0.05,1]，仅清理被投影噪声坐标的Adam矩，保留actor/critic与其他优化器状态。精确迁移记录明确策略分布发生变化；不改物理与任务奖励。新增std/动作截断比例遥测及observer校验，相关90项测试通过，实际第400轮CPU迁移确认网络权重逐项不变。独立复核无阻断；v122已从400轮续训至405轮，std实测0.9316–1.0、动作截断比例51.40%；第403轮策略/Adam有限且12个网络参数张量已更新。新增3次有界联动误差复位，无全局停机，实际抓取成功仍0。噪声失控已受约束，尚不证明抓取或长期学习效果。

- **v117停机与v121柔顺联动候选**：v117在第294轮后因右拇指越过关节限位保护停止。v118从同一模型诊断捕获另一起接触冲击超速（3.08249 rad/s）；法向接触力矩主导，不能归因于摩擦过大。TGS对照更早失败，未采用。PGS下仅将mimic自然频率从刚性0改为400 s⁻¹，v120完整通过272步故障动作重放：原环境641持续4.533秒且未提前复位，主动速度峰1.1523 rad/s、接触力16.673 N，另4环境发生有界联动误差复位。新增严格单向契约迁移及偏离配置拒绝测试；相关166项通过、1项跳过，复核收紧后训练模块15项通过。v121从真实v117第294轮策略启动4096环境正式PPO，排除诊断重放检查点；实测续训至第306轮（新增12次更新、每环境288控制步），无全局停机，新增1次有界联动误差复位；主动速度峰0.79269 rad/s、全关节2.970008 rad/s、接触力15.5686 N。第296轮读回策略与Adam有限、13个策略张量已更新。尚不能证明原拇指故障根治，实际抓取成功仍0。

- **v116诊断完成，v117启动续训**：从v115第196轮真实策略连续采样800控制步（13.333秒/环境），未触发新速度守卫；主动速度峰2.31621 rad/s、全关节2.970024 rad/s、接触力31.27765 N，新增3次联动误差和2次超力受控复位。保留第229轮完整PPO更新、丢弃最后8步不足rollout；诊断无动作重放或物理参数覆写，独立复核确认已完成检查点可续训，策略/优化器有限。v117以4096环境从229轮启动，已实测完成第230轮新更新，力/摩擦/奖励未改。106项相关测试通过。该有限时长诊断未复现v115的8.834 rad/s，物理尖峰原因仍未隔离；已有修复确保下次每子步拦截并报告，历史异常数据不追认为有效。
- **v115运行超速与守卫修复（诊断中）**：监控发现第149轮子步实测主动关节速度峰值8.834 rad/s；旧逻辑仅在终止/复位时检查速度，observer也未检查物理峰值，因此直到第196轮人工发现并暂停。新增每子步全部关节3+1e-5 rad/s检查，超限在奖励与复位前全局停止；精确单向检查点契约迁移。observer补充致命峰值和遥测缺失/格式校验。相关101项测试通过，补充schema后observer11项通过；独立复核无阻断。v116从第196轮策略做诊断采集，正式训练暂未恢复；守卫修复不等于消除物理超速，也不能追认此前超速后的训练数据有效。
- **v115首批完整回合实测**：第113轮核对，相对真实第75轮检查点，4096个环境各新增1个回合；4078次超时、16次未建立夹持、2次受控超力复位，无全局停机，实际抓取成功0。逐子步峰值30.622 N；策略/优化器全部有限，13个策略参数张量发生更新。继续原配置学习，未将进入lift阶段计为抬起成功；只读后台观察器仍运行，不会自行修代码或重启。
## 当前：完整 GPU PPO 与 close / hold / lift 分阶段奖励

- **实测夹持准入候选（v115已启用）**：v112诊断环境545曾双手各≥1 N且物体低速保持1.279秒，但按随机力目标跟踪判定的ready最长仅0.01667秒。新增可选 `--grasp-readiness measured_load`：每手≥1 N且≤1.25×配置最大目标+死区，沿用速度/滑移约束及0.2+0.3秒保持后允许尝试lift。力目标和过压暂停保持；奖励系数不变，但+3准入奖励的触发条件改变并写入检查点契约。仍须实际离桌30 mm稳定0.5秒才算成功。338项相关测试通过；独立复核无阻断，v115真实PPO已产生hold→lift阶段转换，尚未达到实际离桌成功标准。另新增逐子步物理峰值遥测，复位后仍保留峰值，相关51项测试通过。
- **退出状态监控修复**：将 Isaac 启动纳入异常处理，保留最后训练计数并原子发布退出状态；处理清理/写文件失败及非零 SystemExit，先保存结果再执行可能不返回的 Isaac 关闭。9项退出监控测试通过，独立复核无阻断。另修复异常路径丢失最后计数和未完成rollout物理峰值的问题；设备读回/写盘失败不会覆盖原始训练异常，非有限峰值显式标记。新增4项回归测试，训练与退出监控共22项通过；该遥测补丁仅下次启动生效，不为此重启v115。该修复不代表抓取物理故障已解决。
- **硅胶指垫柔顺候选（v115已启用）**：厂家CSV将 finger14/23/33/43/53_link 标注为硅胶A25；新增 `--hand-material grip_compliant` 仅将这5个末端碰撞体设为柔顺接触，K=10000 N/m、C=30 N·s/m为仿真估计，非实测标定。逐碰撞体USD绑定与逐link PhysX材料读回核验，检查点迁移仅允许已定义刚体物理配置。正式v114路径615步Isaac重放已通过，34个link×4096环境的运行时材料审计通过；K由PhysX负restitution编码验证，C验证USD配置，非独立运行时阻尼读回。不宣称稳定抓取。
- **v107历史故障与v115验证边界**：真实PPO续训到第75轮后，环境545出现50.872 N并保护停机。基座写入修复通过已知动作重放，但不是全局根治；第75轮策略已保留。v108已从同一第50轮检查点精确复现（615个控制步）；v109/v110仅放宽主动限速的诊断虽将峰值降到27.108 N，却出现4.415 rad/s主动关节速度，未采用。v112全手柔顺、v114正式仅软指垫均在原0.3 rad/s主动限速下通过615步重放；v114峰值11.311 N、主动速度峰值0.889 rad/s、最大联动误差0.01022 rad，新增物理复位0。v115从真实第75轮检查点续训；以上不代表长期稳定或抓取成功。未放宽过力保护。

- **固定基座重复写入修复候选**：v103精确复现v102的45.669 N冲击。v105保持原545步PPO动作、4096环境及物理/材料/奖励/保护参数，只跳过未变化的float32基座位姿写入，完整重放最高29.423 N，新增有界过力复位0（累计2来自源检查点）。正式实现按最后提交位姿比较，仅写变化环境；回合复位始终强制写选中环境。新增精确检查点迁移；304项相关测试通过（1个既有溢出测试警告），独立复核5项通过。正式生产路径v106完整545步重放通过且峰值与v105一致；原故障环境2645未提前复位（elapsed=3.0667秒），最后64子步峰值1.5695 N。补充乱序复位/实际reset接线后6项测试通过。v107从正式v102第50轮检查点恢复4096环境PPO，诊断重放模型未用于续训；长期稳定与抓取成功仍待验证。

- **后续监控发现仍有接触冲击**：v102在第50轮后环境2645出现45.669 N并保护停机。16次速度迭代只通过了原故障重放，不能视为普遍修复；已启动v103按同一第28轮检查点和随机状态精确复现，补充接触法向、接触点和驱动/几何记录。未放宽护栏。
- **接触求解修复（2026-10-25）**：v99精确复现v98的28轮指标和故障。原动作重放仅将创建场景时的速度求解迭代4→16（位置64、480 Hz不变）；v101通过675步，原故障环境未提前复位，末段峰值57.070→3.590 N。全4096环境峰值37.714 N，10次有界过力回合复位，仍无抓取成功；这验证了该故障轨迹，不代表所有探索轨迹永不失败。启动脚本固定16次，检查点记录参数并只允许精确迁移，原30/45 N保护、12 N即时目标、奖励不变。运行中改USD未生效的v100已排除。验证数据见 `runs/diagnostics_force_root/`、v99/v101目录。 新增配置传递/精确迁移测试先红后绿，299项相关测试通过（1个既有溢出保护测试预期警告）；`bash -n scripts/train_dh116_tensor_ppo.sh`通过。v102从本次全新v98第28轮模型恢复，实测到第44轮/4325376样本，全部4096环境均新增至少1回合；新增2次有界过力复位，其余未建立抓取超时，未全局停机。策略/Adam有限且策略已更新，成功仍为0；运行状态以活动记录为准。

- **按用户要求从零训练**：历史 `runs/` 与 `archive/` 已备份到 `/home/xk/coding/load/dh116_before_fresh_20261025_174131/training_history.tar`，tar逐内容比较通过并保存SHA-256后，已删除原目录（归档32398407680字节）。新训练不传resume/initialize-from，使用全新策略、Adam及回合计数。场景输入移至 `configs/dh116_training/` 并保持字节哈希一致；默认启动入口不再依赖旧runs。保留4096环境、每环境1024回合、12 N即时目标及close前1秒奖励。旧v97第243轮发生45.807 N冲击并停止，故障记录保存在备份；重新初始化不视为该物理问题已解决。
- **v98后续发生故障**：全新训练在第28轮后、环境297的close阶段出现57.070 N单链接触冲击并保护停机。保留本次全新训练的检查点；下述求解修复通过故障动作重放后再续用该模型，不恢复已备份的历史训练。

- **当前采用PPO力目标立即更新**：用户确认取消力目标爬升，PPO仍在每手0.1–12 N内自主选择，不固定12 N。新独立配置 `v2_pressure_12n_instant_candidate.json` 使用 `force_rate_n_s: null` 明确关闭目标限速；标量/NumPy批量/GPU控制一致。旧有限速配置与检查点保留。目标相关卸力门槛随之立即改变，实际力仍由接触、驱动及原关节/偏移速率决定；不宣称实测力能瞬间到12 N。
- v97从第225轮恢复至第229轮/22511616样本；运行合同确认目标速率为null，全部4096环境驱动读回通过，策略/优化器有限。已出现1个超力失败回合并按原规则复位，训练继续；尚无hold/lift或成功。297项回归通过（1个既有溢出保护测试预期警告），独立复核4项通过。TensorBoard选择 `v97_instant_force4096`。
- **历史v96：单手力目标上限12 N及2 N/s斜坡候选**：新增独立 `v2_pressure_12n_candidate.json`，单手合计物体接触力目标0.1–12 N；为在6秒抓取超时内让斜坡目标可达，加力速度由0.5改为2 N/s（0.1→12 N需5.95秒）。关节力矩仍0.2 Nm，保护与奖励不变；这不是指尖12 N实测或官方15 N指标的标定复现。旧4 N配置保留。策略/Adam/计数续用，但力动作映射和相关观测归一化发生变化，不宣称行为等价。78项相关测试通过，实际受力及长期稳定性待验证。
- v96实际从第209轮继续至第212轮/20840448样本，运行合同确认为单手12 N、2 N/s；全部4096环境双手驱动限值读回通过，策略/优化器有限。当前无失败记录、仍close且成功0；这是短期更新验证，不是12 N实测承载或长期稳定验收。TensorBoard选择 `v96_force12_4096`。
- v95从第170轮策略/优化器恢复，实际核验第174轮：全部4096环境的一秒窗口已过期，该轮contact奖励为0；无失败，策略/优化器有限。61项相关测试通过。TensorBoard选择 `v95_close_window_4096`；仍未证明抓取/lift成功。
- **contact 改为每回合仅一个1秒窗口**：默认每回合close开始即启动，即使没有接触也计时，之后连续计时，不因接触丢失暂停，也不因再次接触或阶段切换重开。窗口内每物理步奖励 `5 × clip(总接触力/10N, 0, 1) × 有效秒数`，每回合合计最多5分，过期后为0。保留显式首次接触计时模式供历史复现；当前训练采用close_start。原close/hold进度奖励关闭，夹稳首次进入lift一次性+3；真实提升/保持及成功+25、失败−25保留。
- **状态与续训**：新模式观测由148增至150维，新增窗口时钟与是否发放夹稳奖励；20维动作不变。旧原生检查点只允许精确合同迁移，保留原策略、Adam状态和计数，新增输入列及对应动量补零。旧奖励模式仍可显式使用；默认启动脚本启用新窗口。262项相关测试通过，独立复核28项通过；实际运行见 `runs/active_tabletop_run.json`。
- **新奖励实际续训（v94）**：从第151轮原生检查点迁移，核验到第155轮/15237120样本，策略及Adam状态有限、权重已更新；150维观测生效，部分环境窗口已到期，接触累计分最高1.4179（上限5）。当前无整批停机、成功0，尚未进入hold/lift。TensorBoard选择 `v94_contact_window_4096`；每30秒只读观察器继续记录，不自动修复代码。
- **物理问题分开验证**：v92的立即终止复位修复通过前三批回合，但第134轮仍在新发生的超时终止步测得3.00001788rad/s并停机。v93从第134轮恢复，仅将被动关节驱动限速从3改为候选2.97rad/s，保留终止保护3rad/s及固定1e-5容差；全部4096环境实际限速已读回核对。v93至第150轮已新增一批4096回合（4095个未建立抓取超时、1个有界耦合失败复位），无整批停机；第151轮完整检查点保留后切换奖励。该候选给求解器留余量，不保证所有接触条件下实测速度都被硬钳制。尚未验收稳定抓取/lift。

- `--tensor-pipeline` 将碰撞凸包净空、状态读取、控制、观测、奖励、rollout 和 GAE 放在 GPU；使用已安装 RSL-RL PPO。4096 环境，每环境 24 控制步更新一次，即每轮 98,304 样本。网络为 128×128、20 维动作；旧模式148维、新窗口模式150维观测；每轮 5 个优化 epoch、8 个 minibatch，KL 自适应学习率。旧 SB3 检查点保留；新格式为 `tensor_ppo.pt`，不能冒充同优化器续训。
- 保留刚体位置、强摩擦5/5、物理480Hz、每控制步8子步、64次位置求解、关节/手掌/力命令限制和实际离桌成功标准。无 MPC、无 approach。CPU 仅参与场景初始化、少量保护标志判断和每轮汇总/落盘。
- 当前阶段由实测力/速度/滑移门控，不要求四点接触；contact窗口及首次夹稳奖励见上文。lift按真实物体底面升高至离桌30mm（最多5分）及稳定0.5秒（最多2分）奖励。阶段退回扣回提升进展；contact额度只在该环境新回合复位。
- 实测：16环境完成两次GPU PPO更新，每轮约5.6秒。4096环境v84第一轮采样16.489秒、更新0.182秒，合计16.671秒，5896.8样本/秒，约旧流程8.8倍；显存约5.53GiB。每轮迭代不是完整episode，不能把此速度称为每回合16秒。
- v84随后误停；v85重复复现：正常运行被误套用复位专用速度判据，读数3.000000238rad/s、力5.6876N、耦合0.01410rad、净空2.29mm。已恢复旧判据适用范围，复位/终止行仍受3rad/s检查。v86续训完成第9轮/884736样本后，在env3981出现32.558N接触冲击、3.171rad/s终止速度而正确停机。v87同动作逐物理步复现，确认反馈偏置已饱和时，独立PPO目标仍可继续闭合；瞬时冲击的全部接触成因尚未隔离。
- 新增显式 `--pressure-relief`：实测单手载荷超过斜坡力目标的1.25倍加死区时，已标定方向缓慢卸力、其他屈曲关节禁止继续闭合，并暂停手掌微调与提升；原关节范围、速度、力矩、耦合优先恢复及硬保护保留。v88从第9轮迁移，保留策略/优化器，完成至第14轮；v89从其检查点加载补充的提升暂停修复。仍有过载复位，不能称为物理问题全部解决或已学会抓取。当前状态以 `runs/active_tabletop_run.json` 指向的monitor为准。
- 验证：GPU/CPU控制与观测对照、阶段奖励、防刷分、完整RSL更新、失败复位/非有限保护等21项通过；与原复位、训练和TensorBoard回归合并185项通过。4096实际GPU与旧CPU初始状态最大差4.77e-7，凸包净空差0；环境映射与单环境物理复位隔离通过。尚未验收稳定抓取/lift；历史全量测试问题不在本轮宣称已解决。
- 启动：`bash scripts/train_dh116_tensor_ppo.sh`（已有Isaac训练/回放时拒绝重复启动）。续训：`DH116_RESUME=runs/<run>/tensor_ppo.pt bash scripts/train_dh116_tensor_ppo.sh`，保持相同24步和物理/奖励合同，输出使用新目录。原本地 `runs/tabletop_launch_20261025/train_ppo.sh` 已转到此入口。默认每环境1024回合，多核多线程。
- v89完成第29轮/2850816样本，最近9轮未新增物理复位，随后首批回合终止时在3.0000066757rad/s处停机，其他物理保护均通过。终止/复位速度检查新增固定1e-5rad/s求解器数值容差（名义3rad/s不变，接受范围略增）；不裁剪测量值、不自动扩大容差，3.00002和3.171rad/s仍拒绝。新字段及旧检查点迁移均记录，v90从第29轮策略/优化器继续。
- v90实际完成至第45轮/4423680样本，所有4096环境至少完成1回合，累计4203回合；本次4094次未建立抓取超时及3次有界物理失败后继续训练，无整批停机。仍未进入hold/lift、成功0；这是完整失败回合复位和优化持续性证据，不是学习收敛或抓取验收。后台观察器每30秒只读记录，不能自动修改代码。
- 本次卸力及速度判据修复验证：压力优先级、饱和偏置、关节范围、选择性复位、过载暂停提升、速度容差边界与检查点迁移等，合并原GPU/训练/复位/TensorBoard回归199项通过；实际长期训练和lift仍待验证。
- TensorBoard：<http://localhost:6006>，选择 `v90_pressure_relief_4096`；monitor记录采样/优化耗时、reward各项、close/hold/lift占比、阶段转移与终止计数，以及卸力介入比例；主要耗时、reward和阶段占比同步TensorBoard。训练成功率仍与资格验收分开。

## 历史 v81：4096 环境、32 步 rollout（已停止）

- v80 在 225280 样本、0 次更新时因 excess_force 停止；v81 完成一次优化 epoch 后，在 241664 样本因 env1056 单碰撞体载荷 30.7225 N 停止。旧吞吐约 668.5 条样本/秒；不能称为抓取成功。
- v82 用相同 55 步动作前缀加 8 步后续检查，验证一次有界失败回合复位，回放完成且没有后续安全失败。这是诊断证据，不是抓取验收。

## 历史 v78–v80：4096环境、多核、64次物理求解迭代

- 用户明确允许多核/多线程；`scripts/python.sh` 默认覆盖可用CPU，数值线程按可用物理核设置，仍支持显式环境变量覆盖。GPU训练入口不再硬编码PyTorch单线程，新增 `--cpu-threads`，启动记录实际亲和性和线程数。
- v78从v77的77824样本检查点恢复（当时0次优化更新），4096环境、每环境1024回合、强摩擦5.0/5.0、close→hold→lift；保留原模型/优化器/回合计数，未完成rollout重新采集。运行记录确认PyTorch6线程、CPU0–11可用，Isaac启动后OpenBLAS环境为12线程。
- v78随后在303104累计样本（本次55个控制步）因env3039的耦合误差0.0218446 rad停止，尚未优化。v79以相同4096环境和完全相同55步动作，仅将位置求解迭代16→64进行对照：全段通过、无安全故障、峰值0.0177825 rad；这只验证该故障序列，不是全训练稳定证明。v80采用64次迭代重新训练，保留全部旧证据和0.02 rad阈值。
- GUI演示已关闭，只运行一个Isaac训练实例。PPO可在净空约束内调整手掌，不修改初始抓取点。物理/动作/奖励及硬保护保持；训练是否存活、步数和安全停机以 `runs/active_tabletop_run.json` 指向的monitor为准。
- TensorBoard：`http://localhost:6006`，选择 `v80_extreme_4096_pos64` 及其live曲线。新线程配置的稳定吞吐和完整抓取成功仍需训练验证，不能把启动或解除亲和性限制称为成功。


## 2026-10-25 当前演示：强摩擦物体

- 用户要求增大物体摩擦；新增 `--object-material extreme_grip`，静/动摩擦5.0/5.0、max组合、恢复系数0，刚体显示粗糙度1.0。这是人为强摩擦实验配置，不是材料实测或PhysX数值最大值。
- 实际单环境Isaac读回物体5.0/5.0/0.0，双手仍为1.0/0.8。v75窗口执行v72检查点，显式 `--allow-playback-material-change` 仅允许推理时比较物体材质，保留原检查点及训练契约；其他物理/奖励差异仍拒绝。
- v72训练已因mimic_error停止；v73复现被用户演示请求中断。目前是GUI推理，尚未恢复PPO优化，也没有验收抓取或lift成功。下一次强摩擦训练须显式选择extreme_grip，不得把旧材质检查点伪装成同契约续训。
- 本次材质/回放测试15通过；PhysX读回和GUI策略执行已核验。TensorBoard接线已实现，地址localhost:6006，演示期间没有新的PPO优化曲线。


## 2026-10-25 当前：全手接触与新关节范围，重新训练 PPO

- 已实现：显式 `--full-hand-contact`，双手各17个碰撞刚体的 Wire 过滤法向合力幅值；保留四指垫诊断，新增五指/手掌分组观测。观测148维，动作20维；力反馈、夹稳门控、接触进展奖励和局部过力保护同步覆盖其他指节。
- `--joint-envelope configs/dh116_joint_envelopes/free_grasp_geometry_v1.json` 使用逐关节 lower/center/upper 映射，压力补偿和耦合恢复均限制在同一范围，再按原速度限制执行。范围经过原USD全部碰撞几何的单关节采样/角点检查，是候选仿真范围，不是实机标定或连续动作箱全域安全证明。
- 已验证：152项相关回归通过（1项既有溢出保护测试的预期警告）；16环境实际Isaac短训8192样本、37优化epoch、16训练回合终止/复位，无安全停机；148维策略权重发生更新，策略/优化器有限且优化器状态非空。
- 尚未成功：短训和评估未建立稳定夹持，不能称为抓取成功。v71原2048环境在633856样本因env61耦合误差停止；旧记录缺少逐关节带符号轨迹，根因尚未证明。本轮保留0.02 rad硬阈值并补充所有失败环境的诊断，没有靠放宽保护绕过停机。
- 当前已启动新合同的v72：2048环境、每环境1024回合，使用全新策略/优化器/计数，保留旧模型。启动不代表训练完成；实际PID/阶段/步数以本地 `runs/active_tabletop_run.json` 指向的monitor为准。观察器只读记录问题，不会自动修改训练。
- 本轮保持刚体、原物体位置、摩擦、力矩上限及30mm离桌/0.5s保持验收。完整示教预训练留到后续。历史全量测试失败项未在本轮全部处理，不声明全量测试通过。

设计与实施记录见 `docs/superpowers/specs/2026-10-25-dh116-free-grasp-design.md` 和 `docs/superpowers/plans/2026-10-25-dh116-free-grasp.md`。

本次以 `10-25` 分支发布当前项目快照，保留 `10-23` 历史版本。下方 10-24 记录中的“本地未上传 / 无 commit/push”描述当时状态；相应源代码、测试、资产和技术文档已纳入本次版本。

- **已实现**：刚性圆杆 / 中空吸管 / DeformX 柔性线入口、20维手指/手掌/夹持力策略、每环境回合预算、检查点续训与显式扩容、单环境 Isaac 策略演示。当前运行采用高摩擦实心圆杆（半径8 mm、长0.5 m、质量0.12 kg），保持原位置；从 close 开始学习 hold/lift。
- **摩擦配置**：手部全部碰撞体静/动摩擦1.0/0.8，粗糙圆杆1.5/1.2，摩擦组合max、恢复系数组合min、恢复系数0。参数是训练假设，不是厂家标定或真实树枝测量。材质绑定覆盖实例碰撞网格，PhysX读回检查所有环境；外部柔性接触有独立材质，拒绝混用刚体材质配置。
- **训练准入**：直接训练只要求复位隔离、两步有限安全状态及运行时保护，允许无接触的失败经历参与学习；显式接触诊断保留120步检查。没有放宽真实离桌/稳定保持标准。
- **吞吐优化**：净空计算改为全部凸包顶点的等价Z坐标计算，无顶点删减，无物理/动作/奖励变化。实际Isaac短测1024/2048/4096环境分别773.92/804.23/742.14条样本每秒，选择2048；这是三步推理/物理基准，不是长期吞吐或显存上限证明。
- **已实际验证**：1024环境手/物体摩擦读回一致；刚体GUI场景实际打开；2048环境恢复并完成首次PPO更新，154624样本检查点策略权重已改变、策略/优化器有限且优化器状态非空。该次更新触发KL早停（0.04），未更改保护参数。首次更新后的实际训练约706条样本/秒。
- **2026-10-25 上传前采样快照**：v71处于training，2048环境、527360累计样本、11个优化epoch，未触发安全停机；完成回合数和成功数均为0。该数字只代表核验时刻，实时状态请读取本地 `runs/active_tabletop_run.json` 指向的monitor。
- **尚未验证**：当前策略的稳定夹持/提升成功率、1024回合/环境的最终验收和实机迁移；不能将代码测试、PPO更新或GUI显示称为抓取成功。
- **运行约束**：单Isaac、CPU0/数值单线程；显存有余量但主要受CPU计算限制。训练日志、检查点、监控与缓存不上传。`--pose-summary`、`--grasp-initialization` 仍依赖本地运行配置，部分资产路径绑定本机；本次为开发快照，不保证全新机器直接复现实验。

**本次发布验证**：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q tests` 得到1632通过、73失败、1跳过（1条预期溢出警告）。已修正其中一个旧approach时序测试；随后53项approach/净空/摩擦/准入/播放/扩容专项全部通过。仍有72项未解决失败，不能声称全量通过；命令、范围和失败节点见 [验证记录](docs/validation_20261025.txt)。`git diff --cached --check` 通过。

## 2026-10-24 历史：柔性 PPO 扩为1024环境续训

- 实测1024/2048环境短基准分别53.70/53.65 transition/s，显存采样峰值3.61/4.08 GiB；扩大到2048没有采样收益而增加更新等待，选择1024，未继续测试4096，不称为显存容量上限。基准含原策略推理及物理推进，不含优化器更新和训练回调；结果在 `runs/flexible_env_capacity_20261024/selection.json`。
- v66安全保存于67072累计样本、10个优化epoch。v67 `runs/tabletop_flexible_20261024_v67_env1024_eps1024` 已实际恢复并推进到69120样本（新增2048），保护未触发；此时扩容后首轮优化尚未发生，不宣称抓取成功。最新状态见 [v67 monitor](runs/tabletop_flexible_20261024_v67_env1024_eps1024/monitor.json)。
- 显式 `--allow-env-expansion` 仅允许环境数增长，保留策略、优化器及原512环境历史，新512环境计数从零开始；每环境仍1024完成回合，未完成回合重新复位。原位置、136维观测、20维动作、480Hz物理、64步rollout及奖励保持。
- 将选定环境的复位合并，避免每个环境分别触发整批驱动写入/读回；零抖动全复位与自动复位采用批量路径，有抖动保持原路径。原环境隔离/终止观测/RNG语义保留，复位不推进物理时钟。实际1024/2048全复位分别3.51/7.25秒。
- 本次验证：192项迁移/生命周期/奖励测试及33项复位/向量/任务测试通过（故意溢出保护用例的一条预期警告）；两个规模实际Isaac启动/隔离/原位置检查通过、退出0，无运行时显存或物理缓冲区错误。实际旧检查点在CPU形状环境扩容加载后策略和完整优化器逐项相等，rollout为 `[64,1024,136]`；该离线检查不代表物理抓取结果。
- 单Isaac、CPU0/数值单线程，无commit/push。保留旧运行；新增进度证明 `expansion_start_verification.json`，长期收敛和扩容后完整优化仍待运行验证。

## 2026-10-24 历史：原位置柔性物体接入原 PPO（v66，已保存停止）

- 仅替换物体：保留原 wire 中心 `(0.43, 0, 0.308)`、Y 轴方向、双手初始根姿态/关节姿态；不使用独立探测的 X 偏移或自动下降到 z=0.305。手指/手掌/反作用力控制、close→hold→lift、奖励权重保持。
- 20 维动作＝双手12个主动关节目标＋双手6个XYZ微调＋双手2个夹持力目标。每手11个关节中6主动、5耦合从动；20不是主动关节数。
- `--object-kind deformx_wire` 使用批量CUDA Cosserat、全部34个手部凸碰撞网格和COM反作用力；观测使用法向接触力，物理反力包含摩擦。碰撞计算按环境独立，子集复位不推进全局时钟。
- 抓取区间按初始指垫位置选定后固定到材料坐标；局部表面最低点、双侧各自滑移与最大局部速度参与验收，避免远端下垂或相反方向滑移相互抵消。保持136维观测形状；物理/观测语义在新检查点契约中明确区分旧刚体。
- 柔性入口取消“先接触成功才准训练”，改为有限状态、复位隔离和有界启动动作检查；没有接触的普通失败回合允许学习。所有关节/数值硬保护仍保留。
- 83项资产、区域测量和原单/批量任务回归通过；另3项适配器测试通过，包括COM力矩平移不变性、局部下垂/相反滑移和非均匀端点的材料长度加权均值。
- 实际双环境Isaac启动通过，`runs/tabletop_flexible_20261024_profile/summary.json` 为 `startup_validated`、退出0；原位置未变。初步PPO已采样12条后因性能定位主动停止，尚无优化更新。已优化近手凸网格距离查询，11项CUDA测试通过；实际初始手姿态512杆每1/480秒推进约0.256秒（独立求解器，不含Isaac/PPO）。512正式运行 `runs/tabletop_flexible_20261024_v66_env512_eps1024` 已通过原位置/复位隔离检查并开始采样，已核验首批32,768条采样后的5个优化epoch，并继续采样；未触发保护，未调整训练参数。尚无完成回合，不能称为学会抓取。
- GUI兼容路径 `SkeletonWire(..., render_mode='mesh_points')` 直接发布原网格蒙皮结果、法线/包围盒，禁用该实例二次蒙皮；15项资产测试通过，实际GUI新路径仍待验证。旧独立demo只保持手指初始目标，不代表PPO动作。

## 2026-10-24 历史：DeformX 独立导入验证（本地未上传）

- 用户选择论文项目自带物体：原始橙色 USDC，长1 m、半径5 mm、20骨骼。Isaac 可读取的入口为 `assets/deformx_wire_v1/wire.usda`，与原USDC、纹理和许可证一同存放；原资产SHA记录在manifest。
- 保留原始2652顶点网格和蒙皮；每实例独立骨骼动画。原文件不带柔性物理，必须由外部Cosserat运行时驱动；单独打开USD只显示模型。
- 10项离线测试通过：目录搬移后的引用、真实UsdSkel网格弯曲、世界/祖先变换、实例独立性、无效状态原子拒绝。命令：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_deformx_wire_asset.py -q`。
- 物理半径对齐网格5 mm（上游示例物理半径6.35 mm存在不一致）；材料采用项目示例密度700 kg/m³、E=200 kPa、G=E/3、damping=0.1、杆步长10 µs，不称为实机材料标定。抓取附近固定材料区域用于后续局部抬升测量。
- 已实际完成单环境Isaac导入：默认初态2步有限/稳定，因新物体更细，旧指垫距表面约3.9–4.4 mm，未接触。独立接触探测仅把线横向偏移4.5 mm，12步（0.025 s）成功测到手—杆接触；单body步均反力峰值0.7626 N、接触冲量平衡残差最大5.42e-20、最大节点位移2.484 mm。实际耗时51.31 s，退出码0，结果在 `runs/deformx_import_20261024_contact/summary.json`。这不是夹持/提升成功或长时间稳定性验证。
- 运行入口：`DH116_NUM_THREADS=1 bash scripts/python.sh scripts/demo_dh116_deformx_wire.py --headless --device cuda:0 --steps 12 --output runs/my_deformx_import`；去掉`--headless`可打开Isaac窗口。独立接触验证另加`--contact-probe-offset-x-m 0.0045`，默认偏移为0。运行前确保没有其他Isaac实例。
- 共34项资产/运行时/原材质/初始化回归测试通过，另29项运行时和原Cosserat物理/后端测试通过。新增入口使用全部34手部碰撞网格与真实COM反力；保留原刚体入口默认行为。修正默认app.close退出等待，沿用已验证的post_quit/skip_cleanup关闭流程。
- v65刚性PPO已保存并停止：2037760累计样本、301优化epoch、每环境10个已完成回合；完整checkpoint ZIP校验通过。按本次更换物体的目标保留旧检查点供回退，未自动继续训练旧物体。512环境柔性PPO尚未接通，单杆求解尚未向量化，不能把这次单环境导入称为批量训练完成。
- 已知限制：外部CUDA显式罚函数接触、Coulomb系数1.0近似高摩擦，无绳自碰撞；材料为项目示例参数。原20骨骼蒙皮对拉伸/剪切及末端位置仅近似（原驱动也如此），实际碰撞采用求解器中心线/动态半径；本次最大单元长度偏差0.055 mm、director/中心线剪切差0.00719。USD祖先1.4倍均匀缩放有小于0.1 mm半精度蒙皮误差，运行时使用未缩放父节点。

## 2026-10-24 历史刚性训练：接触奖励10 N / 5分续训（v65，已保存停止）

- 用户指定所有指垫总接触力0..10 N线性映射0..5分，沿用势差奖励：每步支付当前势值减前一步势值。10 N以上饱和，接触力下降扣回，不按指垫数量判定，不修改力上限或提升条件。
- 单环境与GPU批量任务共用奖励函数；奖励版本为`object_support_progress_v4_10n_5`。新增显式`--allow-contact-reward-change`，只允许原v3到该版本且其他契约完全一致的续训，保留模型、优化器及每环境回合计数。
- v64已保存退出：1710592条样本、256个优化epoch、每环境9个已完成回合。v65从此继续，512环境、每环境1024回合目标；未完成回合重新复位，未优化的部分rollout不重放。
- 237项奖励、批量/单环境一致性、训练与检查点测试通过；包含故意溢出状态保护用例的一条预期RuntimeWarning。编译及diff检查通过。这些不是实际10 N夹持能力或抓取成功的证明。
- v65已核验恢复并开始新采样（后续核验1744384累计样本、261优化epoch，已完成首轮新奖励优化；每环境9个已完成回合），报告记录v3→v4奖励迁移，未触发保护。实际采样/优化状态见[runs实时监控](runs/tabletop_gpu_20261024_v65_contact10n_env512_eps1024/monitor.json)。新旧奖励下原始回报不直接比较；物理成功率仍按真实离桌和保持验收。单Isaac、CPU0/单线程，无commit/push。

## 2026-10-24 历史：512环境、每环境1024回合（v64，已停止）

按用户确认，训练预算改为每个环境至少完成1024个训练回合（总计至少524288回合），
按所有环境的最小完成数验收预算。普通失败回合计入，预检/评估不计入。
每轮仍采样64步并优化；达到目标后完成当前rollout的更新，快环境可能多跑少量回合。

- v63在229888条累计样本、31个优化epoch处保存并退出，各环境已完成1个失败回合。v64恢复该模型、优化器及完成计数，未完成回合重新复位。
- v64已实际恢复CUDA训练，首查累计230912条样本（本轮新增1024）、31个优化epoch，512个环境的完成计数均恢复为1，目标1024，未触发硬故障。后续以 [v64实时监控](runs/tabletop_gpu_20261024_v64_env512_eps1024/monitor.json) 为准；`episodes_per_env_target`与每环境`episodes`区分目标与已完成量。约110 transition/s且每回合360步时约需20天，回合长度和吞吐会影响实际耗时。
- 新增每环境回合预算、检查点回合统计、旧检查点进度迁移校验；修正续训加载时固定CPU的问题，GPU入口按CUDA恢复。161项训练/检查点测试、编译/diff检查通过。
- 保留直接训练、硬保护和原物理/控制/奖励；单Isaac、CPU0/数值单线程。预算不是抓取成功标准，不宣称已经稳定抓取。旧运行保留，无commit/push。

## 2026-10-24 历史：512环境直接PPO（v63，已停止）

按用户要求启用 `--direct-training`：取消训练前的三批完整诊断回合，
启动时仍核对复位、场景隔离、驱动参数和所有手的接触；接触检测满足后立即结束预检。
抓取失败进入PPO学习，逐步物理保护和最终真实离桌验收保留，控制/奖励/物理参数不变。

- 128项向量训练测试通过，包含失败抓取仍更新、初始复位异常拦截、训练物理故障停止及旧模式回归；编译和diff检查通过。
- v62在0更新时停止，历史记录保留。v63已在8个控制步内检测到全部1024只手的接触，并跳过完整准入进入实际PPO采样；首次核验3072条训练样本、0优化epoch，首轮rollout需32768条。预算1048576 transition，实际优化进度见 [v63监控](runs/tabletop_gpu_20261024_v63_direct_env512/monitor.json)。采样不代表已经学会抓取。
- 不再将跳过的完整准入标为通过；报告明确 `mode=direct_training, probes_skipped=true`。单Isaac、CPU0/数值单线程，无commit/push。

## 2026-10-24 历史：按用户要求重新启动512环境PPO（v62，已停止）

`v62_env512`已启动，沿用工装试验前v54的抓取配置：直接close、20D动作/136D观测、高摩擦中空吸管、直径闭合补偿、0.1–4 N每手目标、.2 Nm驱动上限、480 Hz物理/60 Hz策略、16次位置迭代和刚性线性联动。没有采用工装试验的2 Nm或64次迭代候选。

- 训练预算1048576 transition，每8192步保存；单Isaac、CPU0/数值单线程。自身接触预检与物理准入通过后自动进入PPO，硬故障自动停止。
- v54停在预检、0更新且没有检查点，因此本轮使用全新策略，不能称为恢复已学习权重。
- 目前进程已启动，尚待预检/准入；实际采样与优化数以 [v62实时监控](runs/tabletop_gpu_20261024_v62_env512/monitor.json) 为准。[活动运行指针](runs/active_tabletop_run.json)已更新，IDE中的v18是历史运行。
- 本次只重新启动任务，没有修改控制、奖励或硬保护；尚未解决的工装持续载荷限制见下节及试验报告，不宣称稳定夹持成功。

## 2026-10-24 历史：单指工装加载完成（本地未上传）

实际Isaac完成左右拇指/食指固定平板加载，共56次（含重复及休眠/求解器对照），**0次达到14 N持续1秒**。当前姿态的仿真力能力尚未按官方指标验证，不能将试验峰值称为真实DH116最大力。

- 16次求解迭代下，.2 Nm上限出现约1.31–2.45 N峰值；食指触发.02 rad联动保护，拇指稳态只有约.064/.074 N。增大上限到.5/1/2 Nm时先触发联动保护。
- 改为64次迭代后，.2 Nm四指可安全完成；2 Nm短时最高10.704 N，仍有联动保护。未触发保护的左拇指稳态约.895 N，未达到14 N。关闭休眠无改善。持续载荷传递原因尚待进一步诊断。
- 20项相关测试通过，独立审查问题已修正；保持窗口按480 Hz全采样验收。源资产未改，无实机操作、无commit/push，大力矩试验参数未写回PPO。
- v54已在预检阶段停止，PPO新增采样/更新均0，IDE中的v18仍是历史记录。详见 [试验报告](docs/dh116_fingertip_fixture_20261024.txt)、[测力曲线](runs/fingertip_fixture_20261024_report/force_traces.png)、[原始汇总CSV](runs/fingertip_fixture_20261024_report/measurements.csv)。

## 2026-10-24 历史：V2 控制矫正，v54 已暂停（本地未上传）

已修正加压方向引起的指垫上移：使用源USD关节几何计算沿直径闭合的方向，加入受原净空/速度/位移约束的手掌补偿。显式新配置将每手目标法向载荷范围扩大至0.1–4 N，启用480 Hz刚性线性联动；力矩暂保留.2 Nm、刚度10–20 Nm/rad。4 N是请求上限，不能称为已测夹持能力或厂家最大力。

- 本轮相关离线测试253 passed，联动/CLI专项56 passed、1 skipped；编译、diff检查通过。实际GPU8参数读回、复位隔离和24个零/随机动作准入回合全部通过，无硬故障；抓取成功0/24。
- v48–v52物理对照显示：增大刚度、力矩或迭代次数没有单独解决提升。新几何离线最大局部高度误差.342 mm，但提升时仍可能滑至上方；v52最大物体升高仅.143 mm，未达到30 mm稳定保持标准。不能称为抓取已解决。
- `v54_corrected_env512`已启动全新策略训练流程：512环境，1048576 transition预算，每8192步存档。512环境自身准入尚在执行，须通过后才开始PPO优化；不得借用GPU8结果声称512已通过。单Isaac实例、CPU0/数值单线程，硬故障仍自动停止。
- 查看 [v54实时监控](runs/tabletop_gpu_20261024_v54_corrected_env512/monitor.json) 和 [活动运行指针](runs/active_tabletop_run.json)，IDE中的v18是历史记录。v47已停止且新增PPO更新为0，旧检查点保留。
- 官方CAD质量/惯量异常、线性与非线性联动差异及缺失的电流—力矩曲线仍明确标为未标定，未虚构实机参数。详见 [本轮修正及验证](docs/dh116_v2_correction_20261024.txt) 和 [官网证据](docs/dh116_v2_web_search_20261024.txt)。

## 2026-10-24 历史：v47 GPU 启动验证（已停止，本地未上传）

已启动 `v47_env512`：单 Isaac 实例、CPU0/数值单线程、512 套双 DH116 与高摩擦中空吸管。物理与策略使用 CUDA；保留20D动作/136D观测、原反作用力控制、奖励、阶段判定及硬保护。

- 实测显存约3.7GB；512环境的1024只手均检测到物体接触。当前仍为接触预检，PPO新增采样和优化次数为0；完整准入通过后，该进程自动进入PPO。预检约107 transition/s，仍有单核主机端/仿真吞吐瓶颈，环境数不是线性加速倍数。
- v46的32环境已完成96个完整准入回合，全部无硬故障；抓取成功0/96。512环境的完整准入及新的策略更新尚待运行，不能借用32环境结果声称已通过。
- v41已主动停止并保存17764步/118优化epoch。v47配置从其同维度检查点导入权重，重新建立优化器和本轮计数；训练预算1048576 transition，每8192步存档。
- 修复GPU薄壁凸包烘焙后接触缺失问题，世界几何保持一致；新增批量控制计算与物理I/O。相关回归208项通过，代码审查通过；未宣称稳定夹持或离桌成功。
- 当前看 [v47实时监控](runs/tabletop_gpu_20261024_v47_env512/monitor.json)，IDE中的v18属于历史运行。[详细验证记录](docs/tabletop_gpu_20261024.md)。旧资产和检查点保留，无提交或上传。

## 2026-10-24 历史：反作用力闭环与接触后提升（v41已停止，本地未上传）

按用户确认，增加 `--adaptive-grasp --force-feedback`。PPO 最后两维改为每手目标接触载荷0.1–1.0 N；实际传感器反馈驱动限幅、限速的小步闭合，不足加压、过量回退，提升中持续滑动时提高有上限的目标，稳定后保持。这里载荷是每手两个指垫/物体接触力模长之和，并不等于已形成可靠对夹。目标力、实测力和保护限制分别记录在每环境monitor中。动作20维、观测136维，旧125维模型不混用。

- close/hold按实际载荷、速度和相对滑移连续稳定0.2/0.3秒后才能进入lift，6秒不能建立则失败；不要求四点接触。lift中载荷不足或滑动加剧暂停脚本上移，继续力反馈；原联动卸载、手掌运动约束及.02rad/30N/桌面硬保护保留，真实离桌30mm保持.5秒才算成功。
- v35闭合幅度对照与v36刚度20/40/60/80对照显示受载联动卸载，增大驱动上限的该次试验未解决抓取。2026-10-24厂家参数审计纠正：日志applied_torque来自隐式PD估算，不是实际驱动力矩读回，不能仅据此排除饱和；见 [V2参数核对](docs/dh116_v2_parameter_audit_20261024.txt)。
- v37(nf200)与v38(nf400)证实更硬联动提高了实际接触载荷，但软预警不应清除独立测得的稳定载荷。v39修正保持判据后能进入lift，仍因继续空提而失去接触；v40增加暂停上移后，前三组完成close→hold→lift、滑动时目标上调且接触保持。4个诊断回合均无硬故障，仍为0/4真正提升，实际离桌不到0.001mm。固定目标诊断不是PPO成果。
- v41已启动4环境全新PPO，先做12回合严格物理准入，计划65536采样步、每2048步存档、最后8回合评估。nf400只有这次独立对照证据，不沿用历史安全结论；任何硬故障仍停止。初始力动作均值-.95(约.1225N)，其余动作及探索保留，后续由策略学习。已通过12回合严格准入，首次运行核对3584步/14优化epoch、无硬保护停止；2048步检查点已实际重载验证合同一致、权重有限。历史状态见 [v41监控](runs/tabletop_vector_train_20261024_v41_reaction_force/monitor.json)。
- 软件：相关388项回归通过，随后新增滑移时序测试通过；代码审查修正一处导入权重路径的无效初始化后，受影响154项再次通过。编译与diff检查通过；不代表抓取成功或泛化通过。单Isaac、CPU0、数值单线程。
- v34为演示主动暂停，已保存26176步/186优化epoch；IDE里的v18是历史监控。实现方案及证据见 `docs/superpowers/specs/2026-10-24-reaction-force-feedback-design.md`。

## 2026-10-24 历史：高摩擦吸管 PPO 对照（本地未上传）

按用户要求增加 `--object-material high_friction`：吸管碰撞材质静摩擦1.2、动摩擦1.0、恢复系数0，摩擦合成max。这是仿真参数试验，同时影响吸管/手指和吸管/桌面接触。保留10.956g中空刚体、20D动作、奖励与成功标准，新建4环境对照模型。

- v33已完成65536步/472优化epoch，最终8次评估安全超时，成功0/8；已退出，不再运行。
- v34实际PhysX读取验证通过：4×32个碰撞段的静/动摩擦分别约1.2/1.0，材质合成模式正确。12回合物理准入全部通过，已开始4环境PPO，计划65536步并最终评估8回合；准入仍无成功提升，尚不能声称抓取改善。见 [v34当前监控](runs/tabletop_vector_train_20261024_v34_high_friction/monitor.json)。
- 材质参数写入检查点合同，旧材质模型不能直接resume。原资产及旧运行记录保留。当前相关测试222passed，编译和diff检查通过；单Isaac、CPU0、数值单线程。

## 2026-10-23 历史：中空吸管与 PPO 自主调整夹点/力度（本地未上传）

按用户新要求，新增 `--adaptive-grasp`：20D 动作由12个手指目标、双掌XYZ偏移6维和每手夹持刚度2维组成。125D观测包含实测夹点相对物体的位置、物体绝对离桌高度、掌位偏移和实际下发刚度。掌位限幅XYZ为±3/3/4 mm，限速2 mm/s；刚度10–20 Nm/rad、变化率10/s，力矩上限.20 Nm。关闭旧模式的自动掌位微调，奖励和最终成功仍根据真实物体高度/稳定性，不要求四点接触。旧12D模型不能直接恢复，保留旧检查点，新策略重新训练。

- 对象换为真正中空的刚性吸管代理：外径16 mm、壁厚.5 mm、长.5 m、质量10.956 g，32个独立凸环段保留管腔。实际PhysX质量/惯量与资产一致。其成绩不代表原120 g实心物体抓取合格。
- 新预张开初始化的实测夹点高度相对中心约−.31～−.54 mm，双手最低桌面净空约2.18/2.21 mm；复位高度扰动暂为0。闭合目标在线材直径两端，但初始化不伪造已接触状态。
- v32在Isaac启动默认关节检查阶段失败，PPO步数0；已修正adaptive启动时重复预张开导致右拇指低于USD下限的问题，旧模式行为保留。v33已通过实际场景初始化、隔离和12个准入回合，进入4环境PPO学习；首轮核对为1388个采样步、3个优化epoch，无物理停止。计划65536步、每2048步存档，最终8回合严格评估；预算尚未完成，准入回合均未抬起物体。见 [v33实时监控](runs/tabletop_vector_train_20261023_v33_hollow_adaptive/monitor.json)。
- 本次专项软件回归 **484 passed**。真实SB3假物理冒烟完成64步/2次更新与保存重载；不能替代Isaac抓取验证。物理准入、PPO更新和抓取结果分别记录；目前未宣称稳定提升成功。
- 旧v31因演示和本次修改已主动停止，保存12,736个新采样步/103个优化epoch；没有在后台继续训练。IDE中的v18监控也不是当前运行。

## 2026-10-23 历史：v31 protected 与此前多环境试验（本地未上传）

已新增真正共享物理时钟的多环境入口：4 套双 DH116＋刚性线材场景，独立接触、复位和奖励。v30物理频率为480 Hz、控制仍为60 Hz，每个控制周期8个全局物理子步，与环境数无关。PPO 控制手指，手掌由控制器执行，成功不要求四点夹持。保持单核/数值单线程和单 Isaac 实例。

- 2 环境实际隔离/重复复位检查及6个零/随机动作回合通过，均到达lift，但未抓起物体。
- 4 环境 v27 完成8192步训练后在评估中发生联动超限；v25旧单环境也最终在36268步失败，旧短程安全证据不能代替长期稳定。
- 同动作对照中，慢抬升仍失败、禁抬升安全；向量场景加入有限mimic柔顺（200 s⁻¹、阻尼比1）后，原4环境失败序列及延伸回合无硬故障。保留.02 rad硬保护；这是仿真模型调整，尚非实机标定。
- **v28 已完成柔顺版本4环境短程验证：8,192 个总采样步、69 个优化 epoch，随后8次确定性评估全部无物理安全故障，exit 0。抓取成功仍为0/8，8次均超时。** 结果为 `completed_budget`，不代表抓取成功或任意后续动作均稳定；见 [v28 实测结果](runs/tabletop_vector_train_20261023_v28_4env/summary.json)。
- **v29 已因实测物理故障停止：累计33,996步、本轮新增25,804步、287优化epoch。** env2右食指finger22联动误差−0.0247419 rad超出原0.02 rad硬阈值，exit 2；检查点与计数已保存，计划预算及最终评估未完成。介入原因是物理超限，不是低奖励；见 [v29最终报告](runs/tabletop_vector_train_20261023_v29_4env/summary.json)。
- 原失败动作在480 Hz下完成4环境完整回放，手掌实际上升约45 mm；随后原v29策略在480 Hz下完成 **8/8次安全确定性评估、抓取0/8、exit 0**，控制帧峰值联动误差约.01018767 rad。100 ms稳定等待和快恢复两种方案已拒绝，未部署。以上是候选物理配置的有限验证，不保证长期稳定或抓取成功；见 [闭环评估](runs/tabletop_vector_evaluate_20261023_v29_480hz/summary.json)。
- **v30 已通过准入并实际训练，但在612个新采样步、5优化epoch后再次物理超限停止，exit 2。** env2训练回合3的右finger22误差约−.020618 rad；模型和计数已保存，262,144步预算及最终评估未完成。来源v29的33,996步与本轮新计数分开记录。480 Hz回放/确定性评估安全未能覆盖随机训练探索，不能声称升频已解决问题；见 [v30停止监控](runs/tabletop_vector_train_20261023_v30_4env_480hz/monitor.json) 和 [结果](runs/tabletop_vector_train_20261023_v30_4env_480hz/summary.json)。
- 外部运动学锚点原型初始化修正后，速度与跟踪正确（记录最大误差约17.3µm），仍在原v30动作第152控制帧触发同一mimic保护，未部署。保持原后端，仅将mimic频率200→400 s⁻¹：v29/v30各自失败动作前缀加零动作延长均完成901控制帧、4环境安全超时，各手root_offset到.05m，物体仍未抬升；60Hz采样峰值误差分别.0163013/.0184418rad，不能视为完整子步峰值或抓取成功。
- 已实现严格的200→400 s⁻¹显式权重导入，但nf400对v30权重的 [闭环评估](runs/tabletop_vector_evaluate_20261023_v30_nf400/summary.json) 在149控制帧/596总transition因env1联动误差.0215569431rad失败，exit2；计划8回合未完成。该帧力约.8047N、关节速度最大约.195rad/s且数值有限，仍是原.02rad保护下的真实失败。**nf400训练方案仍被否决；v31使用nf200。**
- 单环境故障复位 [临时原型实测](runs/tabletop_protected_mimic_reset_probe_20261023/summary.json) 完成2869个向量控制步、20次有限mimic故障即时终止/复位，29次复位隔离检查全部精确不变。[正式模块探针](runs/tabletop_protected_mimic_production_probe_20261023/summary.json) 随后完成2259个向量控制步、10次实际故障（目标9次，同帧并发达到10次）、17次隔离检查全部通过，含1次末子步复位后的下一真实物理步校验，exit0。这些仍是失败回合，保留原终态和约−25奖励，不是无故障或抓取成功。
- **v31 protected正式入口已启动，首次核对monitor为`admission`，新增PPO步数/优化epoch均为0。** 配置为4环境、物理480Hz/控制60Hz、nf200，计划262144个新transition、8次最终评估、每2048步存档。从v30导入相同权重，优化器与计数重新开始；来源612步单独记录。只有通过原保护复验的有限mimic失败可终止本回合后继续其他环境，其他故障仍全停；原.02rad阈值和严格评估保留。后续准入/学习状态以 [v31实时监控](runs/tabletop_vector_train_20261023_v31_4env_protected/monitor.json) 为准。
- 主线程本次完成相关软件测试 **390 passed**；未据此声称全仓库通过或长期物理稳定。复位seed保留240 Hz来源，权重迁移须显式启用并严格核对其余契约；旧模型不能直接resume到480 Hz或新的失败处理合同。无上传。

详细证据、命令和限制见 [多环境 PPO 记录](docs/tabletop_vector_ppo_20261023.md)。当前仍无稳定抓取成功证据。

## 2026-10-23 历史：单环境联动恢复与续训（v25 已停止）

修正了两类联动失稳因素：预警时按实测从动关节反算主关节恢复目标；direct 模式无驱动从动关节的求解器限速由 0.3 调为 3 rad/s，主动关节仍为 0.3、指令速率仍为 0.05 rad/s。原 0.02 rad 硬误差、力矩、碰撞保护保留，训练继续不要求四点夹持。

- 原 v22 和 v23 失败动作分别回放复现；逐项对照后，最终配置完整回放无硬故障。仅修正恢复目标的 v23 仍在 3,754 步失败，保留该反例，不把单段回放当作全面稳定。
- **v24 实际完成 8,192 步 / 244 优化 epoch，加 5 次确定性评估，全部无物理安全故障，exit 0。抓取成功 0/5，物体仍未离桌。** 该结果证明本轮短程探索和评估可以完成，不保证任意后续动作稳定。
- 当时从 v24 检查点启动 v25 串行自动续训：每轮新增 65,536 步，最多 8 轮；安全预算结束才继续，物理故障即停止。历史监控（最终36268步物理失败）：`runs/tabletop_direct_continuous_20261023_v25/monitor.json`，旧 v18 监控不再代表当前训练。
- 增加真实动作回放入口、direct 续训参数与 seed/pose SHA 校验；改变恢复/速度配置后不兼容旧检查点。专项九文件 **186 passed、1 skipped**，编译与 diff 检查通过；未重跑全仓库。

复现命令、根因证据和限制见 [联动修复记录](docs/tabletop_mimic_recovery_20261023.md)。没有提交或上传源码/运行产物。

## 2026-10-23 历史：直接 close 与物体支撑验收初版（v22 已停止）

按用户最新要求，训练跳过 approach，从实测 close 初态复位。PPO 直接输出 12 个主动关节目标，目标位于标定姿态附近 ±0.08 rad 并受耦合关节范围、0.05 rad/s 速率约束；脚本只负责手掌小幅调整和抬升。**取消四点接触要求，接触方式由策略学习；目前尚无稳定夹持或离桌成功证据。**

- 新策略采用有界高斯均值与较小初始方差，避免旧策略原始均值漂至约 −10～+11 后全部被动作裁剪。新模式为 108D，包含手掌调整量与更清晰的接触力尺度；与旧残差模型契约不兼容，重新训练。
- close 2 秒、hold 0.3 秒之后尝试 lift，不把定时切换算成功。实际物体底部达到 0.33 m、速度低于 0.03 m/s、相对滑移不超过 0.02 m 并保持 0.5 秒才通过；不依赖四个 pad 同时接触。评价范围为 close→lift，不代表重新验证 approach。
- 接触奖励使用总载荷进展，不奖励一直停在接触阈值以下；物体没有离桌时不计已夹持物体的滑移惩罚。耦合预警时仅卸载对应主关节，其他手指仍可调整；原 0.02 rad 硬耦合、30 N 力、穿桌/掉落保护保留。
- 实际 v19：3 次 direct-close 复位/探索探针通过；v20：3 次均到达 lift 尝试且无硬安全异常，但物体未离桌。v21 在 1,048 步因右食指 mimic 超限停止；v22 将预警提前到 0.004 rad，仍在 1,957 步停止（实测 −0.020175 rad）。均保存检查点，均未完成计划预算或抓取验收，当前没有继续自动训练。最新结果：`runs/tabletop_direct_object_train_20261023_v22/summary.json`。
- 奖励 v3 仅惩罚超过 20 mm 的滑移，避免接受的夹持姿态反而低于空抓回报；修正 object-stability 的 close 成功报告。本轮专项七文件测试 **134 passed、1 skipped**，两项回归已验证旧行为失败、新行为通过。完整物理抓取仍待证明，全仓库测试未重跑。下一步需解决抬升接触切换后的联动误差，不能靠延长 PPO 预算掩盖。源码和运行产物均未上传。

## 2026-10-23 历史：v18 PPO 自动续训（已停止）

v18 前五轮均 close 0/5，第六轮中断，最后周期检查点为 337,920 步；该轮没有完整 summary，不能声称干净结束。下方保留此前配置说明，不代表当前训练。

当时配置为双 DH116 手＋刚性圆柱代理，PPO 控制 12D 手指残差，手掌由控制器执行。v18 自动续训原计划每轮每阶段新增 65,536 步，最多 8 轮，close 达标后自动进入 lift；已停止，未完成稳定 close/hold 或 lift。

- 修复接触丢失后手掌不再下降、物体移动后目标位置失配、耦合警告时 XY 仍移动等问题；预备关节姿态、实测目标跟踪与 5 mm 预抓取交接已接入。交接就绪使用物理步后的测量，不能冒充四点接触成功。
- 观测为 `dh116-tabletop-v4` / 104D，包含 10 个 mimic 误差。残差半径仍为 0.02 rad；没有修改摩擦、硬安全阈值或最终抓取标准。旧 47D/82D/94D 模型不能直接恢复。
- 实际 Isaac：v16 累计 81,920 步，close 0/5，最终评估卡在 approach；v17 显式迁移权重并重置优化器与计数后新增 8,192 步、307 优化 epoch，5 次评估均进入 close，但 close 成功仍 0/5，未触发硬安全停止。两组步数不能当作同一连续优化器计数。
- v18 从 v17 检查点恢复，后台进程自行串行续训、定期保存模型并评估；只对安全的预算结束自动继续，物理异常、程序错误或源码变化会停止。运行状态在本地 `runs/tabletop_continuous_20261023_v18/monitor.json`。运行中不修改控制源码，也不要启动第二个 Isaac 实例。
- 本轮专项六文件测试：**116 passed、1 skipped**。测试覆盖阶段交接、训练/恢复和续训调度；合成测试不代表物理抓取成功。全仓库测试未重跑。

实际命令、历史失败和检查点契约见 [PPO 训练说明](docs/tabletop_training_20261023.md)。运行产物不上传；下方保留历史阶段。

## 2026-10-23 阶段快照：`10-23`

当前分支基于 `10-22`，上传目标为 `ssdsj12/dexterous-hand-dh116`。本次保存当前工作区的源码、测试与方案修改，保留历史双臂工具；当前桌面实验入口仅生成双 DH116 手、桌面和刚性线材代理，不添加机器臂。

- 新增 hand-only MPC 参考残差 PPO 实验入口、阶段奖励、安全检查、运行时 mimic、姿态/预览同步，以及候选抓取、预载与恢复规划模块。模块存在不代表完整物理闭环已经验证。
- 当前修正：张手初始化、按下降距离平滑靠齐参考关节姿态、逐物理子步移动根部、接触停止锁存与桌面净空约束；桌面入口默认 PGS，控制 60 Hz、物理 240 Hz。奖励和安全阈值未放宽。
- 本地实际 Isaac 证据：`tabletop_open_guarded_pgs_20261023a` 与 `tabletop_open_guarded_default_20261023b` 均运行约 15 秒，在接触后保持并最终超时。最低净空 8.86 mm，记录最大 mimic 误差 0.00613 rad、指尖力 0.78 N。仅验证该确定性下降/保持轨迹，未完成 close/lift，PPO 更新数为 0。
- 本次上传前重新执行 `bash scripts/python.sh -m pytest tests/test_dh116_tabletop_task.py tests/test_dh116_runtime_mimic.py -q`：9 passed、1 skipped（Isaac schema 插件依赖）。未重跑全仓库测试，不声称全量通过；其他模块作为开发快照保留。
- 下一步：接触后的闭合/提升与探索动作稳定性、独立训练准入条件；现有训练门槛仍要求成功基线，尚未改成仅要求物理稳定。

`runs/`、`archive/`、运行日志、缓存和密钥不上传。新入口的 `--pose-summary` 仍需本地运行产物，部分资产路径绑定本机；此快照不保证新机器克隆后直接复现。详细记录见 [稳定性调查](docs/tabletop_stability_20261023.md)。下方为历史阶段，不代表当前 hand-only 已抓取成功。

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
