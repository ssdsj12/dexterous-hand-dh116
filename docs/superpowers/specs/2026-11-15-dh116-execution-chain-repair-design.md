# DH116 全执行链审查与分包修正设计

2026-11-15补充：用户已批准采用后续PPO与奖励文档的新版本方案；Teacher输入、训练奖励与准入依赖以[原生Teacher修订](2026-11-15-dh116-native-teacher-ppo-reward-amendment.md)为准。以下原始已审设计作为版本记录保留，其余R01–R08约束继续有效。

日期：2026-11-15（UTC）
审查基线：`74e9404aa207be0311de7f963dbd6bb49499db5f`
仓库：`ssdsj12/dexterous-hand-dh116`
目标分支：`codex/dh116-stereo-privileged-dagger`
状态：用户已选择“现有执行链分包修复”；本书面规范等待审阅，尚未开始实现。

## 1. 目标和当前事实

Teacher、视觉和 Student 同等深度审查。目标是让现有架构的真实执行链、来源校验和训练数学一致，再在有证据的柔性 Pinch、Tripod、Power 环境进行分抓法有界 Teacher PPO 验收。保留原有算法、模型身份和成功的控制路径。

本轮语法解析覆盖 1913 个 tracked Python 文件，无语法错误；17 个现有相关 CPU 测试文件为 262 passed。该结果没有覆盖本次新增发现的组合错误，不能证明系统已能训练。Ubuntu integration、Windows 发布裸仓库和 GitHub 基线一致；PR #2 为 Draft。528 个保护文件和两个历史 Checkpoint 校验通过。

目前存在三类不同结论：
- 确认缺陷：BC 两种 Actor 模式崩溃、分类损失语义分叉、原始数据与训练张量绑定不足、Student matcher 准入缺项、非中心主点投影符号、异常清理与封存层错误。
- 未完成链路：真实无相机 collector、当前 backend 的新模型合同、完整有效状态来源和正式数值 R0 尚未闭环。
- 外部或证据限制：原生 GPU 资源、各抓法原生回执及成功数据。等待其中一项不能阻断独立离线修复。

## 2. 已选择方案及替代方案

| 方案 | 影响与代价 | 结论 |
|---|---|---|
| 现有执行链分包修复 | 复用现有 PPO、Actor、源读取器和保护机制；按端到端故障建立回归 | 用户选择 |
| 仅修显式崩溃 | 变更较小，但真实 collector、数据资格与奖励仍不能支撑训练 | 不作为本次完整目标 |
| 整体框架重构 | 需要迁移大量历史身份和证据，扩大回归范围 | 当前没有必要证据 |

不做无关重构。只提取需要多个入口一致使用的功能：Actor 特征构造、字段损失分派、原始数据重放绑定和 Stereo 准入比较。

## 3. 保持的合同

1. 原 371D 公共状态索引、单位与语义、canonical 24D 动作、Tanh Gaussian、既有 PPO/GAE 保留。
2. 现有 3844D Actor/371D Critic、旧 Teacher/Student/MPC、历史 Checkpoint、已批准物理/材料/执行器参数保留。
3. 原生成功要求保持 Lift >=30 mm、连续 Hold >=5 s、固定锚 Slip <=0.5 mm、双侧有效接触且无额外支撑/附着/外力。
4. R2 和未资格化 Force Proxy 保持关闭；仿真估算与真实硬件测量分开。
5. 缺失字段以 validity=false 和明确未知语义表示，不以补零获得资格。measured66+rod 只作为独立身份的诊断模型，不能称为完整 371D Teacher。
6. 刚性代理与柔性任务分开；观察到接触不能直接授予抓取、Hold、Teacher 或 Student 成功。
7. 正式 BC/DAgger 依赖合格成功 Teacher 数据；Teacher PPO 探索训练只依赖真实源、Reset、状态动作、数值奖励和预算资格，不以已学会抓取作为循环前置。
8. 新预处理、损失版本、模型配置及训练会话独立保存。旧模型推理不能因新模式而静默变化。

## 4. 执行链和边界

### 4.1 物理与 Teacher

原始资产/关节约束 → 当前实例 Reset → 同钟状态/接触/健康读取 → 371D 逐字段来源与柔性状态合同 → 原 Actor 采样 → 实际 issued action/SDK target → 480Hz 原始反馈 → 每个60Hz窗口一次 R0 → 原 rollout/GAE/PPO。

每个环节绑定 env、episode、reset、policy/physics tick、时间、实例、source profile、代码配置 SHA。terminal 保存终止前最后有效数据，再 Reset；bootstrap 不得读取下一 episode。

### 4.2 视觉与 Student

同步左右 RGB → 明确标定 K/T → 原色默认基线或独立 opt-in 分割/匹配 → 中心线与 observed_surface 分池 → 掌局部/世界辅助输入 → 现有状态估计器 → TaskStateAdapter → latent/estimated_task_state/hybrid Actor → canonical24。

BC、DAgger 和 PPO 必须复用同一 Actor 输入构造，辅助状态监督复用同一分类/回归/旋转数学。GT、Reference Depth 和 Critic 特权输入只用于标签/训练评价，不进入 Actor。

### 4.3 数据资格

证明文件 → 原始 artifact 字节 SHA → episode/frame/reset/tick 行选择 → 实际 RGB/状态/Teacher 标签/issued action 数值 → 数据集张量。

保存数据集自身摘要只能证明该文件未变，不能代替其与原始数据的关系。Teacher 和 Student 使用同一原理、各自 typed 源读取器。独立初态及其所有重放、Source Family 保持同组拆分。

## 5. 确认缺陷和独立工作包

| 编号 | 缺陷与位置（相对仓库根） | 修正范围 | 验收 |
|---|---|---|---|
| R01 | `tabletop_stereo_bc.py:63` 绕过两种非 latent Actor 的特征构造 | 在 Actor 上复用一次 estimator 结果的特征接口；BC/DAgger 调用该接口 | 三模式各一次真实 CPU optimizer 更新；动作损失梯度进入相应估计分支；原 latent 确定性输出不变 |
| R02 | `tabletop_stereo_losses.py:177` 的 371D 分类维用 Smooth L1，部署做 sigmoid | 按合同分派分类 BCE、回归与旋转；与独立状态监督入口共用数学 | 所有20分类维正/负类梯度；未知标签零贡献；NA/旋转/力学门控保留 |
| R03 | `tabletop_privileged_dataset.py:52` 未将训练张量重放绑定源 | 复用 expert assembly 的真实状态动作 join | 同 proof 下改 observation/action/mask/行序拒收；真实完整源可重建；旧3844D不得变成371D |
| R04 | `tabletop_stereo_supervision_dataset.py:60` 未将 RGB/标签数值绑定 raw | 验证 source artifact、行选择、同步身份和实际监督值 | 改像素、动作、frame、reset、tick、初态组均拒收；成功池与失败 DAgger 池隔离 |
| R05 | `tabletop_stereo_student_cli.py:291` 正式 PPO 仅比较标定 | 复用 collection 的 matcher/processing 准入规则 | 同 K/T 换 matcher 被拒；仅资源批次变化按原合同可接受；已绑定 actor package 仍需源资格匹配 |
| R06 | `scripts/dh116_pinch_acquisition_v2.py:444` clear 失败跳过 STOP 后继续 close | 设计独立清理阶段、保留第一异常和每阶段状态，复用已有所有权观察 | clear/STOP/close 各失败有 CPU 顺序和引用测试；原生需新配方自然退出验证，CPU不得授予生命周期资格 |
| R07 | `tabletop_stereo_camera.py:66,96` 非中心主点 aperture 符号错误 | author/readback 使用真实 USD 投影的互逆公式 | Gf 实际投影与指定 K 一致；非中心及中心、左右、分辨率组合；旧中心默认保持 |
| R08 | `scripts/run_dh116_mesh_reset_numeric_pinch_work_packages.py:28` 层级变更导致 sha 属性缺失 | 明确源/哈希依赖的独立新封存版本 | 复现旧错误；重开保留账本可验证源/时钟/预算；篡改拒收；原失败回执和消耗预算不改 |

以上不表示已修复。来源绑定漏洞的隔离复现明确假设下层原生证明已合格，只证明上层缺少数值关联，未证明当前原生资格可被绕过。

## 6. 实施依赖与并行

A 阶段：R03/R04 原始数据绑定、R05 准入一致性、R06/R08 封存和异常清理。各模块可并行；公共数据校验接口先固定，避免两个 Agent 改同一核心文件。

R06 的异常收尾准则：clear 失败后释放调用侧强引用，独立尝试受控 STOP，原异常和每个收尾异常分别保存。正常 close 必须有冻结、STOP 和拥有资源释放的读回；条件未证实时只能走单独记录的故障关闭尝试，不能写正常释放或生命周期成功。故障路径不得启动第二实例、不得使用强制退出伪造结果；具体调用顺序必须先经异常注入测试，再由对应新原生配方验证。

B 阶段：R01/R02 Actor 与损失、R07 标定投影。依赖 A 的源绑定完成后进行正式数据重放；CPU fixture 数学回归可以独立执行。继续保持原色基线和已有正确采样槽；分割新候选单独记录，5mm 固定判据和分母不变。

C 阶段：真实 Teacher 主链连接。
- 逐字段核实完整371D所需来源与当前 flexible backend。每个字段必须给出可运行读取器、坐标/单位、validity、归一化和同钟证据。无法证明的字段保持未知，完整371D准入保持关闭。
- 独立模型/session 配置绑定当前 backend 与实际可接受证据，历史配置不改。源配置变化使对应资格失效。
- 在独立 R0 版本明确系数、单位、60/480Hz 聚合、阶段适用性、未知测量和终止语义；只有数值来源和正式配置验证后才能向 PPO 提供 scalar reward。历史 null 配置保持。
- 实现真实无相机 collector 工厂，组合现有 state/action owner、原 rollout、原 GAE/PPO。不能将 CPU fixture collector 改名为 actual。
- 每个控制窗口采样后验证实际 issued action；奖励恰好结算一次；on-policy 身份在更新前复查；保存/恢复含 Policy、Adam、RNG、计数和合同。

D 阶段：分抓法原生验收。
- Pinch、Tripod、Power 各自独立 source/pose/config/receipt。失败只影响对应后继。
- 先验证真实 Reset、物理源、动作执行、terminal 和自然关闭；已消耗旧配方不能重跑。
- 后续 PPO 单更新预算必须显式列 env、control/physics steps、账本行数、墙钟、更新数和恢复检查，不能冒充原0-update接触预算。
- 通过短诊断只表示可执行训练，不表示 Teacher 已学会抓取。冻结策略完整评估与 Student 数据资格另行验收。

## 7. 资源和证据纪律

仅 Ubuntu 开发/运行，Windows 导入同 commit 发布；PR #2 始终 Draft，基线11-10只读。单 Isaac 实例，先检查真实 GPU 占用和锁；不停止其他训练。每个新原生入口冻结 SHA 并核对对应授权；本设计不扩充已消耗原生预算。

180秒目前是资格截止而非保证杀进程的硬执行上限；后续实现须把超时观察、受控取消、自然退出和资格失败分开，不能用强制退出伪造自然关闭。5秒Hold至少2400物理步/环境，不能由240步诊断授予。

原始账本、GDB、模型和数据保持私有原位置只读引用。报告仅发布必要摘要、源/结果 SHA、复现命令、失败事实和资格范围。

## 8. 验证和发布

1. 每个修复先增加对应失败回归，确认失败原因，再修改最少生产代码。
2. CPU 验证覆盖三 Actor 模式、两个监督入口、Teacher/Student 数据加载入口、raw RGB 推理与 formal admission，不只测独立网络 forward。
3. 原生证明分别记录 GPU计算、PhysX/杆耦合、源完整性、自然退出、PPO更新和抓取成功。
4. 修复训练数学或预处理后使用新版本合同和独立模型；不迁移或覆写历史 Checkpoint。
5. 每包核对528保护文件、两个历史Checkpoint，明确暂存范围，更新报告/readiness/CHANGELOG/必要README，Ubuntu commit→Windows push→读回同SHA和树→发布清单。
6. CPU诊断包 passed 只表示该CPU范围通过；native、Teacher、Student准入保持各自证据门槛。
7. 调度器 CPU 依赖陈旧状态作为单独状态一致性审查项；当前原生规划/发布已有递归receipt校验，未证明原生绕闸，不按绕闸修复。

## 9. 审阅后下一步

用户审阅本书面规范后，调用 writing-plans，逐包列出修改函数、失败测试、命令、依赖、验收和提交范围，再进入实现。设计审阅前不修改生产源码、不启动新原生或训练。

自审：无未定占位符；“可训练”与“已成功”分开；正式371D和measured66诊断分开；R0诊断与正式scalar分开；源码/资源/实验授权与发布授权分开；未将任何测试或历史结果改写为新增原生成功。
