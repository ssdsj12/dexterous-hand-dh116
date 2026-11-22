# Tripod v9真实更新与父墙钟拒绝证据
实测代码fcb22947d332dcfd788a70615cdb982e87fb8da9；GPU3、1实例、1环境，唯一attempt-r8lnz6af。
指纹8b369481554184fd9ee1e39f73e5fa41838eeb015bd4be413e8085133a1217cb。该指纹已消耗，不重跑、不以其失败Checkpoint恢复。

## 本次原生实际结果
- 2初始化+64测量SDK步=66；8条transition、64Rope调用、768内部solver子步。SDK3..66连续，64行原/新物理guard均None。
- 真实Reset完整、初始clearance通过；原371账本305NA/false不伪填，Teacher实际输入为66公共+41节点+context。
- 同钟输入审计8decision+8bootstrap=16个source pack通过。camera-free小诊断，无新双目图像或Student成功。
- 1次optimizer update、1个accepted minibatch，参数L2变化0.003915731512910579，Checkpoint policy_version=1。
- KL Guard首候选Exact KL0.13438794，回滚Policy/Adam/RNG后候选0.001342237，实际接受lr1e-5；target0.02未增大，未调探索噪声。
- 检查点teacher_clean.pt仅保留失败目录，不上传、不计成功专家、不允许后续resume。
- 更新后父wall检查拒绝，整次诊断失败。Generic生命周期结果内固定ppo_updates=0不是训练计数；实际更新以optimizer_update_accepted、sealed journal和checkpoint counters交叉核对为1。

## 墙钟与退出必须分开
更新前余34.373323s；8个不重叠控制窗口合计36.188891s，仅控制窗口已至少超1.815568s，尚未计更新/封存开销。
旧代码在校验拒绝时没有写after_update原始样本，因此不虚构精确负remaining或完整update阶段耗时。
外层196.260444s <240s，但内层父deadline未通过，不能借外层预算放行。
App.close正常返回，stop/close各阶段无清理错误，stat cache清一次后0项；after_app_close全部14个tracked weakref失活。
外层/Inferior均自然exit1，signal=None，没有SIGSEGV、SIGTERM/SIGKILL、os._exit或强制清理；相关进程已消失。
本次没有SIGSEGV不等于整个系统已根治；native_release_verified仍false，自然exit0仍false，不授予Teacher/抓取资格。没有崩溃就不编造新的C++崩溃栈。

## 实际代码修复与CPU证明
tabletop_native_teacher_actual_provider._record_actual_wall_budget保持原4拒绝条件，拒绝前写*_rejected：
保留原始wall_readback，记录phase、actual SDK、episode和具体rejection_reasons，native/Teacher资格显式false。
沿用原NativeJournal逐行flush/fsync。成功事件结构不变；写盘失败仍抛预算ValueError并保留OSError cause，不吞拒绝。
新增5个行为用例：4种拒绝+写盘失败。旧函数先4失败（没有持久证据）；修改后联合173通过0失败（50.22s）。
这项新日志修复本包0次原生，不能把刚才旧代码的v9结果写成新代码已经实测。
独立只读审查无重大问题；Antigravity额度不可用，无新回复。

## 证据与下一项
readiness列出真实日志/Checkpoint/receipt SHA，私有readback SHA b4eb8747f686d2a7e26e9fb7ac0ca2a351f647df6fd51f8cbba37487b01f670d。
原始JSONL约29MB、模型约2.5MB、日志全部留大盘，不入Git；528保护文件、3旧Checkpoint、双RGB/物理/动作/成功阈值不变。
下一项剖析真实控制窗口：observer合计约15s，并有source/pre/post阶段；只凭实测热点修复，不关闭来源核验、逐物理步账本或fsync来过预算。
三个抓法仍未取得完整柔性Lift30mm/Hold5s/Slip0.5mm成功资格，R2/ForceProxy与正式BC/DAgger门控保持关闭。
