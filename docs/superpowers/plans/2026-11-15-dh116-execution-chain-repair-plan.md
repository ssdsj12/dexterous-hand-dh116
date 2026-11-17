# DH116 Execution Chain Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复已证实的8项执行链缺陷，建立真实Teacher训练的可执行准入审计，随后按完整来源与原生授权逐级接通Pinch、Tripod、Power。

**Architecture:** 复用现有PPO/GAE、Actor、源读取器与Windows发布；只提取多入口必须共用的特征、损失和来源校验。冻结历史native代码与回执，新增生命周期和封存版本。按用户新批准文档建立独立66项公共字段＋rod＋task_context Teacher，先接线profile完成两次更新与恢复，再按来源解锁完整奖励。

**Tech Stack:** Ubuntu Python3.10/PyTorch2.7、pytest、USD Gf CPU、Isaac/DeformX原生验证；Windows只负责导入同commit发布。

---

## 1. 审阅依据、当前位置和计划边界

书面设计已获用户确认：
`docs/superpowers/specs/2026-11-15-dh116-execution-chain-repair-design.md`。

代码基线为 `5b10a4c2f457e965e24ec79ce7033374d46c1852`。审查代码与`74e9404`相同，之间仅文档/发布记录变更。审查报告为`docs/superpowers/reports/2026-11-15-whole-chain-code-review.md`。

已运行：1913个Python AST、262项主链CPU、53项调度发布CPU；这些是基线证据，不能替代新增失败回归。8个缺陷尚未修复，真实Teacher PPO更新仍为0。

用户随后明确批准采用《DH116_PPO_详细修改与奖励设计.docx》的新版本方案；补充规范为 `docs/superpowers/specs/2026-11-15-dh116-native-teacher-ppo-reward-amendment.md`。本轮规划保留R01–R08，并以实际66公共字段、41x13真实rod及独立task_context的新模型替代“所有371槽必须有限”的前置。原371账本305项NaN/false及旧模型保留。两个奖励profile已有明确建议数值，仍需实现、数值回归和实际测量资格；不能声称权重已原生验证。新collector、两次更新恢复及三抓法原生证据是仍需完成的任务，不以先抓取成功为循环前置。

## 2. 子计划和文件边界

| 子计划 | 责任 | 独占核心文件 |
|---|---|---|
| `2026-11-15-dh116-lifecycle-camera-sealing-plan.md` | R06/R07/R08 | 新生命周期v3、相机投影、新保留账本复核 |
| `2026-11-15-dh116-student-vision-data-plan.md` | R01/R02/R05 | Actor特征、BC、分类损失、共同Stereo admission |
| `2026-11-15-dh116-student-source-binding-plan.md` | R04 | Student原始records与训练样本绑定 |
| `2026-11-15-dh116-teacher-source-activation-plan.md` | R03/C01及C02–C05门槛 | Teacher专家source join、逐字段来源和生产准入 |

各子计划包括精确文件、失败测试、最小代码、回归和提交范围；不能只按标题创建空模块。

当前未提交的Pinch重审、Tripod/Power原生入口和contact provider保留。采用其中代码前先核HEAD/status/diff，不能用复制覆盖当前用户文件。尚未集成入口不被本计划授予资格。

## 3. 环境与工作区

- [ ] 读取最新远端HEAD/PR#2，确认Ubuntu integration、Windows relay与GitHub一致，PR仍Draft。
- [ ] 读取当前AGENTS与私有保护清单。为每个源码工作包创建Ubuntu独立任务worktree，使用`codex/repair-rNN-20261115`分支；不在Windows创建开发worktree。
- [ ] 每个Agent只修改自己的文件；公共`tabletop_stereo_training_cli.py`的R02/R04/R05串行集成。
- [ ] 运行环境统一使用原wrapper，避免直接Python缺USD造成假回归：

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py
```

该命令适用于CPU阶段；原生worker只能按其冻结配方设置GPU。不能把CPU的空CUDA设置复制到原生配方。

## 4. 执行顺序与可并行单元

### A：先保护来源、关闭和资格

- [ ] R03 Teacher专家来源重放，R04 Student源绑定并行。
- [ ] R06故障生命周期和R08保留账本独立复核，在互不改冻结文件的任务树并行。
- [ ] R05共享matcher准入，待R04对公共CLI的修改集成后处理。
- [ ] 每包先保存实际红灯，再最小修复、绿灯、检查保护与独立发布。

R03/R04的CPU下层证明double只用于定位上层数值绑定；报告必须标记它们不是原生成功或资格绕过实证。

### B：修训练数学与标定

- [ ] R01三种Actor的BC/DAgger组合；精确保留latent数学。
- [ ] R02统一20分类维、回归和旋转数学；与R01顺序集成同一BC模块。
- [ ] R07相机非中心主点Gf投影可独立实施。
- [ ] 运行涉及的旧回归。改变训练数学或预处理使用新合同/新输出目录；历史模型在其冻结代码中保留，不在新代码下静默接受旧源SHA。

### C：Teacher新版本真实主链

- [ ] C01建立不可变BackendProfile与resolved_reset_manifest；读取实际二进制/资产/源身份，不重贴旧2fea模型为881701。
- [ ] C02建立独立模型输入schema：common371/common_valid账本、真实rod/rod_mask、task_context/context_valid；只将已资格化有限项编码到模型。
- [ ] C03建立approach_wiring_smoke_v1和flexible_train_r1_full；前者只要求可靠距离、命令、时间和共同健康/原约束，后者严格要求全部适用Hold/slip/承托来源。
- [ ] C04实现真实factory：静态核验→唯一owner→本次Reset和源读回→新policy/trainer/reward→actual collector；当前策略请求、实际提交目标和实际运动分层记录。
- [ ] C05复用原PPO/GAE与KLGuard，在clean boundary保存新checkpoint；additional_updates从加载计数累加，完成第一次update1和第二次启动update2。
- [ ] C06冻结独立原生预算后逐抓法验收，报告真实源、有效transition、接受更新数和有限非零参数变化；完整抓取学习与成功另验收。

371D账本语义保持；未知305字段不补零。新网络选择66项不是声称全部371项已测得。未接触、掉落或尚未抓稳在来源可信时是正常探索状态；数值/来源损坏才隔离，不能用巨大负分代替未知观测。

### D：分抓法原生验证

- [ ] 三种抓法分别核实新配方SHA、对应授权、单实例锁和GPU真实占用。
- [ ] 第一批只验证新生命周期/源和Reset，失败不重跑相同指纹。
- [ ] 接触0-update预算不能用作PPO更新预算；新预算列环境数、控制步、物理步、solver子步、行数、更新数与墙钟。
- [ ] 保存实际动作、reward、terminal和pre-reset endpoint；自然退出、外层exit0和恢复各自校验。
- [ ] 短PPO通过只授予训练链可执行；Teacher成功、成功BC池、DAgger及独立Student成功仍分别验收。

240步不是5秒Hold。480Hz下5秒至少2400步/环境，成功标准保持30mm/5s/0.5mm和无外部承托。

## 5. 每包完成检查

- [ ] 确认失败测试失败于真实缺陷，不能把ImportError或假fixture当红灯。
- [ ] 实施子计划最小代码，执行新测试及相关旧回归；记录实际passed/failed/skip、命令和日志SHA。
- [ ] 核对528保护文件和两个历史Checkpoint；`git diff --check`及明确暂存清单。
- [ ] 报告具体修改函数、CPU/原生范围、已知失败和下一依赖。readiness中的`passed`必须附范围；Teacher/Student/native资格不能由CPU布尔授予。
- [ ] Ubuntu commit，排入既有Windows relay；Windows只导入同commit、正常push、读回commit/tree/files与Draft PR。
- [ ] 发布manifest另一个commit，同样Windows发布；核对待发布队列清空和三端同HEAD。
- [ ] 完成后重新计算就绪工作；只阻断真正依赖失败或资源的后继。

## 6. 端到端验收矩阵

| 要求 | 必须出现的证据 | 不足以通过 |
|---|---|---|
| 三Actor训练 | BC/DAgger各模式一次更新、梯度、一次encoder、latent精确回归 | 独立forward成功 |
| 371分类 | 20维正负类BCE与梯度、独立监督一致、NA/旋转门控 | 总loss下降 |
| Teacher/Student源 | 同proof改tensor/像素/动作拒收、实际raw重放join通过 | dataset自身SHA |
| Stereo准入 | 同K/T改matcher拒收、资源批次例外明确 | calibration相同 |
| 相机 | 实际Gf投影回到指定主点；左右/分辨率组合 | 自己author再自己反式readback |
| 生命周期 | clear/STOP/close故障分别记录；新原生自然退出 | Python对象del或CPU回归 |
| 封存 | 新独立复核重开原SHA/时钟/预算；原失败receipt不改 | 事后将原status改passed |
| Teacher新输入 |371账本mask、66有效公共字段、真实rod与版本化context；新旧模型身份拒错 |305未知补零或静默改旧模型 |
| 训练奖励 |两个独立profile、建议数值/单位/聚合/来源；每窗口一次 |改旧null或将fixture reward改名 |
| 真collector |真实source/实际动作/terminal/原on-policy链闭环 |CPUFixtureCollector改名 |
| 原生PPO |各抓法一次更新与完整身份恢复 |共享一个抓法CPU结果 |
| Teacher/Student成功 |完整独立episode与原门槛及作用方证明 |短诊断、MPC成功混入Student |

## 7. 计划自审和交接

本文件为依赖总索引；执行代码以子计划为准。对子计划逐项检查：代码块可解析、方法/参数存在或在该任务定义、新测试路径标明创建、旧测试路径存在、没有未定义fixture、没有用省略实现伪装生产代码。

新Teacher设计已获用户确认，但实际Reset、距离、完整接触/Hold/slip和原生PPO预算仍按证据解锁。smoke与full需要的来源分开，不能因full暂缺来源而永久阻断smoke。新模型/input/context/profile/checkpoint独立版本，保留旧371模型和账本。

下一步执行方式：推荐本聊天中按子任务派发Agent，主代理做规范与代码两轮审查后集成；也可选择主代理串行执行。无论选择哪种，GPU原生始终串行且按独立配方授权。
