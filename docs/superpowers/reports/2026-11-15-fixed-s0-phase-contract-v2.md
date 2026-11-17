# Fixed S0 Phase Contract v2：W3 Task 1–2 CPU 报告

日期：2026-11-15。实施基线：`3f277ed219248280b1640b11a4bab78c4080d981`。
独立工作树：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/phase-v2-cpu-20261115`；
分支：`codex/phase-v2-cpu-20261115`。本报告待 root 对精确提交进行 SPEC、QUALITY 审查及 Windows 发布。

## 范围与授权

用户已批准独立 v2 设计及文档顺序，原 DOCX SHA256：
`1ea4da5342de9a64e5adb3ef5c0a50482d2158bd9a9d3e7f5fab7f6b04e318dd`。
本包仅实现 phase-v2 plan 的 Task 1–2 四阶段数学：
`Close → PreliftHold → Lift → LiftHold`。交付仅为独立模块、对应 CPU 测试及本报告。

没有实现 context51、模型、相机、训练、native producer/bridge、W4 source adapter 或 W5 长 Reset。
没有修改旧 context50、原 phase/evaluator/slip、生产工厂、checkpoint、旧 R0 或保护基线。
CPU fixture 不能获得真实物理资格；`native_steps=0`、`actual_physics_steps=0`，没有 App、GPU 或 Isaac 执行，也没有 Ubuntu push。
根 README、Registry、readiness 等共享发布文档由 root 集成。

## 固定 S0 与来源门

只读核对了原 `candidate_config`、`load_reset_manifest`、`ResolvedResetManifest.data/verify`，
旧 context phase receipt，以及原 task 的 slip/phase 更新实现。
实际 `candidate_config(style="pinch")["resolved_reset_manifest"] is None`，
原 loader 抛 `resolved_reset_manifest_source_missing`，记录为 `fixed_s0_manifest_unresolved`。
未造 manifest、未用照片或 pose preview 推断 S0；实际 factory/full profile 继续拒绝准入。

`PhaseSampleV2` 的三个 bool 是调用方的 CPU fixture 输入。本模块不提供 native producer 接口。
全部 `PhaseReceiptV2` 保持计划中的六个字段，并有不可通过构造参数或 setter 改写的只读
`source_scope="cpu_math_fixture"`、`native_ready=False` 属性。
`success_candidate=True` 只说明整数 tick 数学门槛成立，既不自动物理 terminate，也不代表原生成功或准入。
裸 receipt、调用方 True 或 `state.source_scope` 均不能证明物理来源。
后续 W4 必须见证实际同钟来源；后续 context/source adapter 还须核验实际 phase owner，不能只信调用方 receipt 或 UID 字符串。

## 输入与生命周期合同

- UID 必须为非空原生 str，tick 必须为非负原生 int，拒绝 bool tick。
- 每次 observe 必须为 exact `PhaseSampleV2`、同当前 episode、连续 tick+1；dict、裸 receipt、非 bool、
  None/NaN 等未知资格均拒收，不进行 bool cast。
- 非法 sample 在任何状态提交前拒绝，receipt 与两个内部计时起点保持不变。
- 每个 owner 独立管理一个环境；同一 owner lifetime 禁止重用成功 Reset 过的 UID。
  全局 UID 分配由外层 run/environment/reset/UUID owner 承担，本模块没有进程级永久 UID 注册表。
- reset 原子地清两个起点，返回 Close、两个零时长、candidate false。
  `current_receipt` 只读返回当前 immutable receipt；未 Reset 时拒绝读取/observe。
- sample/receipt 为 frozen dataclass；旧 receipt 在推进后保留其端点值。
  本模块不持有、创建、替换或清除材料锚。

## 两个时钟的边界

频率为 480 Hz，所有时长只从连续整数 tick 间隔计算：

| CPU fixture 端点 | 阶段 | 预抬时长 | 最终保持时长 | candidate |
| --- | --- | --- | --- | --- |
| Reset tick 2 | Close | 0 | 0 | false |
| tick 3，首次 bilateral+stable | PreliftHold | 0 | 0 | false |
| tick 122，119 个预抬间隔 | PreliftHold | 119/480 | 0 | false |
| tick 123，120 个预抬间隔 | Lift | 0.25 | 0 | false |
| tick 124，首次 final eligible | LiftHold | 0.25 | 0 | false |
| tick 2523，2399 个最终间隔 | LiftHold | 0.25 | 2399/480 | false |
| tick 2524，2400 个最终间隔、2401 端点 | LiftHold | 0.25 | 5 | true |

0.25 s 仅是测试传入的新候选参数，没有变更原成功标准。
PreliftHold 中 final=True 不可累计最终保持，也不可越级；切到 Lift 的端点仍不启动最终计时。
预抬稳定性中断重新建立预抬连续窗口；最终资格中断退回 Lift、清最终时钟。
任意阶段丢失 bilateral 退回 Close 并清两个时钟。candidate 后可继续保持，直到中断。
原严格成功（整线净升 30 mm、真实双侧支撑/离桌、连续悬空 5 s、无外援、
材料与切向滑移不超过 0.5 mm、数值健康）的物理验证仍由后续真实来源及独立 evaluator 完成。

## TDD 与验证证据

私有证据目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/phase-v2-cpu-20261115`。

| 日志 | 实际结果 |
| --- | --- |
| `task1-red.log` | 1 failed，预期新模块缺失 ModuleNotFoundError |
| `task1-validation-red.log` | 55 failed，新模块尚不存在 |
| `task1-green.log` | 56 passed，类型/Reset/输入/状态隔离合同 |
| `task2-red.log` | 11 failed, 56 passed，缺少四阶段转移语义 |
| `task2-green.log` | 67 passed，四阶段与双时钟 |
| `phase-context-regression.log` | 126 passed，67 新测试 + 59 旧 context 测试 |
| `cpu-trace.json` | 独立 trace：2522 次 CPU observe、native_steps=0、actual_physics_steps=0 |
| `s0-protection-pre.log` | S0 unresolved；verify_protection 528 源码 + 2 历史 checkpoint 通过 |
| `protection-final.log` | 完成前再次核验 528 源码 + 2 历史 checkpoint |

CPU 测试使用原 canonical wrapper：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_phase_contract_v2.py tests/test_native_teacher_context.py
~~~

覆盖 Reset tick 2→3、多 episode Reset（包括 candidate 后）、119/120 与 2399/2400 边界、
最终/预抬资格中断、各阶段 bilateral 丢失、配置隔离、错误 UID/tick、bool tick、
未知/非 bool/NaN、非法来源参数、immutable receipt、非法 sample 拒收原子性。
旧 `CONTEXT_FIELDS` 的四 label 仍为 Approach/Close/Lift/Hold，旧 context 回归保持通过。
`Repository(root, evidence_dir=...)` 使用原构造方式调用 `verify_protection()`，
没有重写或跳过保护核验。`git diff --check` 与精确三文件暂存检查均在提交前执行。

## 交接边界

本包只证明 CPU 数学。未证明 native_ready、actual fixed S0、实际稳定来源、训练链、
长 Reset、物理抓取成功或性能提升。后续 W3 context/model、W4 同钟 source adapter、
W5 clock/Reset 及正式集成仍须各自实施与验收；本包不解锁这些准入门。
