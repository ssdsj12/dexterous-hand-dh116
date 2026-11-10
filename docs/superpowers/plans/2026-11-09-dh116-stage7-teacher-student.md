## 2026-11-10 发布进度 — D：承托转移八阶段请求状态机（仅CPU）

- [x] 新增未接SDK的APPROACH/CLOSE/STABILIZE/LOAD_TRANSFER/LIFT/AIRBORNE_HOLD/RECOVERY/FAILED状态机。0.2s实测双侧对向稳定及有源object-following后，才建议0.5mm/s、累计请求曝光上限2mm的转移。
- [x] 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- [ ] 对应Native来源/接入/校准及真实任务验收尚未完成。

2mm是请求账而非实际位移；Lift默认1mm/s，至3mm/s需独立证据。缺测/丢接触请求原measured hold、pressure恢复/卸载，zero24不冒充安全保持；原observer成功仅保留来源，首terminal/no-op/reset时钟隔离，不自授任务成功。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。

本节不将下方完整H8/恢复/正式验收任务标成完成。见[本次组合报告](../reports/2026-11-10-stage7-upload-cpu-components.md)。

## 2026-11-10 发布进度 — C：局部力驱动响应与批量FK数学预览（仅CPU）

- [x] 新增完整17D局部implicit capped force-PD数学响应及批量mounted FK，左右手按各自SDK行、关节轴和单位计算。NONE/零cap驱动与缺参数分开，未知量不补0，原始M不对称化、不加单位惯量或完美mimic。
- [x] 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- [ ] 对应Native来源/接入/校准及真实任务验收尚未完成。

未来17D targets/load是显式假设，驱动、接触、补偿符号及真实响应未校准；error_tube=None。批量/标量数学一致不代表原生几何、物理精度、实时deadline或动作准入。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。

本节不将下方完整H8/恢复/正式验收任务标成完成。见[本次组合报告](../reports/2026-11-10-stage7-upload-cpu-components.md)。

## 2026-11-10 发布进度 — B：H8原命令桥接与约束诊断（仅CPU）

- [x] 新增未接控制的64子步命令假设与固定13项约束报告，复用原共享kernel和真实左右SDK命名映射。每手12个直接目标不冒充17D实际响应；反馈冻结/外推必须显式声明。
- [x] 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- [ ] 对应Native来源/接入/校准及真实任务验收尚未完成。

重放原有typed命令诊断、保留当前SDK限位/速度/lead来源与混合单位；物体、接触、碰撞和完整原控制等缺口仍NA/UNCERTIFIED，feasible=False、action=None。同步更新旧生命周期测试夹具，不改变生产控制器。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。

本节不将下方完整H8/恢复/正式验收任务标成完成。见[本次组合报告](../reports/2026-11-10-stage7-upload-cpu-components.md)。

## 2026-11-10 发布进度 — A：三路被动接触观测侧车（默认关闭）（仅CPU）

- [x] 复用现有Wire→桌面/两端支架视图，新增默认关闭的只读采集接线；保留实际路径、每getter时钟/来源和独立normal/friction原值及NA。不创建World/view，不改变原控制或support字段。
- [x] 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- [ ] 对应Native来源/接入/校准及真实任务验收尚未完成。

正常量先own再读摩擦；所有行的count/start、容量、重叠仍严格检查，仅按真实Wire行选择活跃非reset载荷。保留首terminal、退休代及失败partial，不以空报告、未知摩擦或numeric ID缺失填造反力/资格。

M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。

本节不将下方完整H8/恢复/正式验收任务标成完成。见[本次组合报告](../reports/2026-11-10-stage7-upload-cpu-components.md)。

## 2026-11-10 发布进度 — 双手接触状态与原预算委托（仅CPU、未Native接入）

- [x] 7状态包含正式CONTACT_CANDIDATE；未知slip/opposition独立NA，selected reset预检并隔离兄弟，首terminal不被no-op覆盖。原20ms仅pressure/recovery稳定，不是双侧0.2s或任务成功。
- [x] 独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元27项是其中子集，不另相加。该证书对应immutable source15，不把当前工作树其它改动自动算入。
- [ ] 与实际控制/采集的接线、真实slip/opposition来源及原生恢复/双侧接触持续性验证仍待完成。

边界：本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。

本节仅记录该子单元进展，不把下方原计划的完整H8/恢复/正式验收门槛标成完成。命令与SHA见[独立报告](../reports/2026-11-10-stage7-bimanual-contact-runtime-cpu.md)。

## 2026-11-10 发布进度 — H8 完整17D运动学外推（仅CPU、未Native接入）

- [x] 完整17D含被动行，左右各用自己的mounted FK；区间速度逐步积分，缺失/无效数据不补0，不把目标当实测响应。
- [x] 独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元39项是其中子集，不另相加。该证书对应immutable source15，不把当前工作树其它改动自动算入。
- [ ] 动作响应模型、未来物体/接触/驱动、完整约束/求解器、单实例原生校准与正式配对仍待完成。

边界：本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。

本节仅记录该子单元进展，不把下方原计划的完整H8/恢复/正式验收门槛标成完成。命令与SHA见[独立报告](../reports/2026-11-10-stage7-h8-kinematic-cpu.md)。

# DH116 Stage 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Preserve the ongoing native process and use the checkboxes to record actual outcomes.

**Goal:** Complete the four current visual BC source evaluations and identify actual Teacher/Student divergence before upgrading the bimanual Teacher/MPC under the user's unchanged success and safety standards.

**Architecture:** Keep M0, the old Teacher rollback, 3844D four-frame ReLU/hemisphere Actor, 371D privileged Critic and canonical24 Tanh contract. Add SHA-bound diagnostic event references and a Stage7 divergence adapter before adding the new opt-in Teacher state, dynamics, objectives, constraints, solver, contact and load-transfer runtime. Teacher privileged fields never enter Student observations.

**Tech stack:** Existing PyTorch/IsaacLab, source FK/Jacobian and nonlinear geometry, existing constrained optimizers, JSON/Torch provenance, one Linux native lock, CPU/real-Torch-CUDA/native evidence separately labelled.

**Approved specification:** User attachment `C:/Users/xk/.codex/attachments/11bff5af-7a43-4c72-a5ba-e6420bba8c13/已粘贴的文本.txt`; base `e643ad6e9521225dd2ec0c0aa9dc97c820a2b3b9`; worktree `C:/Users/xk/Documents/project/dh116-publish-11-5`; branch `codex/dh116-lift-repair-20261106`. The user's detailed design and instruction to execute supply the design approval; no additional approval loop is needed.

## Commit 1 — Current sources and real divergence

- [x] Confirm remote `e643ad6` and the live process/lock before work. Original source driver PID3608491/SSH82267 was the sole native owner at entry; B1 was not stopped and frozen01 remained unchanged. All original source jobs subsequently finished naturally. The subsequent finite driver96407 also finished naturally; the final7d05c1b9 handoff confirms no native process or lock holder.
- [x] Let B1 and then B2 finish naturally in `stage6_sequence_20261109_v01/native_sources01`, retaining actual issued24, full480Hz state, first terminal, object lift, contact and failure evidence. All four models completed0/4 pure Student success, all Hold0 and below30mm; B2 report `8b3bd969002f608f3909ef5b9e674864e36b4d95da0edbcfa56ed6da4e3c6aa1`. Original SSH82267 exited0; the subsequent TeacherReplay02 also completed before the finite branch queue started.
- [x] Complete the second independently reset Teacher tape repetition:1175 controlsteps,actual4/4 Teacher first-terminal success,Student0; report `32a5b02e3e2f5c33e8b95f99fa528035f2fe7b2ea040c253e09391e1548aedfb`, independent UUID2dfcd10c…;296 raw/audit files checked and lock/process handoff verified. These repeat the same four sources, not new independent cases.
- [x] Add `tabletop_vision_stage7_event_reference.py` and `tests/test_stage7_event_reference.py`. TDD rejects wrong source/report/raw/UID/action/clock, missing pre-contact row and contact initially true. Actual reference02 SHA `d9eb43da8ef087b341d03d7e05627a5c3c83fe4f74a7e07baef53f55eef450af`: actual first post-contact121 and PRE120; old pre-action CONTACT122/123 separately retained.
- [x] Extend only the opt-in branch path in `tabletop_vision_branching_eval.py` and tests. Native PRE_CONTACT requires the event-reference identity and common source input match. Old source, Teacher replay, CONTACT/LIFT_ONSET/HOLD scheduling and all four controller bytes remain unchanged. Initial scoped86 CPU checks passed; independent review found no P0/P1. The later2862-file freeze/cache and actual64 execution completed, as recorded below.
- [x] Add a Stage7 finite batch driver and tests. Fixed64-job schedule, strict actual4-source gate, unique3000 completed reports and per-job one-native/reset/freeze/cache checks; final stableCPU capsule02 actual142 passed/381.36s/exit0, validation SHA `67548f963ea31399ba275687b969d10c6f53928c515c8b32ab0441f9404e349b`. The actual64 execution and posthoc completion are recorded separately below; old24 is superseded.
- [x] Implement `tabletop_vision_teacher_student_divergence.py` plus tests for exact single-Tanh saved-observation predictions,24D and left/right thumb/index/palm/rotation MAE/P95/max/direction; Teacher-zero denominators explicit. Actual all4offlinev02 complete, event-neighborhood strata remain distinct from actual event-entry states; completed source and branch six-first extraction are recorded separately below.
- [x] Recheck full Teacher shadow parity, then compare each validL16 window's independent Teacher/Student accumulated joint, palm and rotation targets against the same saved measured feedback. Actual all4 offlinev02 SHA `a114be9286fbcec1b5a3451483b19aae8383302ac81dce7805af8c45243d04c0`:4698 observations,4638 windows,593664 overlapping valid physical samples,Teacher float64 error0; native units/normalization separate, no future physical/contact prediction.
- [x] Extract all four pure-source models' six independent first-event timelines with exact primary numeric comparisons, separate repeat-range and original-qualified ACTIVE triggers, accepted/bearing/contact-repeat distinctions and valid fixed-anchor NA. Actual v03 allfour reports and original implementation snapshots retained; registrySHA `1fc5e3e0dab2b2211f5af725e9af1c4b440f86b8274b688060fd69b831f9cd4d`. No added native trials or unique initial states. Fresh independent review closed primary numeric/repeat and pre/post SDK provenance issues. The explicit unowned-tail proof reader subsequently passed the integrated310 checks and all64 actual bounded branch analyses.
- [x] Run the final independent CPU02 combination: actual310 passed/300.22s/exit0 across11 files; source2865-file snapshot and original analysis source byte hashes unchanged, native0. Validation SHA `cae569f33bed80ae5575b6bff9165fea0b810f71bac454ba6d64b9b2a38b3d41`. Preserve CPU01 exit4/0tests caused by missing summary files; only the independent CPU02 snapshot composition was corrected.
- [x] Complete all64 cold-reset branch jobs and64 posthoc comparisons:256 bounded windows,41472 Student physics substeps,635392 Teacher prefix substeps,missing0;final summary656b9236...7923, native handoff7d05c1b9...dc94, final qualified projection13270507cc48769e8d2fd43061e612b1e162a179baa6af15e947985ef1f5cdbd. No Student success/generalization certificate. Diagnostic3mm-anchor remains separate from native30mm/5s anchor; all original data and controller bytes retained.
- [x] Review the actual sources and full64 outcomes before objective selection: no qualified winner; B0 control, B1 event-diagnostic baseline and B2 command-loss ablation retained. No B3/PPO/default Teacher promotion. Source-only and bounded mixed-control outcomes remain distinct.
- [x] Complete CPU RED/GREEN, actual310 integrated checks,64 native branches and independent source/data/document reviews; README/CHANGELOG/main report/512-row aggregate appendix record actual commands,SHA,failed qualification gates and raw pointer scopes. Review findings on historical progress and peak-versus-event provenance wording closed; runtime source remains byte-identical to the310 snapshot.
- [x] Published the first Stage7 evidence unit as59c52bf91bb88153a37af61aac15495aa9e89df7; remote ls-remote matches, PR1 open/Draft/unmerged and title/body/head rechecked by API. Begin Commit2 with these completed outcomes; no B3/PPO/Teacher promotion.

## Commit 2 — Unified Teacher state and actual contacts (native verification in progress)

- [x] Add immutable `tabletop_bimanual_mpc_state.py` with source/unit/frame/timestamp/valid/missing reason and run/env/episode/World/SDK/policy identities. Map supported actual measurements; explicitly declare missing qualified support, partner IDs, passive reactions and unbound centerline rather than claiming complete measurement capability. Archive terminal/reset snapshots but reject them for planning.
- [x] Reuse original `TensorWorld.refresh/measure`, `TripodNativeEvidenceBuilder`, actual `read_common_pause_targets` and body COM readback. Preserve force-side/origin semantics, raw weighted centroids, scalar/vector distinction and whole-hand versus accepted-pad attribution. A neutral measurement label does not qualify A–E or a new expert.
- [x] Add default-OFF original-Teacher-only capture, original-refresh read-only receipts, per-env first-terminal closure, actual/cohort masks, exact issued24/eight-substep clocks and independent cleanup. Keep invalid raw nonfinite evidence reconstructable without making DTO measurements valid.
- [x] Complete independent freeze03 and path-bound freeze04 CPU validation:185 passed,1 CUDA-unavailable skip,58 subtests; capture22 stdlib tests, measurement33, state25 and targeted independent review/probes. Original four controllers unchanged. Runtime freeze04 changes only four asset-manifest path strings with matching actual USD SHA; bootstrap failures remain preserved.
- [x] Finish the single-instance freeze04 180-step native capture: actual11520 states/1440 frames, exit0 and closed gzip/raw/state/action/clock integrity passed. Terminal0/all8 right-censored; qualified support and passive reactions remain NA, so this is limited native measurement-chain evidence.
- [x] Complete the independent original1900-budget first-episode collection: actual1800 controlsteps,14400 rawframes,84461 first-cohort states,8 genuine terminals; oldTeacher pinch4/4,tripod0/4(two drops,two timeouts). Closed raw/DTO/clock/action/UID/reset/no-op analysis passes;30739 closed slots excluded. These repeat nominal clones, not100 independent initials or newTeacher/Student success. New solver execution and full measurement qualification remain unearned.
- [x] Publish the limited state/measurement evidence unit as804214e26fe94db8ef2285dffa4820578a361c88; remote SHA and PR1 open/Draft/unmerged/head/title/body verified. No formal success or Teacher promotion. The EOF-only test hygiene exception has independent33-pass evidence.
- [x] Extend opt-in measurements with actual composed Cylinder/COM/inertia and three low-level static views, strict installed contact-report provenance and readonly aux480Hz archive. Source05 CPU331 passed but real Shadow callback validation rejected collection; preserve that failure. Correct exact Sequence->aux->core source chain without changing original controllers/Sequence; independent source06 CPU336 passed,1CUDA-unavailable skip,93subtests and genuine16-control smoke exit0/128frames/1024states. Closed aux+core integrity passes;all8episodes right-censored. Static view metadata/owned data delivered,global full report empty;qualifiedsupport/sign/frame/completeness/partner IDs remain NA. No60Hz wall-time or task qualification.
- [x] Publish the auxiliary evidence unit as4012f12bf69ead3c03e38f9abf495692f7fa065e with updated README/CHANGELOG/report; remote SHA and PR1 open/Draft/unmerged/title/body verified.15 files only,all staged runtime/test bytes match native06 overlay;future model/objective modules and raw artifacts excluded.
- [x] Add default-off full-articulation model capture and independent post-step/full-report observer arms. Preserve source07 CPU fixture incompatibility and source08 actual enum-binding failure; source09 protected enum equality passes actual1/16-control Model capture and closed raw/DTO/source/clock/action analysis. Model16 has128frames/1024states,44fields each1024valid and6deliberateNA;each hand actual17DOF/23links/22joints. Both contact observers still deliver no global full-report data. Public observer-only independent CPU526passed/1CUDA skip/147subtests;no new force/dynamics/task/60Hz qualification.
- [x] Published the focused observer/articulation unit as c443c7dd3cf86f4a2a68ab4a7970a2dcf85f19ce with README/CHANGELOG/native report and exact frozen09 runtime/test staging; remote SHA and PR1 open/Draft/unmerged/title/body verified. Future F0 metadata and objective arithmetic remain independent units. No defaultTeacher/B3/PPO promotion.
- [x] Audit only the current full17D getter arithmetic behind the exact passing Model proofs:8/128 physical frames and64/1024 active samples per hand, finite symmetric raw matrices and positive computed spectra. Tool/provider/proof/input SHA and source binding checked before raw access, with19 independent CPU tests. This supplies no future-state evaluator, isolated drive torque, constraint elimination or physical dynamics certificate.
- [x] Finish limited fresh filtered-contact metadata F0 closure and its independent16-control test before numeric contact F1/sign/frame/ownership work. Source10 independent CPU630passed/1CUDA skip/223subtests; public scope582passed/1skip/223subtests independently executed. Actual1/16-control captures exit0 and closed source/SHA/CRC/CoreAux/action/clock/F0raw-replay audits pass:8/128frames,2176/34816actualpairrows,0terminal/all8right-censored per arm. Published08bb290 with exact frozen runtime/tests,remoteSHA and PRDraft checks. Strict v1 previous-terminal refusal is preserved; the subsequent separate v2/F1 unit is recorded below.
- [x] Implement independent default-off v2 first-cohort retirement/explicit generation receipts and F1 sensor-side values. Source11 fullCPU687passed/1skip/238subtests; actual independent public scope639passed/1skip/238subtests21.95s. Actualepoch1/values1/values16 exit0; closed source/SHA/CRC/CoreAux/action/clock/F0/epoch/F1 replay audits pass,8/8/128frames. Numeric normal/friction2176/34816valid and0NA; each has0nonzero observations, not accepted contact/support evidence. All8episodes per run censored/0terminal, so mixed realterminal/generation remains Native-unverified. Values16 proof985a532db6389bd0cd2b05b5638b1f462e84bf4df052112d9cb59f9945ca5cc1. No force/task qualification. Separate180-control contact-establishment observation explicitly started only after16closed proof; no result yet, no H8 or Student training.

- [x] Add default-off mounted descriptor and independent full17D FK/J hypothesis; source12 independentCPU769passed/1skip/238subtests,251-case integration review and19FK tests. No Native job for this descriptor yet; the prepared1-control plan must wait for the existing sole Isaac capture. No current/FK/J/dynamics/control qualification is issued by CPU checks. See 2026-11-10-stage7-mounted-model-cpu.md.

## Commit 3 — New constrained horizon8 Teacher MPC

- [x] Preserve the first descriptor Native failure:8 frames/64 samples all structure NA; closed integrity passed. Locate the installed USD PropertyStack missing-TimeCode ABI and minimally repair the two property-stack calls. Source16 CPU1364passed/3skipped/238subtests; original source12/15 and failed archives unchanged. The independently prepared descriptor1_v02 has executed0 Native jobs; FK/J remains uncalibrated. See 2026-11-10-stage7-descriptor-usd-abi.md.

- [x] Implement the independent eight-term objective arithmetic only, with explicit weights/source/units/masks/NA, current-constant versus candidate-dependent scope and no control wiring. Exact source14b2976664817ba2b6171da596d102105dfc9704726b1823601a02ee4972b2ab; independentCPU state+objective73passed41subtests0.38s,Native0. This does not complete the following H8/dynamics/solver/constraint tasks or confer action admission.
- [ ] Add `tabletop_bimanual_mpc_dynamics.py`, `..._objective.py`, `..._constraints.py`, `..._solver.py`, `..._runtime.py`; test each as a separate responsibility. Reuse the actual24 action contract and shared control computation, each hand's measured full-joint FK/Jacobian, nonlinear contact-region and collision checks.
- [ ] Horizon8 at60Hz predicts0.1333s with8 original480Hz substeps per control. Replan from fresh actual state and submit only the first validated action. Rigid cylinder prediction is explicitly rigid; real Cosserat/DeformX advance mutates true state and cannot be used as a fake reversible candidate oracle.
- [ ] Record each raw/weighted geometry/contact/support/slip/bimanual/joint/smooth/risk cost, solver status/iterations/initial/final objective, all active constraints and infeasible reason. Use explicit experimental weights rather than claiming optimality.
- [ ] Measure solve and complete validation time with monotonic wallclock. A deadline/infeasibility cannot relax original joint, velocity, lead, torque, collision, penetration or clearance limits; reject unverified actions and use the original qualified safe hold/unload path. There is currently no16.67ms qualification evidence.
- [ ] Before native use, test horizon shifting/warm reset, FK predictions, independent masks, timeout/infeasible/uncertified refusal, correct output inverse/SDK mapping and original path parity. Native smoke and deadline outcomes must remain separate from success evidence.

## Commit 4 — Bilateral per-digit contact recovery

- [ ] Add `tabletop_bimanual_contact_runtime.py`, reusing the existing N×2×3 recovery ledger and per-hand geometry. States are NO_CONTACT/CANDIDATE/VALID/STABLE/SLIPPING/LOST/REACQUIRING; distance≤1mm/cos≥0.8 are geometry candidates only.
- [ ] Confirm actual object contact body, load, normal, impulse source and point while excluding table/support/other fingers. Each physicalepisode/hand/digit keeps the original cumulative1mm/1s budget and0.5s no-progress stop, without contact-flicker reset. Keep bearing digits under original pressure control and use the existing missing/bearing10×/regularizer.01 geometry objective.
- [ ] A recovery that loses the opposite hand fails the coordination experiment. Run fixed-wrist A/B first in independent freezes; moving recovery remains blocked without actual cooked/live geometry qualification.

## Commit 5 — Controlled support transfer and lift

- [ ] Add `tabletop_bimanual_load_transfer.py` for APPROACH/CLOSE/STABILIZE/LOAD_TRANSFER/LIFT/AIRBORNE_HOLD/RECOVERY/FAILED. Use actual per-hand contact and physical progression, not training iteration or predicted geometry as contact.
- [ ] Require0.2s stable bilateral opposition before0.5mm/s initial transfer, capped2mm. Monitor actual object following, support, reaction, slip, tilt and tracking; pause/reoptimize when the object does not follow. Start Lift at1mm/s, progress to3mm/s only with evidence, never exceed original10mm/s.
- [ ] Preserve30mm net object lift,5s actual airborne qualified hold and≤0.5mm fixed-anchor slip. Reject external force/attachments/table or support help. Validate static then short transfer then full original task, logging all failures.

## Commit 6 — Full-episode common pause and Teacher priority

- [ ] Connect an explicit Teacher mandatory-unload interface; do not infer unload from negative actions or apply the current surface-only no-unload wrapper blindly to Teacher/MPC. Keep measured backend and cache separate, cancel queued wrist targets in the actual chart without teleportation, and preserve original constraints.
- [ ] Run original/unilateral/common complete first episodes with fresh snapshots. Record old-limit behavior, new force/left failures/deadlock and original pinch Teacher success. The earlier0.533s result and CPU/CUDA fixtures cannot replace this native evidence.

## Commit 7 — Paired old/new Teacher validation

- [ ] Freeze at least100 unique matched initial states per style. Original and new Teacher use the same object/grasp/physics/material/friction/camera/timestep/actuator/success settings; identities and first-terminal evidence must be unique and complete.
- [ ] Report native success,30mm rate,5s hold, fixed-anchor slip, bilateral contact duration, limits/overload/drop/time/recovery/reactions and shape. Rigid proxy shape is not flexible deformation evidence.
- [ ] Require no degradation for either style and no safety increase with sufficient paired statistical evidence; report fallback usage separately. Keep original Teacher as default/rollback until the qualification artifact actually passes; no promotion by code or loss.

## Commits 8–9 — Qualified new expert pool and Student transfer

- [ ] Only after new Teacher qualification, collect separate pinch/tripod RGB-D, actual3844D/issued24/previous action and physical identity/proof pools; failures separate. Preserve existing4698 pool and all original checkpoints.
- [ ] BC→event sampling→conditional command loss→pure Student source closure→native-qualified correction states→DAgger→short PPO. Retain PointNet probes every250 and no silent reset; B3 only after actual B2 native improvement, RECOVERY only after qualified recovery data.
- [ ] At present B3/PPO/default Teacher replacement are prohibited. Later conditional stages require actual preceding results, not fixture or implementation completion. Update README/CHANGELOG/report per commit, push verified SHA, and keep PRDraft until qualification is sufficient.
