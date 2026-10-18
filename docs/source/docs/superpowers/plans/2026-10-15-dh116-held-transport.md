# DH116 Held Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: executing-plans. 已选 Inline Execution，不派代理，不重新询问执行方式。

**Goal:** 在已批准的闭指→保持→抬升设计内，补齐双指共同平移的速度分配核心；当前物理 pilot 不热更。

**Architecture:** 复用实际两点 Jacobian + 掌心角 Jacobian，在原法向力反馈速度上叠加共同世界平移速度。默认零平移与旧调用逐元素一致；是否允许抬升由独立实测证据门决定，核心不授予抓握资格。

**Tech Stack:** NumPy、pytest、当前实际 USD FK/点 Jacobian。

## Global Constraints

- 继承批准设计：无 wire 绑定、无位姿伪抬升；臂 .05 rad/s、手 .02 rad/s 命令限速，dt ≤100µs，所有碰撞/接触/滑移门不变。
- 仅仿真，非 DH116 硬件驱动力标定；工作树已有修改，不提交/推送。
- 同一批双手需要共同缩放速度，不能独立饱和导致差动拉伸；本任务只输出单手候选，实际双手执行前必须在调用端共同缩放。

## Task 1: Translation-capable candidate allocator

Files: modify `Go2Pvcnn/go2_pvcnn/control/dh116/coordinated_closure.py`; tests `Go2Pvcnn/tests/test_dh116_coordinated_closure.py`.

- [ ] RED：新增 `transport_velocity_w=None` 默认等价、非零切向平移、旋转协变/统一缩放、NaN/形状/>.01m/s 拒绝测试；单核 pytest 观察缺少参数失败。
- [ ] GREEN：将目标构造改为下面代码，保留原奇异性、限速、法向反馈及全部输入验证。

```python
transport = np.zeros(3) if transport_velocity_w is None else np.asarray(transport_velocity_w, dtype=float)
if transport.shape != (3,) or not np.isfinite(transport).all() or np.linalg.norm(transport) > .01:
    raise ValueError('invalid coordinated transport velocity')
b = np.r_[(n * speed[:, None] + transport).reshape(6), np.zeros(3)]
```

- [ ] 实际新同截面左右 USD 验证：在力误差为零时，输出两点同速、掌心零角速、共同缩放后方向不变、所有关节限速通过。只属运动学，不属实际抬升。
- [ ] 运行 DH116 / moving-contact 回归；记录实际结果及未接运行时的边界。

## 后续依赖（不由本核心自动晋级）

新的同截面 pilot 实测稳定后，在同一连续运行中接近、实测接近门、闭指、1s 保持，再由证据门授权抬升；左右目标速度共同缩放，任一接触/保护失败即终止。不得将独立切触 pilot 的初态当成已完成接近。

## Task 2: Common bimanual scaling

Same module/test files. Interface `coordinated_bimanual_increment(J:(2,2,3,10),n:(2,2,3),W:(2,3,10),loads:(2,2),dt,*,target_loads_n:(2,2),transport_velocity_w:(3,)) -> (2,10)`; 左右顺序显式固定。

- [ ] RED：构造右侧 Jacobian 比左侧小10倍，要求两手输出点速度相等、均满足各关节速度上限；坏维度、单侧奇异/非有限证据拒绝。
- [ ] GREEN：单手已有验证和最小范数计算抽为私有未限速候选；单手入口保持原缩放，双手入口堆叠两个候选，仅一次全局缩放：

```python
scale=max(1.,np.max(np.abs(velocity[:,:7]))/.05,np.max(np.abs(velocity[:,7:]))/.02)
return velocity*(dt/scale)
```

- [ ] 两个实际新 USD 姿态一起验证正向同速抬升、掌心零角速、统一限速；运行专项及合并回归，记录仍未下发运行。

## Task 3: Explicit preload before hold

Files: `Go2Pvcnn/go2_pvcnn/control/dh116/grasp_evidence.py`, `Go2Pvcnn/tests/test_dh116_grasp_evidence.py`.

当前空载5mN只证明接触，不证明已到支持线重的预载。加入构造参数 `minimum_hold_loads_n=None`：默认保持原5mN合同；连续抬升试验显式采用所选目标的90%。这只增加 close→hold 的准入条件，不降低任何原门。

- [ ] RED：四路5mN以上但低于显式预载连续2s仍在close；达到阈值后完整.1s才hold，再完整1s才permit_lift。坏尺寸、NaN、<.005N、>3N拒绝；默认旧门行为不变。
- [ ] GREEN：验证并保存shape(4,)阈值，只在close分支加入条件：

```python
if not contact or np.any(load < self.minimum_hold_loads):
    self.contact_since=None
```

- [ ] 运行观察器/材料/协调核心及DH116全回归，记录边界：预载门尚不证明物理保持/抬升。

## Task 4: Measured continuous-sequence coordinator

Create `Go2Pvcnn/go2_pvcnn/control/dh116/grasp_sequence.py` and `Go2Pvcnn/tests/test_dh116_grasp_sequence.py`.

Interface `TabletopGraspSequence(material_ids, target_loads_n)`; `update(time_s, *, settled, approach_position_errors_m, approach_normal_errors_rad, station_heights_m, loads_n, valid_contacts, normals, anchors_local, min_surface_clearance_m, safety_ok, active_constraints)` consumes measured values only. `transport_velocity(time_s)` returns common world velocity; it does not move robots or wire.

- [ ] RED：0–.5s settle，.5–4s approach；真实误差>.002m/5°不允许close；落定不通过/绑定/安全失败abort；完整预载→保持→实际双站高度和1s dwell后才finish；保持前transport=0，lift缓启动到.005m/s，两站≥35mm时停止继续上行；坏输入/时间跳变拒绝。
- [ ] GREEN：在.5s记录真实落定站点高度；4s验证实际误差后才构造 `GraspEvidenceGate(material_ids=...,approach_qualified=True,minimum_hold_loads_n=.9*targets)`。所有后续证据原样传入，station_rise=actual_z-settled_z。只有stage lift时返回 `[0,0,.005*clip((time-lift_start)/.2,0,1)]`，两个站点都达到.035m后返回0；abort/finish均返回0。
- [ ] 核心与既有观察器回归。此纯组件不代表连续运行接线：在pilot7实际闭指结果支持晋级后，再将实际PhysX/DeformX测量接入、保留每步源/局部/反力保护、双臂共同限速，并进行独立连续GPU验收。

Task4证据更新：纯组件与时间连续性/过期命令拒绝已实现，合并388 passed in49.18s，`/tmp/dh116-grasp-sequence-full.log`。连续物理接线仍未完成。

## Task 5: Bounded-memory evidence before long continuous runs

目前所有记录保存在列表，长时100µs试验可能占据大量内存。新增独占创建的JSONL journal，每步严格序列化并flush，仅保留最后一条；总结按需流式重读，异常退出close但不重写/截断。flush不是断电持久化保证。不改变任何控制或通过门。

- [ ] RED：即时可读、NumPy转换、原对象修改不改证据、1000条仅保留最后、非法JSON不污染已有记录、拒绝覆盖、异常关闭、空记录/关闭后追加拒绝；运行脚本使用journal且finally不截断。
- [ ] GREEN：实现最小append/len/last/iter/close上下文接口，接入当前验证脚本；运行中pilot7仍用启动前快照，不热更。
- [ ] 专项与DH116/moving-contact全量回归；更新证据和连续接线未完成边界。

Task5完成证据：6专项、完整394 passed in50.21s（`/tmp/dh116-journal-full-green.log`）；原finally写文件AST合同改为loop实时写/退出不截断。新记录方式未进入已启动pilot7；连续物理状态机仍待接线。

## Task 6: Continuous measured runtime wiring (next implementation)

Scope: `verify_fixed_dual_panda_dh116.py` + runtime contract tests; pure sequence and allocator already tested. Do not hot-patch pilot8 or run a second Isaac instance.

- [ ] RED: explicit `--grasp-sequence` mutually exclusive with independent closure pilot; requires pinned prepose/approach/references/contacts, reactions, coordinated allocation, dt≤100µs and explicit force targets. Legacy pilot keeps initial .308m and no grasp qualification; sequence uses free .318m wire and measured settling.
- [ ] GREEN: separate `grasp_mode` (data capture/controller availability) from `closure_active` (pilot or sequence close/hold/lift). Before closure all pad bodies retain non-contact source protection. Keep effort headroom/actual speed/force/region guards for all grasp modes.
- [ ] RED/GREEN: post-physics actual reference points/normals vs opened-hand approach FK; actual base_link material anchors; fixed station heights and rod surface height using current rod radii. No command-derived evidence, no nearest-point reassignment. Initial material IDs derive from fixed goal regions, not drifting contacts.
- [ ] RED/GREEN: sequence update every step starting t=dt; .5s actual settled baseline, 4s actual approach gate, initialize closure command from opened-hand endpoint. Record phase/evidence in each flushed row. Fail closed on invalid/missing measurements or forbidden constraints.
- [ ] RED/GREEN: two-hand candidates collected before target publication; call `coordinated_bimanual_increment` once with sequence-authorized transport. Maintain force control while holding and lifting; any gate/guard failure aborts, never silently retracts/reopens or binds wire.
- [ ] RED/GREEN: termination requires actual sequence finish and all physical guards, not mere step count; preserve measured approach gate from transition rather than comparing lifted hands against old approach endpoint. Independent pilot summary always denies grasp success.
- [ ] Full DH116/moving-contact regression, then only after independent pilot supports progression run the new continuous prepose→approach→close→hold→lift. Final goal still requires actual evidence and repeatability, not this code checklist.

Prepared candidate88 approach: `postpilot8_approach_source_v1.json`, 116 source joint-box checks passed. Only fixed prepose→opened hand endpoint, duplicate prepose at legacy leg0; no home motion claim. This source certificate does not qualify dynamic approach or lift.

Task6代码接线已实现，完整420 passed in50.40s（`/tmp/dh116-sequence-runtime-full.log`），真实非法CLI启动前拒绝。已按实测阶段隔离接触许可、实际材料高度/半径、双臂共同分配、finish与最终probe门；仍未实际运行新连续模式，物理验收项保持OPEN。见notes/log/2026-10-15-dh116-continuous-runtime-wiring.md。
