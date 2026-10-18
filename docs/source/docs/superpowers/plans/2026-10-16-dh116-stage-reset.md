# DH116 Stage Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use executing-plans; user selected Inline Execution. Track each checkbox and retain physical evidence.

**Goal:** 从最近合格阶段初始化，重新验证接触，再执行未完成阶段，最终通过连续抓取验收。

**Architecture:** 阶段证据门与工件校验独立于Isaac；运行时负责注入机器人/杆状态、重建接触和时间。阶段测试不授予端到端资格。

**Tech Stack:** Python/numpy/pytest、Isaac Lab、PyElastica-Mesh。

## Global Constraints

- 设计：../specs/2026-10-16-dh116-stage-reset-design.md，用户已确认。
- 现有worktree codex/t500-dual-panda-o6-mpc，保留dirty修改；Inline，不创建新worktree、不自动提交。
- 所有测试taskset -c 0，数值库单线程；不热改PID15485运行的模块，不启动第二个物理实例。
- CPU一核；原碰撞/压入/速度门不变，无wire绑定，保存来源SHA。
- 只做上方对捏；下降阶段不伪造已有实现。

## Task 1: 恢复后的独立复验门

Files: create `Go2Pvcnn/go2_pvcnn/control/dh116/restore_gate.py`; tests `Go2Pvcnn/tests/test_dh116_restore_gate.py`.

API: `RestoreGate(entry_stage, *, minimum_loads_n, saved_anchors_local)`；`update(time_s, *, safety_ok, active_constraints, settled, approach_ok, loads_n, valid_contacts, normals, anchors_local)` returns `ready, aborted, reason, entry_stage`。不输出连续抓取成功。

- [x] RED：测试未达.1s不得释放approach/close；hold/lift必须新计时1s，滑移相对保存锚点≤2mm；绑定/安全失败锁存abort；非法时间与NaN拒绝。18项因模块缺失失败。
- [x] GREEN：实现小型纯状态门，仅从时间零附近首个采样开始接受≤.01s连续测量；接触/预载不满足清零持续计时；保护失败不自动恢复。
- [x] 验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_dh116_restore_gate.py tests/test_dh116_grasp_evidence.py tests/test_dh116_grasp_sequence.py -q`。52 passed，未接真实恢复运行时。

## Task 2: 状态工件与来源资格

Files: create `Go2Pvcnn/go2_pvcnn/control/dh116/stage_state.py`, `Go2Pvcnn/tests/test_dh116_stage_state.py`。

API: `validate_state(payload, expected_pins)` returns validated copy; `save_state(path,payload)` exclusive atomic publish; `load_state(path,sha256,expected_pins)` fail closed。

- [x] RED：源前序阶段未PASS、错误SHA、缺关节/杆静止结构、NaN/非正交director/越界、目标阶段不合法均拒绝，旧文件不能覆盖。
- [x] GREEN：显式schema保存q/qd、根状态、控制目标、杆动态与静止数组、固定材料ID和锚点、落定高度、驱动/材料/资产pins、源run/step/合格阶段。SHA用序列化实际字节，发布前完整校验。
- [x] 校验有效roundtrip及所有拒绝用例；生产者仅在实测状态机相邻阶段转移、所有前序资格与已有点Jacobian探针通过时发布。

## Task 3: 后端物理状态注入

Files: `/home/xk/coding/DeformX/PyElastica-Mesh/co_sim/free_rod.py`、`moving_contact.py`及对应tests。

- [x] RED：变形杆恢复后静止长度/应变不变，速度和director一致；陈旧冲量/预测缓存不能保留，配置不符拒绝。
- [x] GREEN：FreeRodEngine新增export_physical_state/restore_physical_state，仅fresh实例接受一次；MovingContactRodEngine继承此API，保持其新建空缓存，不额外修改moving_contact.py。
- [x] core0真实后端单步对照重力与非零接触响应通过；不是PhysX精确断点恢复。实际机器人恢复瞬态仍属于Task5。

## Task 4: Isaac运行时与阶段时钟

Files: `Go2Pvcnn/scripts/verify_fixed_dual_panda_dh116.py`、`go2_pvcnn/control/dh116/grasp_sequence.py`及新增`tests/test_dh116_stage_reset_runtime.py`。

- [x] RED：恢复参数必须有外部SHA；非法状态阶段拒绝；恢复前不授权lift；summary区分imported_evidence和本次执行结果。
- [x] GREEN：加载Task2状态，机器人按关节名检查实际限位/固定根并写q/qd刷新，Task3杆初始化，Task1复验后才进入局部阶段时钟；控制目标和材料基线保留，不重复已施加冲量。代码接线，未宣称实际Isaac恢复验收。
- [x] --save-stage-states在阶段通过时独占发布工件；--restore-state/--restore-state-sha256在新实例恢复，失败不自动重启。默认连续流程保留。
- [x] DH116及相关后端合计489回归通过，Inline自查；无“加载姿态即连续成功”分支。实际Isaac验证仍在Task5，不将静态接线测试当物理验收。

## Task 5: 实际阶段恢复和端到端验收

- [ ] 当前trial2结束后才启动下一个物理实例。从其合格状态依次做落定→接近、接近→闭指、夹持→抬升恢复，检查原保护/载荷/滑移和短时冲击。
- [ ] 从未达标阶段继续，状态工件无资格时明确退至最近合格入口。
- [ ] 完整连续运行通过：四点夹持≥1s、滑移≤2mm、两材料点升≥30mm保持≥1s；整根离桌≥2mm独立统计；同配置重复3次。
- [ ] 更新notes与CODEX_RESUME，仍未通过的项不得标完成；下降放回属于另一个执行扩展，必须保留未完成标记。
