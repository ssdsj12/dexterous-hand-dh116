# DH116 symmetric grasp and transport implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. User selected Inline Execution; no subagents. Steps use checkbox tracking.

**Goal:** Physically verify mirror grasp, sustained bilateral hold, synchronized 30 mm lift, descent and supported release.

**Architecture:** Separate functional-frame geometry, measured transport coordination and joint allocation. Generate new pinned configurations before enabling the new runtime; keep the running GUI unchanged.

**Tech Stack:** NumPy, SciPy, USD kinematics, Isaac Lab, DeformX, pytest.

## Global constraints

- Work in the user-selected independent repository and feature branch; preserve pending startup/GUI fixes.
- One Isaac process, CPU affinity 0 and numerical threads 1. Never attach wire to hands.
- Keep existing collision, load, slip, reaction and Jacobian checks. Simulation tuning is not hardware calibration.
- Spec: `docs/superpowers/specs/2026-10-18-dh116-symmetric-grasp-transport-design.md`.
- Baseline: 595 tests passed before these changes. Old pins cannot be rewritten for new configurations.
- User clarification: mirror targets apply only to this grasp; height synchronization is a lift-stage policy, not a global allocator/model constraint. Post-lift task handoff must accept independent bilateral commands. Synchronized return/release is an explicitly selected demo branch, not mandatory after every lift.

## Task 1 — Functional-frame mirror and bilateral velocity allocation

组件已实现并验证：针对性 40 项通过；Task 2 后完整回归 629 项通过。尚未接入新物理配置。

Files: create `src/go2_pvcnn/control/dh116/symmetric_geometry.py`, `tests/test_dh116_symmetric_geometry.py`; modify `src/go2_pvcnn/control/dh116/coordinated_closure.py` and `tests/test_dh116_coordinated_closure.py`.

Interfaces: `mirror_functional_pose(position, rotation)` accepts (3,), (3,3) world functional frame, returns mirrored position and a proper rotation. X and Z functional axes are mirrored; Y is reconstructed to keep right handedness. Do not pass raw USD link rotations without functional-frame calibration. Allocator accepts common (3,) or per-hand (2,3) transport velocities, preserving one common joint-rate scale.

- [ ] Add tests for involution, det=+1, mirrored X/Z axes, invalid matrices and nonfinite points.
- [ ] Add tests applying unequal bilateral Z velocities through synthetic and actual USD Jacobians, checking the same scaling and zero palm angular velocity.
- [ ] Run `bash scripts/python.sh -m pytest tests/test_dh116_symmetric_geometry.py tests/test_dh116_coordinated_closure.py -q`; observe red for missing mirror module/new shape rejection.
- [ ] Implement `S=np.diag([1.,-1.,1.]); right_R=S@R@np.diag([1.,-1.,1.])`. Validate shapes, finiteness, orthogonality and det(R)=+1 before mapping.
- [ ] Normalize transport input: broadcast (3,) to (2,3), reject other shapes/nonfinite/norm>0.01 per hand, then pass `transport[i]` to each `_velocity_candidate`; leave common scaling unchanged.
- [ ] Re-run targeted tests and complete regression; record results without claiming grasp success.

## Task 2 — Measured synchronous transport

数值组件已实现：20 项针对性测试通过，完整回归 629 passed in 53.47s。检查点序列化/加载和运行器接线留在 Task 4 验证，不能认为已完成实际同步抬升。

Files: create `src/go2_pvcnn/control/dh116/symmetric_transport.py`, `tests/test_dh116_symmetric_transport.py`.

Interface: `SynchronizedTransport.update(time_s, rises_m, contacts_valid, direction)` returns `velocity_w` (2,3), common target, height difference and waiting state. Direction is +1 lift or -1 descent; baseline-relative measured rises are required. Checkpoint state stores target, time, direction and wait state, not inherited contact dwell.

- [ ] Test equal rises, left/right leader, 1 mm enter/0.5 mm exit hysteresis, 2 mm rejection, invalid contacts, time reversal, NaNs and direction reversal.
- [ ] Run targeted tests and observe failure before implementation.
- [ ] Implement bounded target integration at 5 mm/s, 0.2 s ramp, target lead <=1 mm above lagging progress. Use 0.25 mm deadband and 2 /s height correction capped at 1 mm/s; clamp signed transport to [0,5] mm/s. Pause leading hand when hysteresis active. Reverse signed progress for descent, never lift during descent to compensate.
- [ ] Run target tests and integrate synthetic closed-loop unequal-response test proving wait/recovery without overriding joint limits.

## Task 3 — New mirror geometry and full approach certificate

Files: create `scripts/solve_dh116_symmetric_grasp.py`, `tests/test_dh116_symmetric_candidate.py`; consume `usd_kinematics.py`, `collision_scene.py`, calibrated surface references and `symmetric_geometry.py`. Write candidate evidence under a fresh `runs/` folder; accepted configuration goes in a new `configs/symmetric_grasp_v1` only after validation.

- [ ] Test functional axes derived from calibrated normals, mirrored target construction and opposite-material-station matching (36/4).
- [ ] Solve bounded bilateral IK using each model's actual joint limits and nonlinear coupling; residual includes pad center/normal targets and functional frame symmetry. Keep unselected fingers at verified clearance poses. Search seeds from existing valid configurations, not arbitrary penetrations.
- [ ] Compute and record center residual <=1 mm, axial offset <=1 mm, radial normal errors <=5 degrees, functional frame mirror error <=5 degrees. Reject invalid source geometry; run existing full swept approach screen for both arms.
- [ ] Compare analytical point Jacobians with actual USD finite differences and generate fresh assets/configuration pins. Preserve rejected candidates and reasons; do not overwrite the old configuration.

## Task 4 — Runtime, descent/release and restoration

Files: modify `grasp_sequence.py`, `stage_state.py`, `stage_runtime.py`, `scripts/verify_fixed_dual_panda_dh116.py`, `scripts/run_current.py`; expand corresponding tests.

- [ ] Test that new mode requires new symmetric configuration and explicit policy fingerprint; legacy mode remains compatible.
- [ ] Pass measured station rises and contact qualification into the transport coordinator, per-hand velocities into allocator; log target, mismatch, wait state and real loads.
- [ ] Test phase-scoped synchronization: other-task handoff bypasses mirror/height coordinator and accepts independent legal per-hand commands, while retaining physical safety. Do not carry synchronization target or waiting latch into another task.
- [ ] Add optional explicitly selected descent/support-confirmation/release demo phases after independently verified lift, not an automatic replacement for other tasks. Require <=1 mm baseline proximity, actual table support and <=1 mm/s rod speed for 0.1 s before release; require 0.5 s settled wire and sub-noise hand load afterward.
- [ ] Serialize/restore new transport fields with strict pins; requalify contacts/low speed after restore, never import old dwell as current evidence.
- [ ] Run all affected tests and full regression. GUI-only rendering must not change physical qualification criteria.

## Task 5 — Actual single-instance verification and handoff

- [ ] Cooperatively stop the old GUI when ready, confirm its PID exited, retain all evidence. Start only the newly pinned candidate.
- [ ] Verify approach, contact/closure and 1 s hold; then both material rises >=30 mm for 1 s and height difference <=1 mm, with all old guards passing.
- [ ] Verify synchronized descent and supported release separately. Save stage states, raw forces, actual motion, stop reasons and GUI frames/video.
- [ ] If rejected, use measured failure to revise the candidate/controller; do not weaken guards or claim partial lift as success.
- [ ] Update README/CHANGELOG with exact tests and physical evidence. Review diff and tests before committing; upload only if requested, with stage/improvement notes and remote SHA verification.

## Plan review

All spec components map to Tasks 1–5. Task 1 is pure numerical behavior and does not enable a new runtime. Task 3 must qualify geometry before Task 4's new policy can be launched; Task 5 remains required even if every unit test passes.
