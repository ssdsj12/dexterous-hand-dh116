# DH116 Hand MPC with Panda Approach and Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete dual Panda + DH116 tabletop scene with Panda owning approach/lift and a 12D DH116-only MPC owning grasp closure/hold.

**Architecture:** Keep `verify_fixed_dual_panda_dh116.py` as the integrated Isaac entry point and preserve its USD assembly, wire backend, and measured sequence evidence. Route actual geometry-driven `ParallelContactMPC` outputs through a pure-Python phase/action router; only active DH116 coordinates pass through its action port. This is a local point-Jacobian prediction model, not a full Cosserat physics rollout. The phase coordinator freezes Panda targets during close/hold, retains the grasp target during lift, and logs the owner and target for each subsystem.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab/Isaac Sim, pytest, existing DH116 control modules and DeformX wire backend.

## Full-GPU rod priority (user continuation, 2026-10-21)

Latest re-verification: CUDA suites **27 passed in 5.26s**. Control/CUDA/physical close initialization/replay and imitation suites **93 passed in 10.41s**; compilation and diff checks pass. Rod dynamics/contact computations are CUDA; the outer Isaac/data/guard boundary still uses CPU. Do not interpret this as whole-scene real-time performance.

Full repository run: **788 passed, 71 failed in 53.79s**, including legacy four-channel fixtures/resources versus the existing six-channel CPU/contact/evidence contract and old approach timing expectations. Several affected production files/tests are unchanged from HEAD. Do not claim the full suite passes; do not change the production channel count to satisfy obsolete fixtures as part of this CUDA work.

- [x] Keep shear/stretch, bend/twist, inertia, director integration and PyElastica-compatible translation/rotation damping on CUDA.
- [x] Keep finite-table and uploaded finger-mesh contacts, prescribed substep collider prediction and independent rod/body reaction-impulse accumulation on CUDA. Source USD files are unchanged.
- [x] Preserve exact contact observations outside the force broad phase and verify production-size dynamic-state restoration.
- [x] Run `bash scripts/python.sh -m pytest tests/test_gpu_cosserat_physics.py tests/test_gpu_cosserat_runner.py tests/test_gpu_cosserat_backend.py -q`: **26 passed in 2.69s**. Includes a 40-element, 0.5 m / 0.008 m rod under nonuniform load and table contact.
- [x] Run the full USD scene with `--rod-backend torch_cuda --table-wire --probe-channel left_index --probe-overlap-m .0001 --contact-subsamples 16 --gravity-hold --arm-effort-limit 30 --external-force-iterations --physics-dt .0001 --steps 121 --render-every 121 --evidence-flush-policy buffered --profile --headless --device cuda:0`; use the six-channel `configs/symmetric_grasp_middle_v1/contact/manifest.json` and matching point references. Result: `runs/dh116_full_cuda_rod_probe_20261021_v3/summary.json`, **passed=true**, 121 steps, 0.272332425 N peak, robot reaction submitted, six point/Jacobian probes passed. This is an initial-overlap coupling diagnostic, not a qualified tabletop grasp/lift or paired-response test.
- [ ] Continue actual grasp/lift with separated Panda/hand actions and then collect successful physical trajectories for teacher/student training. The existing offline `train_dh116_teacher_student.py` backend flag is metadata; it does not itself instantiate the GPU rod or prove physical planning.

Performance evidence: this small eager-Torch integration measured 0.0975202 s/outer step (RTF 0.001025 at 100 us). Full CUDA residency alone has not established a speedup or real-time play. First probe v1 stopped before stepping because the default 5 Nm arm limit could not compensate gravity; v2 completed 20 steps but had no scheduled point probes. Preserve both as diagnostic evidence.

Combined verification after migrating the old four-channel sequence test fixtures to the current six-channel contract: CUDA suites plus `test_dh116_integrated_hand_mpc.py`, `test_dh116_panda_lift.py`, `test_dh116_sequence_runtime.py`, `test_dh116_teacher_student.py` and `test_dh116_parallel_mpc.py` **93 passed in 2.95s**. `git diff --check` and solver/runner `py_compile` passed. The verified GPU source SHA is `986da8eb01e0e0c7cbd4fc04cb76ab075f2fd31c8a52a3dceeb5598f5dc710a2`, matching the Isaac run's backend pin.

## Global Constraints

### Actual contact MPC and grasp-local entry (continuation)

- [x] Add `contact_mpc.py`: bounded 12D shooting MPC with a first-order joint-tracking/point-Jacobian prediction model, exact linear least-squares trajectory seed and CUDA-parallel CEM refinement (128 candidates, 8-step horizon, 4 iterations). This is a reduced kinematic MPC, not a full Cosserat rollout optimizer.
- [x] Keep the previous actuator command as the reference and enforce per-update joint-speed/limit bounds. Round float32 bounds inward for the float64 coupling interface; an actual step-200 limit-conversion failure was reproduced and fixed.
- [x] Feed measured DH116-only Jacobian columns `[:,7:]`, and current CUDA wire geometry, into the integrated controller. Save optimized actions, their previous reference, world pad/goal points and optimizer diagnostics in physical evidence rows.
- [x] Add `--grasp-entry close` for direct physical initialization at the nominated grasp pose; phase summaries mark settle `INITIALIZED` and approach `NOT_RUN`. Add `--grasp-contact-mode pinch`; middle support remains mandatory for the existing default `pinch_hook` mode. In pinch mode the middle finger is parked and is not a pre-lift qualification requirement.
- [x] Prepare `configs/contact_mpc_close_v1/pose.json` from the symmetric nominal endpoint with both palms raised 0.5 mm using bounded Panda IK; hand coordinates/rotations unchanged. New static and initial measured clearance screens pass. These are initial-state checks, not a grasp.
- [x] Run a real direct-close trial with full dual-Panda/DH116 scene and CUDA wire. `runs/dh116_contact_mpc_close_gpu_20261021_v3` executed 200 valid close steps and six point/Jacobian probes (max Jacobian discrepancy 3.37e-7 m/rad), then stopped at the float32 coupling-bound issue. Arm ownership stayed frozen; hand ownership was MPC. No grasp/lift passed.
- [x] Diagnose the original fixed-reference targets: five fixed-palm nonlinear IK seeds per side leave 7.27–8.82 mm residual. V4 was cooperatively stopped to correct that observation model; evidence and the offline result remain in its run directory. Do not interpret this local IK result as a global infeasibility proof.
- [x] Add a CUDA `closest_contact_point` observation query and use the actual closest mesh-surface point for the point Jacobian and radial wire target. Initial measured-geometry candidates are 1.37–2.69 mm away, with thumb/index opposition dots -0.9647/-0.9524. The selected calibration reference still defines the allowed pad region but is not the MPC's enforced contact point.
- [x] Focused CUDA/hand/control suites: 84 passed in 3.00s. Latest provider wiring/control checks: 61 passed in 1.88s; runner compilation and diff checks passed.
- [x] V5 confirms decreasing four-channel gaps but terminates at step 91 because the old local pad region is centered/oriented at the obsolete reference: non-front allowance 0.999776 mm. Recalibrate the four pinch references at the actual CUDA collision-hull closest points using `scripts/calibrate_dh116_wire_contact_points.py`; keep the middle references and byte-identical contact meshes. All six new static pad-region screens pass. Output: `configs/contact_mpc_close_v1/calibrated/`.
- [x] Inspect V6: four real pinch contacts stabilize near 0.012 N, insufficient for 1 N targets. Cooperatively stop at 4765 steps after saving actual robot and rod physical state at step 4727. This is a failed/partial diagnostic, not a grasp success. Fix the excessive tangential point penalty and add measured-load reference feedback; focused tests pass.
- [x] V7 cooperatively stopped at 1603 valid steps: loading plateaus near .14 N. Map recorded contact normals through USD point Jacobians: 1 N requires .074–.079 Nm; default Kp1 cannot meet 1 degree tracking at that load. Add a hand-only `J.T*n*load/Kp` equilibrium offset to the local MPC and choose simulation Kp6 with torque-preserving initial-target rebase, unchanged .15 Nm cap.
- [x] V8 reaches four pinch loads .50–.55 N, then stops at 485 on left-thumb non-front bound .999842 mm. Fine recomputation bounds it above 1.19 mm. Add bounded adaptive line-segment sampling only for ambiguous bounds, preserving the original cap, 1 mm margin and penetration checks. Portable regression retains actual-penetration rejection; the recorded failure reproduces and now passes.
- [x] Wire neural teacher/student `.pt` into the full-scene adapter (`--hand-policy-role`), retain JSON compatibility, validate neural control interval, preserve parked-joint commands, and retain diagnostic/success training metadata. Fix pinch-only lift synchronization so it does not demand middle support.
- [x] Inspect V9: session82195/PID134144 **terminal**, failure step259, last valid258. Loads .527–.589 N, no hold/lift. The next coarse source guard fails on left finger13_link (.826 mm). Refined convex-plane envelope plus the unchanged motion debit certifies 1.0694 mm at the recorded failure; all 50 source bodies pass offline. Save last valid contact state rather than repeating approach: V9/contact_initialization_feedback.json SHA `9bf576b3efb2e0b387dcf17dfb890e1141e870bb6e2d37891baa9ece365eec70`.
- [x] Prove a local hand-only clearance direction exists: retain the actual CUDA wire gap and increase finger13 clearance to1.6 mm with thumb changes left `[+.02916045,-.00920029]` and right `[+.01096427,-.00371734]` rad. Panda/other hand joints fixed. `V9/hand_only_clearance_feasibility.json` is offline geometry only, not a qualified trajectory or new approved pose.
- [x] Latest focused checks (GPU/control/replay/imitation/close initialization/region regressions): **111 passed, 4 deselected in9.21s**. Four excluded tests use the known incompatible legacy four-channel fixture. Compilation/diff pass; full repository suite not rerun and prior71 failures remain unresolved.
- [x] Further convex-envelope and source-geometry checks:27 passed/1 legacy deselected in5.94s. Latest full suite804 passed/71 failed in29.00s (`/tmp/dh116-regression-20261021-contact-control.log`); do not claim global passing.
- [ ] Train teacher/student models from the actual optimized physical trajectories, then run the student with Panda-owned lift. The existing offline reference/trajectory checkpoint is not the result of this new physical MPC collector.

- The integrated scene contains both Panda arms, both DH116 hands, table, and wire; no hand-only scene substitution.
- Panda controls approach and lift. Its seven joints per side never appear in the MPC action.
- MPC controls only 12 active hand coordinates; passive hand joints remain derived by `expand`.
- During close/hold, Panda arm targets remain at the measured approach endpoint. During lift, hand targets remain at the MPC-established grasp target.
- A lift is successful only with measured grasp/contact and wire-rise evidence from the existing sequence; a finger-close command is not success.
- Keep single-core/single-thread configuration and never launch a concurrent Isaac instance.
- Preserve existing dirty files and the untracked user file `1332`; do not add runs, logs, caches, or generated evidence to git.
- No GitHub upload is in scope.

### Next required fix: include non-pad clearance in the hand MPC prediction cost

The current guard can stop an invalid motion, but the MPC objective only tracks pad/wire geometry. A feasible thumb null-space direction has now been measured offline, so connect that information to planning rather than repeatedly restarting at the guard.

- [ ] Add optional current non-pad clearances `(C,)` and hand-only clearance Jacobians `(C,12)` to `ParallelContactMPC` and `IntegratedHandMPC`. Penalize predicted clearance below a 1.5–1.6 mm planning buffer with a squared hinge; the physical guard remains1 mm. Include active linear constraints in the directed seed so CEM does not need to discover a null-space thumb rotation randomly. Keep the existing point/force objectives and 12D action/previous-command bounds.
- [ ] Gather live near-wire non-pad constraints from the mounted hand geometry (finger13 is currently critical), with Panda frozen. Only six hand columns per side enter the derivatives. A point-plane witness or a finite-difference source-FK Jacobian can be checked against the recorded geometry solve; no Panda coordinates enter the optimizer port. Do not exempt finger13 or widen the allowed pad patch.
- [ ] TDD: a two-thumb fixture must retain radial contact while increasing non-target clearance; reject Panda-width constraint matrices; no-constraint behavior stays unchanged; use recorded V9 state for an offline consistency check. Keep the source/region guard active.
- [ ] After wiring and tests, launch exactly one new full-scene trial from V9 step258 saved physical state above. No approach replay, no thumb teleport to the offline candidate, no imported successful-stage evidence. Verify actual loads, all-body clearance, hold and Panda-only lift.
- [ ] Only then train successful physical teacher/student data and run the learned `.pt` controller in the full scene. A low cloning loss or a fast inference benchmark is not completion.

---

## File Structure

- `src/go2_pvcnn/control/dh116/integrated_hand_mpc.py`: pure phase routing and 12D hand planner adapter; no Isaac imports.
- `src/go2_pvcnn/control/dh116/panda_lift.py`: 7-DOF-per-arm transport increment computed from measured fingertip-point Jacobians with hand columns excluded.
- `tests/test_dh116_integrated_hand_mpc.py`, `tests/test_dh116_panda_lift.py`: action ownership, dimensions, bounded arm-only lift, and hold tests.
- `scripts/verify_fixed_dual_panda_dh116.py`: add an opt-in `--hand-mpc` mode to the existing complete-scene runner, keep the existing measured Panda approach and physics evidence, route close/hold targets through hand MPC, use the hand-excluded Jacobian allocator for Panda-only lift, and log per-phase control ownership.
- `tests/test_dh116_sequence_runtime.py`: assert integrated runner preserves both arms and wires the MPC output only to DH116 active targets.
- `README.md`, `CHANGELOG.md`: update only after actual Isaac verification, clearly separating code/test success from simulation grasp/lift result.

### Task 1: Add a pure decoupled phase/action contract

**Files:**
- Create: `src/go2_pvcnn/control/dh116/integrated_hand_mpc.py`
- Create: `tests/test_dh116_integrated_hand_mpc.py`
- Reference: `src/go2_pvcnn/control/dh116/parallel_mpc.py`
- Reference: `src/go2_pvcnn/control/dh116/linear_coupling.py`

**Interfaces:**
- `IntegratedHandMPC(planner: ParallelJointMPC)` accepts measured `hand_poses: ndarray(2,7)`, `active_joints: ndarray(2,6)`, `grasp_progress: float`, and `previous_action: ndarray(12,)`; `plan(...) -> ndarray(2,6)` calls the existing planner using a `HandOnlyState` with `wire_features=[[grasp_progress, 0.0]]`, returns only bounded active-hand targets, and updates its own previous 12D action.
- `route_phase_targets(phase, arm_approach, arm_lift, arm_frozen, hand_approach, hand_mpc, hand_hold)` accepts arm arrays shaped `(2,7)` and hand arrays shaped `(2,6)`, and returns `(arm_targets, hand_targets, owners)` where owner labels are explicit strings.
- Supported phases are `approach`, `close`, `hold`, and `lift`; unknown phases, non-finite arrays, and wrong dimensions raise `ValueError`.
- Routing table: approach → (arm approach, hand approach); close → (frozen arm, MPC hand); hold → (frozen arm, retained MPC hand target); lift → (Panda arm lift, retained MPC hand target).
- `panda_only_transport_increment(point_jacobians: ndarray(2,3,3,7), palm_angular_jacobians: ndarray(2,3,7), transport_velocity_w: ndarray(2,3), dt: float, speed_limit_rad_s: float=0.05) -> ndarray(2,7)` solves a per-arm least-squares Cartesian transport command using only Panda columns, limits joint speed, and returns joint increments. It rejects singular/non-finite input.

- [ ] **Step 1: Write failing contract and routing tests**

```python
import numpy as np
import pytest
from go2_pvcnn.control.dh116.integrated_hand_mpc import (
    IntegratedHandMPC, route_phase_targets,
)
from go2_pvcnn.control.dh116.parallel_mpc import ParallelJointMPC


def inputs():
    return dict(arm_approach=np.ones((2, 7)), arm_lift=np.full((2, 7), 2.),
                arm_frozen=np.full((2, 7), 3.), hand_approach=np.ones((2, 6)),
                hand_mpc=np.full((2, 6), 4.), hand_hold=np.full((2, 6), 5.))


@pytest.mark.parametrize("phase,arm_value,hand_value,arm_owner,hand_owner", [
    ("approach", 1., 1., "panda", "approach_trajectory"),
    ("close", 3., 4., "frozen", "hand_mpc"),
    ("hold", 3., 5., "frozen", "hand_mpc_hold"),
    ("lift", 2., 5., "panda", "hand_mpc_hold"),
])
def test_phase_routing_isolates_arm_and_hand_actions(
        phase, arm_value, hand_value, arm_owner, hand_owner):
    arms, hands, owners = route_phase_targets(phase, **inputs())
    assert arms.shape == (2, 7) and hands.shape == (2, 6)
    np.testing.assert_allclose(arms, arm_value)
    np.testing.assert_allclose(hands, hand_value)
    assert owners == {"arm": arm_owner, "hand": hand_owner}


def test_hand_mpc_returns_only_bounded_12d_active_action():
    planner = IntegratedHandMPC(ParallelJointMPC(smoothing=0.0))
    action = planner.plan(np.zeros((2, 7)), np.zeros((2, 6)), 0.5,
                          np.zeros(12))
    assert action.shape == (2, 6)
    assert np.isfinite(action).all()


def test_route_rejects_invalid_phase_and_action_shapes():
    with pytest.raises(ValueError):
        route_phase_targets("approach_and_close", **inputs())
    bad = inputs(); bad["arm_lift"] = np.zeros((2, 6))
    with pytest.raises(ValueError):
        route_phase_targets("lift", **bad)
```

- [ ] **Step 2: Run tests and confirm the missing interface fails**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_integrated_hand_mpc.py -q`
Expected: collection fails because `integrated_hand_mpc.py` does not exist.

- [ ] **Step 3: Implement only the tested adapter and router**

Implement `IntegratedHandMPC` by constructing a `HandOnlyState` with the live inputs and current previous 12D action, calling `planner.plan`, converting the returned tensor with `to_numpy`, reshaping `(12,)` to `(2,6)`, and storing a copy. Do not add arm fields to the MPC state or action. Implement `route_phase_targets` with the four-row table above, validating exact shapes, finite values, and returning copies so the caller cannot mutate controller inputs through aliases. In hold/lift, `hand_hold` is the latched MPC result and the reported owner remains `hand_mpc_hold`.

- [ ] **Step 4: Run the focused tests**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_parallel_mpc.py -q`
Expected: PASS; phase routing shows that only hand targets change under `close`, and MPC output is `(2,6)`.

- [ ] **Step 5: Add Panda-only lift Jacobian allocation tests**

Create `tests/test_dh116_panda_lift.py` with one full-rank synthetic fixture: for each side, use three point Jacobians whose first three arm columns are `0.02 * eye(3)` and remaining four arm columns are zero; provide a zero palm angular Jacobian; request `[0, 0, 0.001]` m/s at `dt=1e-4`. Assert result shape `(2,7)`, both hand coordinates are structurally absent from the API, reconstructed fingertip velocity is `[0,0,0.001]` (within `1e-8`), each joint speed is at most `0.05` rad/s, and a zero/singular Jacobian raises `ValueError`.

- [ ] **Step 6: Implement Panda-only lift allocation and run its test**

Create `panda_lift.py`. For each side form `A = vstack((point_jacobians[side].reshape(-1,7), 0.05*palm_angular_jacobians[side]))`; form `b` by repeating the requested Cartesian velocity for each of the three points and appending three zeros. Solve `np.linalg.lstsq(A,b,rcond=None)`, reject rank below 3 or non-finite results, scale the vector uniformly when its maximum absolute speed exceeds `speed_limit_rad_s`, then multiply by `dt`. Return `(2,7)` increments. Run `bash scripts/python.sh -m pytest tests/test_dh116_panda_lift.py -q` and expect PASS.

- [ ] **Step 7: Commit the tested control contract**

```bash
git add src/go2_pvcnn/control/dh116/integrated_hand_mpc.py \
  src/go2_pvcnn/control/dh116/panda_lift.py \
  tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_panda_lift.py
git commit -m "feat: define Panda and DH116 MPC control interfaces"
```

### Task 2: Route the existing full-scene Isaac sequence through hand MPC

**Files:**
- Modify: `scripts/verify_fixed_dual_panda_dh116.py`
- Modify: `tests/test_dh116_sequence_runtime.py`
- Use: `assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd`
- Use: `src/go2_pvcnn/control/dh116/grasp_sequence.py`

**Interfaces:**
- Add opt-in CLI flag `--hand-mpc`; require `--grasp-sequence` when set. Existing non-MPC behavior remains unchanged when the flag is absent.
- Instantiate `ParallelJointMPC(device=args.device)` and `IntegratedHandMPC` only for this mode.
- Capture two measured world hand poses from `base_link`, two six-active-joint vectors from each integrated arm, and previous active targets. Update MPC every `--hand-mpc-update-steps` simulation steps (default 1000). At each update, use `advance_progress` once with the maximum measured active-pad load and measured-valid-contact latch; use only its grasp progress (lift progress is always zero because Panda owns lift).
- During approach, retain the existing Panda+hand approach endpoint. At transition into close, snapshot actual Panda arm targets and initialize the MPC target from measured hand joints. During close, hold the Panda snapshot and route MPC outputs only to the 12 hand active joints. On transition to hold/lift, latch the last MPC hand action. During lift, compute contact-point Jacobians with the seven Panda columns only, obtain the measured `transport_velocity_w` from `TabletopGraspSequence`, use `panda_only_transport_increment`, and hold the hand target fixed at the MPC latch.
- Continue applying the current measured source-clearance, Jacobian/contact, reaction, and grasp-evidence checks. Do not run the 10-DOF `coordinated_bimanual_increment` to change hand targets in MPC close mode; for lift, use only the new seven-column Panda solve. Non-MPC runs retain their legacy path.

- [ ] **Step 1: Add failing static/runtime boundary tests**

Add tests in `tests/test_dh116_sequence_runtime.py` that parse `scripts/verify_fixed_dual_panda_dh116.py` and assert: (a) `--hand-mpc` exists and is gated by `--grasp-sequence`; (b) the script still spawns `/World/Rig` from `fixed_dual_panda_dh116.usd`; (c) calls `route_phase_targets` and `panda_only_transport_increment`; (d) the MPC action is indexed by `ACTIVE` joint names only; (e) arm targets during close come from the frozen approach snapshot; and (f) lift Jacobians are sliced to seven Panda joints before allocation.

- [ ] **Step 2: Run the new test and confirm it fails on current runner**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_sequence_runtime.py -q`
Expected: FAIL because `--hand-mpc` and the routing call do not yet exist.

- [ ] **Step 3: Add CLI gating and controller initialization**

Add `parser.add_argument('--hand-mpc', action='store_true', help='Use 12D DH116 MPC for close/hold while Panda controls approach/lift')` and `parser.add_argument('--hand-mpc-update-steps', type=int, default=1000)`; reject `args.hand_mpc and not args.grasp_sequence`, and nonpositive update intervals, before `AppLauncher`. Inside the existing integrated runner setup, instantiate the planner and adapter only when enabled. Leave the full USD scene creation and arm articulation unchanged.

- [ ] **Step 4: Route each sequence phase without changing action ownership**

At each control tick, compute the existing approach Panda targets as before. For `--hand-mpc`, collect live hand poses/active joints and actual contact values. At each configured MPC interval in `close`, update grasp progress from measured contact using `advance_progress`, call the adapter, and cache the resulting `(2,6)` target. Route approach/close/hold with `route_phase_targets`; in lift calculate each active pad point Jacobian through existing `point_jacobian(... )[:, :7]`, calculate palm angular Jacobian with the same seven arm indices, and call `panda_only_transport_increment`. Publish arm targets by Panda joint names and hand targets by `ACTIVE` names followed by existing `expand` for followers. In close/hold, arm target must equal the frozen approach endpoint; in lift, hand target must equal the stored MPC hold target. Keep legacy coordinated force/closure control for runs without `--hand-mpc`.

- [ ] **Step 5: Record ownership and actual routed values**

For `--hand-mpc`, add a `control_ownership` object to each evidence row with `phase`, `arm_owner`, `hand_owner`, `arm_target_rad` (two arrays of seven), `hand_target_active_rad` (two arrays of six), `mpc_action_dim: 12`, `mpc_update_steps`, and current `grasp_progress`. Add matching run-config flags to the output summary. Do not mark the row or run successful because the routing itself executed.

- [ ] **Step 6: Run focused and regression tests**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_panda_lift.py tests/test_dh116_parallel_mpc.py tests/test_dh116_sequence_runtime.py tests/test_dh116_state_triggered_sequence.py tests/test_dh116_stage_reset_runtime.py -q`
Expected: PASS; existing legacy sequence tests still pass, and new tests assert the full scene and separated targets.

- [ ] **Step 7: Commit the integrated runner change**

```bash
git add scripts/verify_fixed_dual_panda_dh116.py tests/test_dh116_sequence_runtime.py
git commit -m "feat: use DH116 MPC between Panda approach and lift"
```

### Task 3: Verify the full Isaac scene and report measured outcome

**Files:**
- Modify after the run: `README.md`
- Modify after the run: `CHANGELOG.md`
- Evidence output: a new exclusive directory under `runs/` (never tracked)

- [ ] **Step 1: Run the full focused CPU/unit suite**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_panda_lift.py tests/test_dh116_sequence_runtime.py tests/test_dh116_grasp_sequence.py tests/test_dh116_state_triggered_sequence.py tests/test_dh116_parallel_mpc.py -q`
Expected: PASS before launching Isaac.

- [ ] **Step 2: Run exactly one Isaac Sim integration instance**

Use the project's existing `verify_fixed_dual_panda_dh116.py --help` command to construct the established full-scene tabletop grasp invocation, preserving its pinned active pose, approach certificate, point references, contact manifest, wire backend, gravity, and required closure-load arguments; add `--grasp-sequence --hand-mpc`. Choose a new output path and use `--device cuda:0` only if the existing Isaac environment supports it. Do not start a second instance while it runs.

- [ ] **Step 3: Inspect evidence for scene, ownership, and physics results**

Confirm the USD path and run configuration identify the full fixed dual Panda/DH116 assembly; confirm each phase's owner mapping and target dimensions; inspect measured approach transition, contact/load history, held contact, wire-height rise, and final stage status. Report `lift PASS` only if the existing final grasp/lift gate passes. If MPC close or lift fails, preserve the failure evidence and report the first failed physical criterion; do not claim a successful trajectory.

- [ ] **Step 4: Update progress docs with this run's actual result**

Add one concise README progress entry and a CHANGELOG entry stating current stage, improvement (Panda visible and owns approach/lift; hand MPC owns close/hold), exact tests and Isaac command/outcome, simulation-only limitations, and remaining step. Never copy an earlier run result as the result of this run.

- [ ] **Step 5: Commit the verified implementation and result record**

```bash
git add README.md CHANGELOG.md scripts/verify_fixed_dual_panda_dh116.py \
  src/go2_pvcnn/control/dh116/integrated_hand_mpc.py src/go2_pvcnn/control/dh116/panda_lift.py \
  tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_panda_lift.py tests/test_dh116_sequence_runtime.py
git commit -m "feat: integrate DH116 hand MPC with Panda approach and lift"
```

Do not add run outputs, the existing unrelated working-tree modifications, or `1332`.

## Plan Self-Review

- Spec criteria 1–2 are covered by the full-scene runner and phase ownership evidence in Task 2, then the rendered Isaac run in Task 3.
- Spec criteria 3 is covered by the adapter shape/bounds tests and runner active-joint routing assertions.
- Spec criteria 4–5 are covered by retaining the existing measured grasp/lift gate and requiring a real integration result; failure evidence remains failure.
- The original hand-only/arm routing tasks preserve the USD assets and hardware calibration claims. The user's later full-GPU instruction adds the rod-backend work above before continuing the grasp/lift tasks.
- Existing unrelated modifications and `1332` remain outside every `git add` command.
