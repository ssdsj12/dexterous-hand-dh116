# Retain the completed close grasp during lift

> Execution: current session, with independent review. Existing user authorization explicitly requires holding the grasp after close and raising the hands, and repairing verified problems during monitoring. No new success/reward design or guard relaxation is authorized.

**Goal:** Prevent fresh PPO grasp actions and midpoint compensation from changing an established grasp during lift.

**Architecture:** A per-environment `LiftGraspHold` captures the applied finger command, pressure-feedback offset, force setpoint and applied palm exploration offset on the accepted close/hold-to-lift substep. Coordinated control uses these references throughout lift. Feedback may adjust closure around the saved command to maintain force; independent safety governors retain priority. This holds controller references, not exact physical forces.

**Stack:** PyTorch tensor task/control, Isaac/PhysX world adapter, existing PPO/checkpoint contracts.

## Evidence and constraints

- Latest actual run stopped after iteration 37: environment 831 produced a 58.4288 N link force spike. The preceding trace contains lost contact on both hands, including seven consecutive zero-load samples on the left. This does not establish policy resampling as the sole cause of that spike.
- `tabletop_coordinated_grasp.control` recalculates joint targets, force requests and measured-midpoint palm compensation for all live phases, including lift. This violates the user's established close-then-hold-and-raise instruction.
- Keep close mechanical dwell 0.2 s and current normal-Z >= 0 gate, 4 s close timeout, 2–12 N policy force range per hand, existing rewards and 156-element observations.
- Keep successful lift defined by bottom >= 0.33 m, speed < 0.03 m/s, slip <= 0.02 m, signed normal-plus-friction Z >= 0 for 0.5 s. This task does not redefine success as a force-only event.
- Preserve 3.00001 rad/s measured stop, 0.02 rad hard joint tolerance, force/mimic/clearance guards and validated isolated resets. No concurrent Isaac instances; retain all prior runs and dirty files; no commit/push.

## Task 1: Capture and consume lift references

Files: new `src/go2_pvcnn/control/dh116/tabletop_lift_grasp_hold.py`; existing `tabletop_tensor_task.py`, `tabletop_coordinated_grasp.py`, `tabletop_tensor_world.py`; new `tests/test_dh116_lift_grasp_hold.py`.

- [x] Write failing behavioral tests for accepted-transition capture, unchanged references under opposing PPO actions, pressure feedback still active, applied-palm capture rather than queued request, safety overrides, and selective reset/inactive environments.
- [x] Run `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_lift_grasp_hold.py` and confirm expected missing-behavior failures.
- [x] Implement buffers `active`, `command`, `feedback_offset`, `force_target`, `palm_offset`; `capture(task, mask, palm_offset)` copies selected rows once, and `reset(ids)` clears only those rows.
- [x] Initialize the hold with coordinated control; capture after valid transition to phase 3 using `measurement['commanded_palm_offset_m']` from `TensorWorld.measure`. Manual phase-3 fixtures capture once at their first control step, using existing palm targets as fallback for direct phase-3 fixtures.
- [x] During lift use `saved_command + (current_feedback_offset - saved_feedback_offset) * direction`, keep `saved_force_target`, freeze applied palm exploration offset, and retain all downstream governors and common lift motion.
- [x] Verify the focused controller/task/world tests, including unchanged close behavior.

## Task 2: Provenance, review and actual validation

Files: `tabletop_tensor_training.py`, focused training/probe tests, this record.

- [x] Record exact `lift_grasp_hold` contract whenever coordinated hold is enabled. Strict training resume must reject missing/changed/reverse contracts; use a fresh output directory for actual validation.
- [x] Obtain independent review, fix demonstrated findings, rerun impacted tests and `git diff --check`.
- [x] Launch one fresh 4096-environment, 1024-episode-per-environment candidate with the same velocity-64 physics, seed 42 and unchanged reward configuration; repoint named TensorBoard only after verifying startup. Compare checkpoint contracts and save the difference.
- [x] Observe accepted close events, contact retention, actual bottom clearance, lift gates and all hard guards. Record success honestly; do not classify event count or hand motion as successful object lift.

## Offline verification (2026-10-28)

- Initial behavioral tests: 4 failed, 1 passed before implementation; contract helper failed before implementation; applied-palm validation additions: 3 failed, 1 passed before the validation fix.
- Final regression command: `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_lift_grasp_hold.py tests/test_dh116_coordinated_grasp.py tests/test_dh116_close_instant_force.py tests/test_dh116_tensor_task.py tests/test_dh116_tensor_world.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_control_probe.py tests/test_dh116_close_completion_telemetry.py` — 221 passed.
- Independent read-only reviewer: no remaining blocker for a fresh candidate; 43 focused tests passed independently. `git diff --check` passed.
- Guards still override retained references. Holding references does not guarantee constant physical force or successful object lift.
- Fresh actual candidate launched at 12:50:02 local time: `runs/tabletop_tensor_20261028_125002_lift_hold_fresh`, PID 1908893. Runtime validation is pending.

## Startup integration correction

The 12:50:02 candidate stopped in strict adapter validation before PPO sampling: the tensor measurement gained `commanded_palm_offset_m`, but the host reference did not expose it. Added the same applied-command field as a copied array to `GpuWorld.measure_batch`; strict comparison remains enabled. New `tests/test_dh116_applied_palm_adapter.py` covers fixed/mounted offsets, carriage lag, independent-copy behavior, successful startup comparison/selected reset and rejection of a mismatched offset. All three cases failed with the observed KeyError before the fix.

`DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_applied_palm_adapter.py tests/test_dh116_gpu_batch_reset.py tests/test_dh116_tensor_training.py tests/test_dh116_lift_grasp_hold.py` — 60 passed, independently repeated by the reviewer; `git diff --check` clean.

Fresh replacement: `runs/tabletop_tensor_20261028_125219_lift_hold_fresh`, PID 1909338, same launch arguments. First candidate artifacts retained.

The replacement passed actual adapter/preflight validation and completed PPO iteration 1. `training_start.json` confirms null resume source and zero initial iteration, timesteps, episodes and optimizer entries. `lift_hold_contract_comparison.json` shows exactly two differences from the 12:02 baseline: the new `lift_grasp_hold` contract and the corresponding `palm_exploration.lift_policy`; other recorded contracts are unchanged. TensorBoard on port 6009 and the read-only watcher now point only to the replacement run.

## Initial actual validation

At 2026-10-28T12:56:50.452136+08:00, the replacement was training at PPO iteration 7 / 688128 transitions, with 53 close completion events and 0 successful lifts. Maximum commanded lift offset in this snapshot: 0.004833333333333346 m; maximum sampled object bottom in the rollout: 0.3000093891960302 m. Lift remains frequently blocked by insufficient measured grip load. This validates actual startup and execution of the new controller, not reliable grasp retention or successful object lifting. No hard guard stop was recorded in this initial observation. Ordinary mimic failures reset only the affected environment. Snapshot: `runs/tabletop_tensor_20261028_125219_lift_hold_fresh/lift_hold_initial_validation.json`.

TensorBoard server and the VSCode tab were refreshed to the 12:52:19 run; the API data location was verified. The background watcher is read-only and does not perform automatic repairs.
