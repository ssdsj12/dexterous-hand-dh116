# Continuous Motion and Recoverable State Faults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement the user's approved three rules: finite mimic deviation is diagnostic and continuously corrected without blocking lift; existing actuator limits remain; invalid physical state resets only affected environments and aborts only when recovery fails.

**Architecture:** Add an explicit continuous_motion mode to the existing tensor pipeline, enabled by standard manual/free-grasp launchers. Preserve legacy behavior when disabled. Keep raw measurements, actuator limits, signed force rewards and reset isolation; separate failure lifecycle from controller correction.

**Tech Stack:** Python 3.10, Torch 2.7, Isaac Sim 4.5, PhysX, pytest; actual tests run on the hand server through scripts/python.sh.

---

## Approved scope and evidence

User approved the three rules on 2026-10-30. This amendment supersedes the older requirement to preserve every historical mimic-error cutoff in the GraspXL A draft. It does not authorize a fixed-object curriculum or long PPO training.
Baseline diagnostic: Wire/Table are independent; empty wrists reach 10 mm; the normal grasp path resets after 20 steps at left-index error -0.020860 rad and wrist displacement about 1.63 mm.
No manufacturer calibration is asserted: angle/velocity/effort bounds are the existing configured bounds.

## Task 1: Controller and manual lifecycle

Files: tabletop_policy_control.py, tabletop_policy_force_region.py, tabletop_policy_lift_hold.py, tabletop_manual_control.py; tests/test_continuous_motion_control.py.

- [x] Write behavioral tests with task.continuous_motion=True: finite mimic .021 does not veto upward wrist permission, force correction remains bounded, hold does not pause solely on mimic, recovered manual reset preserves Run/lift intent, explicit Reset+Pause stays paused.
- [x] Upload only new tests to untouched server source and run:
  `bash /root/hand/scripts/python.sh -m pytest -q /root/hand/tests/test_continuous_motion_control.py`.
  Confirm assertion failures for old behavior before production edits.
- [x] Implement the mode branches:
  ```python
  continuous = getattr(task, 'continuous_motion', False)
  mimic_blocks_motion = severe & ~torch.full_like(severe, continuous)
  # Continuous correction stays subject to joint/slew/effort bounds.
  # Do not clear or fabricate the measured mimic_errors.
  ```
  Keep legacy branch unchanged; ensure ordinary mimic does not enter a whole-environment motion veto in new mode.
- [x] Add ManualCommands recovery that retains running/lifting, clears queued height, and restarts height from zero; call only after validated recovered failure. Fatal recovery errors propagate.
- [x] Run new tests and existing policy/manual suites; review spec compliance and code quality.

## Task 2: Measurement, isolated numerical recovery and integration

Files: tabletop_tensor_env.py, tabletop_tensor_world.py, tabletop_tensor_task.py, tabletop_tensor_training.py, scripts/train_dh116_tabletop_gpu_ppo.py, scripts/manual_dh116_control.sh, scripts/train_dh116_tensor_ppo.sh; tests/test_continuous_motion_recovery.py.

- [x] Add regressions using two environments for finite mimic .021/.031, NaN/Inf state, float32 observation overflow, clean reset, failed reset, immediate repeated failure, unaffected environment progress and reward finiteness.
- [x] Run those tests on unchanged server production files; save expected failures.
- [x] Add CLI `--continuous-motion` and propagate a bool onto TensorWorld, TensorTask and TensorEnv before reset/control. Add a versioned contract, prohibit silent old-model resume through existing contract equality.
- [x] In new mode, leave finite mimic values in telemetry but remove mimic-only failure labels and the .02/.03 physical guard cutoff. Disable the artificial early .005-rad training failure boundary; retain actual articulation joint boundaries and hardware-configured actuator saturation.
- [x] For invalid returned state, use the last valid task snapshot for the failed transition, mark failure/nonfinite, apply the existing failure penalty once, and reset only affected world slots before another physical substep. Never feed NaN to reward reductions or policy observations.
  ```python
  numerical = nonfinite_physics | task_numerical_failure
  candidate = legacy_recoverable | (continuous & recovery & live & (numerical | ~safe))
  # A pending/frozen slot that is invalid immediately after reset aborts;
  # do not repeatedly reset the same failing state.
  ```
- [x] Validate reset immediately and at the next real physics substep. If reset throws, is invalid, or immediately fails again, report the affected IDs and stop. Missing contacts with valid=false remain ordinary state.
- [x] Keep existing overload/drop/real geometry invalid-state criteria, actuator caps and force reward directions; new mode attempts one validated isolated reset for any invalid physical row, including faults outside the old recovery band. Record numerical reset counts and recovery mode in run contracts.

## Task 3: Runtime validation and delivery

Files: deployment-only diagnostic wrapper, README.md, CHANGELOG.md, this plan and design amendment.

- [x] Run targeted CPU suite on final server sources; no new dependency installation.
- [x] Launch one headless Isaac instance using the exact previous diagnostic with continuous mode enabled. Verify that the previously interrupted path completes lift; report requested/actual wrist displacement, raw force, actual object height and any reset reason. The new controller need not reproduce the same .02-rad crossing; CPU regressions cover finite errors through .04 rad and the real run records .012-rad diagnostics. Continue to distinguish wrist motion from successful object lift.
- [x] Inject one controlled invalid returned observation in a two-environment real runtime wrapper. Verify one isolated reset, finite policy observation/reward and continued other-slot physics. Confirm no optimizer/PPO updates.
- [x] Verify configured joint targets, velocity and effort limits are unchanged. Do not declare hardware calibration or successful grasp without evidence.
- [x] Obtain independent spec review, then quality review; fix concrete findings and re-run affected checks.
- [x] Update README/CHANGELOG with current implementation stage, exact commands/results and limitations. Run git diff --check, record a local commit and deployed source hashes, end diagnostics and confirm no Isaac process remains. Long PPO remains stopped.

## Execution notes

Use the existing dedicated local publishing clone on branch10-29; no additional worktree or dependencies are needed.
A bounded controller worker owns only Task1 files. Main owns Task2, launchers, deployment, integration and docs; no overlapping source writes.
Server source files are backed up and baseline SHA checked before replacement. Deployment-only services and relocated asset manifest stay intact.

Review amendments completed: raw SDK pre-return faults and reset readback validation; invalid action/state collection isolation without temporal credit gaps; tensor-only startup validation; complete failure statistics; independent packet recurrence guard. Final related suite: 27 files / 493 passed in 11.66s. Actual CUDA collection: 80 physics steps, 2 isolated recoveries, 0 optimizer/PPO updates. Delivery source identity is recorded outside the Git tree in delivery_receipt.json.
