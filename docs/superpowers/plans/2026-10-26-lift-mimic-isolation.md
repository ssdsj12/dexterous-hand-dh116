# Lift-only mimic pause diagnostic

User requested the next step after the failed control experiment. Continue the approved control-first investigation; do not change training rewards or deploy a relaxed gate to PPO.

**Design:** Eight slots: fixed force 4 N / 8 N × lift-only threshold 0.004 / 0.008 rad × two spatial repetitions. One full episode per slot is the initial falsification test. Additional repeats are justified only if outcomes need confirmation. Zero additional palm shift; original compensation retained. Joint recovery, force-offset pause and micro-palm pause still use 0.004. Independent 0.02/0.03 coupling protection, speed, force and table guards retained.

**Acceptance:** actual 3 mm / 0.1 s stable lift with original load/speed/slip conditions, then original30 mm /0.5 s success. Increased commanded movement alone is not success. If candidate fails, do not deploy it or alter rewards; use signed per-joint error and recovery targets to diagnose the next cause.

**Files:** `tabletop_tensor_task.py` receives a diagnostic-only per-env lift threshold; `tabletop_control_probe.py` configures pairs and records signed mimic errors; CLI adds probe mode and episode budget. Normal training has no override.

- [x] Test that 0.006 rad blocks default lift but allows candidate, while joint recovery, palm pause and pressure blocking remain unchanged.
- [x] Implement isolated override and signed joint telemetry; strict CLI probe-mode validation.
- [x] Run CPU tests including original environment/reset guards.
- [x] Run 8-env same-checkpoint paired simulation; preserve pre-reset traces and report actual outcomes.
- [x] Keep training paused unless a validated control configuration warrants resumption.

Review: single intervention; threshold is an experiment, not a claim of calibrated correctness. No fixed-joint correction or physics parameter changes.

## Outcome

v136 ran the eight paired slots but was stopped by the unchanged hard joint-limit guard at 14.4604 s. One episode completed with mimic_error; seven records are truncated. The 0.008 rad candidate is rejected, not deployed. See `../specs/2026-10-26-grasp-control-probe-results.md` and the run-local `rejection_summary.json`. Default lift pause remains 0.004 rad. Checkpoint 583 is preserved; PPO remains paused. CPU validation: 50 passed, 1 skipped (CUDA disabled).
