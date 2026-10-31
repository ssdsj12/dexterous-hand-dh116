# Student Transfer Repair Implementation Plan

> **For agentic workers:** use executing-plans for main-thread work and requesting-code-review for independent review. User has authorized autonomous execution.

**Goal:** repair the demonstrated teacher-to-student failure and publish verified source on branch `11-1`.

**Architecture:** retain the existing native 351/24 environment while diagnosing the first divergence. Separate telemetry, supervised fitting, deterministic execution and on-policy PPO evidence. Only introduce a versioned observation/action change if the diagnostic proves it is necessary.

**Tech Stack:** PyTorch, Isaac Sim4.5/IsaacLab, RSL-RL PPO, pytest, SSH, Git.

## Task 1: Preserve and reproduce

- [x] Compare teacher proof, BC load metadata, current PPO action fit and physical success contract.
- [x] Save immutable 4096 checkpoint and maintenance state; SIGINT only the matched training process.
- [x] Create branch `11-1` in the existing working directory without discarding pending source changes.
- [x] Run existing pure-BC deterministic evaluation for1800 steps from the original triad reset.
- [x] Record actual evaluation result and first divergence; no success claim from MSE.

## Task 2: Independent diagnostics and contract review

Files: `src/go2_pvcnn/control/dh116/tabletop_surface_student_evaluation.py`, `tests/test_surface_student_evaluation.py`.

- [x] Add meaningful tests for pure evaluation telemetry, model immutability and per-environment hold diagnostics; verify expected failure before implementation.
- [x] Export real per-condition/hold/drift and progress diagnostics while preserving executed actions and success criteria.
- [x] Independently audit observation aliases, teacher hidden state, normalization and optimizer/noise behavior; use exact source locations.

## Task 3: Minimal evidence-driven correction

Files: `tabletop_surface_imitation.py`, its CLI, and dedicated regression tests; runtime source only when the actual diagnosis requires it.

- [x] Check conflicting labels and critical holding commands on recorded teacher states with CPU analysis.
- [x] Write the smallest failing regression for the confirmed cause.
- [x] Implement the targeted correction; keep source dataset immutable and emit a derived actor/dataset provenance record.
- [ ] Re-evaluate exact physical-unit command errors and execute the corrected student from normal native resets, serially for both styles.
- [ ] If the correction fails, use actual hold/condition diagnostics to select one next hypothesis; do not bundle unrelated physics changes.

## Task 4: PPO continuation and release

- [x] Run core regression plus existing teacher/student/reset/MPC tests.
- [x] Independently review the final change and resolve material findings.
- [x] Validate actual PPO updates with corrected initialization/regularization and explicit provenance; preserve saved episode counts and budget.
- [x] Point TensorBoard and monitor to the current corrected run and clear maintenance only when native training is actually resumed.
- [x] Update README, CHANGELOG and repair report with commands, results and limitations.
- [ ] Commit only project source/config/tests/docs and necessary service files; exclude runs/cache/checkpoints/keys. Push `11-1`, verify remote SHA, report actual outcomes.

## Release decision (2026-11-01)

Original teacher replays still succeed 2/2. Baseline BC and holding-only BC fail; deterministic PPO475 lifts above30mm in16/16, but holds at most0.25s and succeeds0/16. Thus source defects and reward/optimizer/noise alignment are corrected and regression-tested; stable student grasping is still a learning/validation outcome, not a completed claim. The explicit repair resume retains475/53533 episodes and351D inputs. The optional401D experiment is not enabled for this continuation.

Native qualification of both student reset styles remains outstanding. The user-authorized release proceeds with clearly documented limits and ongoing native-success monitoring; no success threshold or hardware/physics bypass was added.
