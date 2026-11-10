# Tripod recovery staged implementation plan

> **For agentic workers:** Use subagent-driven development with test-first implementation, separate spec/quality reviews and root ownership of integration and the sole native Isaac instance.

**Goal:** Recover and maintain actual right-hand Tripod contacts under the original per-episode budgets, then advance through native A–D gates before Student training.

**Architecture:** Independent pure recovery ledger, calibrated nonlinear source-region geometry, and native substep diagnostics feed an opt-in staged expert. Published pinch/pressure/acquisition/constrained paths remain unchanged. Root owns post-guard execution integration, native freezes and publication.

**Tech Stack:** CPU PyTorch/NumPy/SciPy and source USD geometry; existing 480-Hz TensorEnv, 60-Hz canonical 24D controls, actual SDK link/joint state and fixed RGB-D.

**Current evidence:** Pure components and the opt-in A runtime are implemented and independently reviewed. Latest full CPU uses the final1334-file source:5917 passed, all5786 baseline node states unchanged,304 new nodes all passed, and the final source manifest remains unchanged afterward. Original96 failures/28 errors/49 skips remain. The earlier CPU05 5883/270 A02 snapshot and targeted owner-first52-pass set remain historical evidence. Native01 failed before A control (zero A trials); native02 entered real 480-Hz substeps but was operator-stopped and independently marked censored/invalid:481 logical controls,482 physical intervals,3856 raw SDK substeps,0 approach submissions and0 complete episodes. The original rigid-cylinder proxy is not validated flexible-cable or hardware grasping. No A pass or B–E execution is claimed below.

## Task 1: Isolation, baseline and evidence

- [x] Verify linked worktree/branch at46dc091, preserve untracked artifacts and all historical runs/models.
- [x] Audit state-machine locks, nonlinear FK/regions, pre-reset observer and physical effort/support fields independently.
- [x] Create fresh server stage `/mnt/mydisk/xk/projects/dh116-tripod-recovery-20261107` and run root `/mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z`; record path-only manifest rebase and text fixture EOL restoration.
- [x] Run existing focused baseline with hidden CUDA and vendor USD libs:168 passed. Compare full CPU04 with the previous phase's CPU05 baseline:all5786 old node states unchanged.

## Task 2: Pure per-pad recovery ledger

Create `src/go2_pvcnn/control/dh116/tabletop_tripod_contact_recovery.py` and `tests/test_tripod_contact_recovery.py`.

- [x] RED/GREEN: use `TripodContactRecovery(...).observe_substep(...)` with480-Hz measured-packet fixtures; nine valid samples do not qualify20ms, ten do. Assert no predicted geometry creates contact.
- [x] RED/GREEN: test independent right thumb/index/middle states, loss/unknown distinctions, stable/flash transitions, reset isolation and stale/duplicate execution IDs.
- [x] RED/GREEN: commit zero on denied candidates; commit only final approved approach; conserve1mm/1s across retries/flicker and lock exhausted pads until reset. Test .5s measured no-progress stop and owned snapshots/events.
- [x] Implement the pure tensor state machine and ledger, preserving requested/approved/measured quantities separately;37 focused tests and independent review pass. These are CPU cases, not native contact recoveries.

## Task 3: Nonlinear FK and real pad regions

Create independent `tabletop_tripod_nonlinear_geometry.py` and tests, with focused submodules only if geometry responsibilities become too large.

- [x] RED/GREEN: zero-delta link identity, ideal-USD equivalence, measured follower residual preservation, full-joint limits, left/right frame distinction and rotating normals in CPU/source fixtures.
- [x] RED/GREEN: finite-cylinder witness on actual source facet∩hull∩ellipse, middle-pad independence, negative center-gap not a contact/penetration predicate, same-point distance/facing checks.
- [x] RED/GREEN: complete34-hull coverage, source identity, table/original supports/self/interhand/rod clearance, uncertain separation rejection, no sampled fallback and encoded/substep trajectory revalidation.
- [x] Implement source/layer validation, immutable calibrated snapshot, nonlinear region evaluator and whole-trajectory candidate planner. Freeze explicit threshold sources;26 geometry and15 planner tests plus independent source-math reviews pass.
- [ ] Complete independent native moving-frame FK/geometry comparison from the new A run. Zero-delta calibration and ideal/source CPU fixtures do not satisfy this item.

## Task 4: Native480Hz evidence and support gates

Create `tabletop_tripod_support_diagnostics.py`, `tabletop_tripod_support_allocation.py` and their tests. Root integrates optional hooks in `tabletop_tensor_env.py`, `tabletop_free_wrist_grasp.py` and native sensor snapshot code only after pure interfaces pass.

- [x] RED/GREEN: per-substep six-pad continuity cannot be replaced by means; terminal before reset and fatal flush; all three clocks/env/episode/stage IDs; palm reset baseline; null slip/torque coverage.
- [x] RED/GREEN: post-guard execution charging/remaining-budget reservation; raw requested action differs from actual final command; no effect on old/pinch paths.
- [x] RED/GREEN: normal/friction/rejected load identities, simulated applied/computed effort vs FF command and actual drive readback; noncollinear manifold sums. SDK capture qualification remains subject to native audit.
- [x] RED/GREEN: COM asymmetric support wrench allocation, invalid capacities and infeasible force/moment balance; no default half-share. Static proposals are not force-control validation.
- [ ] Implement/test D-specific loss cancellation of queued common upward targets against measured palms and resume only with real bilateral qualification; no teleport or limit changes. D runtime is not enabled by the A-only entry.
- [x] Implement opt-in raw-first observer, execution guard, diagnostics and A/B/C/D diagnostic gates; independent spec/quality reviews pass. Missing original table/support load coverage remains unknown, so weight support cannot be certified.
- [x] RED/GREEN: preparation-prefix recovery cannot qualify A; track stage-local loss/recovery separately while conserving physical-episode histories and budgets.

## Task 5: New expert and stage controller

Implemented `tabletop_tripod_recovery_runtime.py`, `tabletop_tripod_pressure_hold.py`, staged experiment runner and tests. Keep existing profile files unchanged; route only the explicit stage-A option.

- [x] RED/GREEN: published pressure preparation prefix, fixed-wrist A transition, right-only recovery, actual-object-bound targets, fixed left finger targets and pinch bypass.
- [x] RED/GREEN: candidate nonlinear/complete-collision rejection, per-pad transient retry without permanent solver lock, fresh pressure reference after actual contact. Prior pressure action is revoked when the next physical substep cannot rebuild its model; blocked mandatory relief aborts rather than silently holding.
- [x] RED/GREEN: A/B/C/D diagnostic pass conditions and failure stop; C contact holding vs attested weight support; explicit A entry returns before PPO/Student construction and rejects later stages.
- [x] Implement scene/assets/current-state verification and first-terminal preservation before native launch. Native01 exposed a Table-Xform/child-collider readback bug; a read-only descendant-union fix passes3 focused+2 independent USD cases without changing physical bounds or USD bytes.
- [ ] Implement native B/C/D controllers and their serial stage hand-offs after A recovery evidence passes independent audit. Diagnostic gates alone do not complete these controllers.

## Task 6: Serial native validation and publication

- [x] Freeze candidate sources/argv/assets/models; obtain CPU regressions and independent code review before A. All01 failure artifacts and13 protected originals remain unchanged;02 has exclusive output, launch gate and1331-file source snapshot.
- [x] Hold the single-instance lock and attempt the independent A cold-reset experiment with pinch control; preserve every available actual substep/action/contact/limit/geometry budget and first terminal. A02 ended by operator SIGINT at481 logical controls/482 physical intervals/3856 raw SDK substeps;8 first episodes censored,0 complete,0 approach submissions and0 budget consumed. This is not an A qualification or a 0/4 trial.
- [x] Independently reconcile A02 and report whether right-hand recovery was tested: audit_integrity=false with25 issues (missing final diagnostic rows and final FK SDK joints among them), so no A pass/fail or recovery claim. Only a future complete A audit can permit B; only B can permit C; only C can permit D.
- [x] Run frozen CPU04 and compare every baseline node:267 new pass, all5786 old statuses unchanged; original96 failures/28 errors remain. Final scene/logging18 targeted cases pass separately.
- [x] Finish CPU05 on the exact A02 frozen source and recheck baseline/source identities:5883 passed,270 new all passed, all5786 old statuses unchanged;1331 frozen files unchanged after CPU. XML SHA256:87d83ad90af0e39146f68cda5684771c11ed990844f5dd3ff850f86d6bd2ecbe.
- [x] Rerun the full CPU suite after the owner-first cooked-readback and latest stage additions:5917 passed,304 new all passed, all5786 old statuses unchanged,6090 nodes, final1334-file source manifest unchanged. XML SHA256:fc518fdbe4e29761db99cc61933d0c4ebfbc583cfc003724ff583f1116cc3440. The targeted owner-first set remains52 passed in13.34 s.
- [x] Update README, CHANGELOG and native report with implementation, CPU, actual stage results, censored A02 state, unrun stages and unresolved physical limitations. Preserve all artifacts outside Git.
- [ ] Commit/push the existing branch; verify remote SHA and unchanged11-5 base, Draft and unmerged status. Do not train Student or promote PR without native evidence.
