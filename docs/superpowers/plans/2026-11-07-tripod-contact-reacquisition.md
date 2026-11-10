# Tripod contact reacquisition implementation plan

> **For agentic workers:** Use subagent-driven development with test-first implementation and independent evidence review. Root owns the only native Isaac instance and the single-instance lock.

**Goal:** Test whether bounded geometric reacquisition can restore the missing tripod contacts without changing the original simulation or Student/PPO contracts.

**Architecture:** Add an explicit experimental Teacher-only profile beside the unchanged pressure-reference controller. Existing bearing pads keep their pressure targets. Missing thumb/index/middle pads receive separately identified CAD-to-cylinder geometric targets only when their actual orientation, distance and modeled motion permit approach. Preserve the exact Teacher wrist actions in this first single-factor experiment. Do not turn a geometric prediction into a measured contact or a successful demonstration.

**Tech stack:** CPU PyTorch regression tests; source USD pad atlas and actual native body poses/Jacobians; original 480-Hz physics, 60-Hz canonical 24D commands and fixed real RGB-D. Existing 3844D Actor, 371D Critic, Tanh Gaussian and default v2 remain unchanged.

## Evidence and controls

- Published baseline: `0fce5e9f6f4d9bd90dd3e13522d8c56e0d806389`.
- Original pressure replay loses right-hand opposition at controls339..409. From control437 onward, all reference finger velocities are zero while both wrists continue rising at3 mm/s. Missing pads have zero geometric weight and cannot be reacquired by the existing controller.
- A wrist-only gate after437 cannot restore contacts lost earlier. It is not the first proposed fix.
- Independent stage: `/mnt/mydisk/xk/projects/dh116-tripod-20261107`.
- Independent outputs: `/mnt/mydisk/xk/dh116/runs/tripod_contact_20261107T0400Z`.
- Baseline targeted CPU run:90 passed,11.89 seconds, `baseline_cpu.xml` in that output directory.
- Preserve all original checkpoints, output directories, physical parameters, actuator bounds, reset identities and net30-mm / continuous5-s / drift0.5-mm native success criteria. No long PPO or new reset candidate is authorized by this experiment.

## Tasks

- [x] Save a read-only failure timeline from pre-terminal480-Hz samples, separating source exhaustion, missing contacts, wrist motion and actual object dropping.
- [x] Write failing tests for a pure geometric target helper: actual source-CAD point/normal, finite-cylinder side boundary, inward sign, wrong-facing pad rejection, missing/invalid data, per-pad isolation and bounded approach speed.
- [x] Implement only those geometric target rules; label every output `geometry_only` and never mutate measured pad loads/validity/points.
- [x] Write failing adapter tests for missing-contact weights, original loaded-pad pressure behavior, exact wrist preservation, finite per-digit acquisition displacement/time budget, invalid/rejected contact fallback, native command/lead/rate limits and first-terminal exclusion.
- [x] Implement a separate opt-in adapter/profile, factory and CLI routing. Historical profiles and source qualification stay unchanged.
- [x] Reproduce candidate predictions on recorded native snapshots. Explicitly report geometrically ineligible or locally unreachable fingers; do not enlarge limits to manufacture feasibility.
- [x] Run focused regressions and independent code/constraint review. Freeze all candidate runtime source hashes before native launch.
- [x] Run one bounded both-style native replay with original initial state, cameras, source and physics. Pinch is a control; report tripod pads, opposition, losses, force/depth, hold, net Lift and original success independently.
- [x] Independently replay raw physical evidence, synchronize actual issued actions/RGB-D and distinguish incomplete trials, native execution and success. Failed trajectories cannot enter successful Student supervision.
- [x] Use the result to decide the next single factor. If12-finger geometry cannot improve the missing pad's distance/orientation, record the failure before considering a separately identified24D wrist-aware solve or readiness controller. Never bundle reward changes.
- [x] Update README, CHANGELOG and validation report with actual results and unresolved limits before publishing the existing Draft branch. Do not merge or mark Ready.

## Acceptance boundaries

CPU tests and lower modeled error do not establish grasping. Native pad establishment must come from original object-filtered PhysX contact with the audited CAD/style gates. Native success must satisfy the unchanged observer. Four fixed replay environments do not replace100 independent per-style trials or an unseen validation set. A successful Teacher/MPC episode is not a successful Student episode.

## Evidence-driven second factor: constrained geometry planning

Sequential replay of actual experiment02 decisions0..44 exactly reproduces all six first geometry stops. Four reject only a future step, two reject the first step. Independent witnesses satisfy the same wrist motion, point Jacobian, response, joint/lead/slew bounds and exact cylinder-gap/facing/side checks in all six cases. Thus a rejected soft-cost candidate does not establish geometric infeasibility. The next factor changes only the solver, keeping original wrist timing, objectives/weights, pressure rules and cumulative budgets. It does not add retries or relax the existing exact admission.

- [x] Add pure `tabletop_geometric_constrained_mpc.py` and its test-first regression suite. Preserve the original loaded kinematic recurrence, four-step horizon and cost. Put positive-gap/improvement/facing/side and overloaded-pad direction constraints inside a bounded small Teacher-only solve. No-geometry calls retain the exact old planner; unsuccessful constrained solves return the old candidate for existing admission/fallback.
- [x] Add separate `tabletop_vision_teacher_constrained_mpc.py`, exporting `ConstrainedAcquisitionTeacherReferenceMPC` with the same explicit geometry/source arguments. Keep the existing18c acquisition module unchanged. Preserve per-pad cumulative limits and permanent-stop lifecycle for this solver-only comparison.
- [x] Register opt-in profile `pressure_geometry_constrained` in native CLI/factory. Observe the new route/validation tests fail before implementation; preserve every old profile's behavior.
- [x] Independently reproduce exact constraints and original cost on all six real contexts. Bind final source hashes and test results before a second serialized native experiment.
- [x] Complete and audit experiment02 first. Execute the new profile in a fresh directory only after its CPU/evidence gate passes and the Isaac lock is free; use unchanged physical/reset/camera/source configuration. Report feasible planning, actual contact and native success separately.
