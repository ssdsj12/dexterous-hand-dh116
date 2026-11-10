# Tripod recovery and nonlinear contact design

Base: `46dc091738023503d21da980c944afbceef986d7`, existing isolated worktree on `codex/dh116-lift-repair-20261106`. The user's ordered P0/P1 requirements and subsequent explicit episode-budget clarification define this implementation. Original physics, source assets, actuator limits, 3844D Actor / 371D Critic / 24D actions, Tanh Gaussian, v2 behavior and net 30 mm / continuous 5 s / 0.5 mm success criteria remain unchanged. No Student training before native A–D gates pass. PR remains Draft.

## Separate opt-in path

Keep every published Teacher profile, especially successful pinch and historical permanent-code4 behavior, unchanged. Add independent recovery, nonlinear geometry, diagnostics and staged experiment components. Existing default call paths do not create these components. The original runtime observer remains authoritative for native success; A/B/C diagnostic gates cannot create successful lift supervision.

The approach is a measured per-pad recovery controller with full candidate geometry, rather than another fixed-normal Jacobian solver or increased PPO budget. Wrist optimization is deferred until the finger-only model has explicit evidence of insufficiency. Common wrist ascent in D is a separately recorded cooperation gate, not a new unrestricted wrist optimizer.

## Contact state and conserved budgets

Use owned `[env,hand,digit]` arrays, initially enabling only right thumb/index/middle for A. Actual object-filtered pad loads above the original 0.05 N threshold establish CONTACT. Invalid measurements mean unknown, not a synthetic contact loss. STABLE additionally requires continuous valid normal/depth/rejected-contact gates for the existing 20 ms recovery dwell (10 samples at 480 Hz); that dwell is not the 5-second grasp success standard. CONTACT immediately resumes pressure feedback; STABLE stops acquisition charging. Loss leads to LOST and bounded REACQUIRING, with explicit transient blocked reasons rather than permanent solver-failure latches.

Per physical episode and per pad: 1 mm cumulative executed approach command, 1 s cumulative REACQUIRING simulation time, 0.5 s without measured geometric progress. No loss, contact flicker, temporary CONTACT, stable recovery, optimizer failure or retry replenishes either budget. Distance/time exhaustion locks automatic acquisition until genuine episode reset. A no-progress round stops; any later round requires a new measured condition and remaining budget, never a tick-based watchdog reset.

Only a native-guard-approved approach command spends distance. Preserve raw requested action, post-guard target/action, execution identity, executed approach command distance and measured physical progress as separate fields. Geometric planning success and arbitrary nonzero finger actions are not execution evidence. Snapshot episode identity and monotonically increasing clocks make duplicate execution commits idempotent and stale/reset packets rejectable. The reservation before execution must prevent exceeding the remaining cap; the post-guard record commits only what was authorized. Relative measured patch/object motion supplies progress, not wrist command motion or predicted progress.

## Nonlinear source geometry

Load both independent hand USDs with layer hashes, metre/unit and collision-mesh audits; retain all 17 hulls per hand and the original five independent distal pad regions. A pad region consists of actual supporting facets clipped by the original central ellipse and all source hull halfspaces. A finite set of seeds is permitted for optimization initialization, never as the final region certificate. Distance, normal, sidewall and source-region residuals must refer to the same witness point/facet.

Bind every planning snapshot to actual named SDK full joint positions/velocities, all link poses and the measured base_link pose. Preserve follower offsets. For each edge freeze `B = inverse(T_parent_actual) @ T_child_actual`, and predict increments by `T_child_candidate = T_parent_candidate @ t0 @ R_axis(delta_full) @ inverse(t0) @ B`; fixed edges use `T_parent_candidate @ B`. Full candidate angles are actual full angles plus the existing named mimic increment matrix times active increments. Zero increment must reproduce every measured link, and ideal states must match independent USD FK. This is calibrated short-horizon kinematics, not a dynamics prediction.

Every candidate and encoded action is checked with recalculated nonlinear link/pad poses and normals. Audit full active/follower limits, measured lead and slew separately from region distance, orientation, full-hull overlap and obstacle clearance. Include complete rod segments, table, original supports, self pairs with existing exclusions, and all interhand pairs. Conservative separating witnesses can certify clearance; an absent witness is UNCERTIFIED and rejected, never silently accepted through sampled-point penetration. Explicit fixed thresholds retain their source identities; do not mix static-proposal defaults with native limits. Check actual physics substep trajectory candidates and bound unsampled rotation/translation or reject uncertain sweep coverage.

## Native diagnostics and cooperation

Add an opt-in pre-reset observer and post-guard execution hook. Record each 480-Hz live substep before normal reset; add an error snapshot/flush path for fatal guards. Keep simulation step/time, TensorWorld step, control/substep, env, physical episode, stage and execution IDs distinct. Record actual pad normal loads/vectors/points/depth, raw contact and friction buffers with ownership, six-pad continuity, opposition, hand normal/friction vertical force, rejected loads, measured palm xyz/rotation and reset-relative motion, original/actual drive limits, feedforward commands, simulated applied/computed torque and saturation coverage, actual net lowest-point Lift and anchored object-in-hand drift with validity. Missing fields remain null/unknown. No simulated estimate is hardware feedback.

Raw friction and supporting-surface contact coverage must be explicit. Six compressed pads alone do not attest weight support. C may report contact holding while the object remains supported; unsupported weight-bearing requires native hand-force/weight evidence and original table/support unloading evidence. Do not infer these from absence of a hand contact or a geometric friction cone.

Allocate gravitational wrench from measured grasp points, COM, directions and existing capacities; do not default to 50/50. Infeasible equilibrium remains infeasible with residuals. D pauses both upward drives at the physical substep following detected loss, cancels accumulated upward targets against measured palms without teleporting physical state, and retains bounded finger recovery. Resume only after real bilateral support qualification. Pinch actions and targets are excluded from this logic.

## Native experiment order

All experiments use cold reset, original assets/materials/limits, independent output and frozen source, original first-terminal data and the existing single-Isaac lock. A fixed published pressure-reference preparation prefix may create the naturally observed right-hand loss; it is separately recorded and does not inject a state or consume new recovery motion. The A recovery interval freezes both wrist targets and permits only bounded right-finger recovery, keeping left pressure maintenance. No artificial contact loss, external force, attachment or added support is used.

1. **A:** Record genuine CONTACT→LOST→REACQUIRING→CONTACT→STABLE for each right required pad with fixed wrists and conserved budgets. Initial missing-pad establishment is reported separately; an unobserved loss is untested recovery. Audit measured wrist motion/drive lag rather than calling a zero command perfect fixation.
2. **B:** Only after A, fixed grasp locations and finger-only feedback must show actual simultaneous six-pad contact and both-hand opposition; distinguish compression from support of weight.
3. **C:** Only after B, fixed wrist/grasp pose must hold continuous qualified contact for 5 s with original 0.5 mm relative drift bound. Separately attest weight support/unloaded original supports; otherwise label contact holding only.
4. **D:** Only after C, allow coordinated upward transport, require unchanged native 30 mm/5 s/0.5 mm success plus stricter continuous six-pad evidence. Loss pauses common ascent.
5. **E:** Remains disabled until A–D pass. Teacher/MPC results never count as Student success.

If a stage fails, preserve its complete evidence and stop later native stages. CPU implementation of later gates is not a claim that their native experiment ran.
