# Selected Teacher seed before the first PhysX reset

## Actual repair
The frozen fast candidate source first called SimulationContext.reset() (two SDK setup steps), then applied the selected grasp roots and q. Its spawn_scene call also constructed one pair of assets whose return was discarded, before the candidate constructed a second pair of views at the same prims. These are source-level ordering/ownership findings; they are **not a proved unique cause of the earlier SIGSEGV**.

The new opt-in N=1 diagnostic now:
- reuses exactly the two original scene-owned Articulation assets;
- initializes the selected per-hand roots, original 11 joint states/coupling, and six zero wrist states before the first SDK reset;
- authors in-memory FK body transforms and PhysxSchema JointStateAPI state, keeping wxyz/XYZ conventions explicit;
- aligns only existing drive targets, preserving stiffness, damping, effort, velocity and position limits;
- compensates the retained builder's explicit 40mm spawn offset exactly once;
- reads actual SDK q/qd and base pose after the two setup steps and **before** legacy post-reset writes can conceal errors;
- checks all 17 DOFs, not just fingers. Any failed initialization receipt is saved and aborts motion.

The installed PhysxSchema generatedSchema.usda:1421–1465 states angular JointStateAPI position is in degrees and angular velocity in degrees/s. Solver/readback contracts stay radians and metres. No original USD file, joint axis/frame, material, physics parameter, historical checkpoint or old default control path was edited. The diagnostic contract is independently named force_contact_seeded_v2_no_ppo_no_success_admission; the frozen candidate source SHA remains f76d42c53bd7adb904f5d7e28741f83657fe688db7b5f64d66a436e60cecf9d6.

## Verification
TDD first showed four missing-initializer/wiring failures; later regressions caught missing wrist zero state/readback and a wrist drift that a correct base pose alone would hide. Actual original USD CPU tests check both hands, all 34 source body transforms, source joint bounds, passive coupling, nonidentity quaternions and nonzero environment origin. Invalid coupling is rejected before any stage writes.

CPU regression: **277 passed, 0 failed, 0 skipped**. The exact command, environment and raw log SHA are in the readiness file. A separate reviewer additionally measured a 4.44e-16 maximum world-FK matrix difference under nontrivial rotations/origin. CPU fixtures and USD authoring are not native initialization proof.

All 528 protected source hashes remain unchanged. Native runs, solver steps, Teacher queries and PPO updates in this work package: **zero**. No model or raw evidence is published.

## Remaining gates
The initializer has not been run natively. The next bounded proposal is two SDK setup steps with no control/solver/Teacher/PPO step, then actual 34-owner stored-cooked readback and normal shutdown verification. It must preserve NA for unavailable TensorWorld clocks. Cooked readback itself does not prove live shape attachment or grasp.

Pinch offline acquisition now has a separate geometric candidate, while Tripod remains blocked; neither is a successful Teacher model. Final success remains net Lift >=30mm, continuous Hold >=5s, fixed-anchor Slip <=0.5mm, bilateral opposed contact and no external support/upward force/attachment.
