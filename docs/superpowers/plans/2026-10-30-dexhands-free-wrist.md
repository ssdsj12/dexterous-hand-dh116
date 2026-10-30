# Free Wrist Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute bounded tasks with independent spec and quality reviews.

**Goal:** User-approved free finger/contact/palm orientation exploration and a measured rod-lift attempt inspired by DexterousHands.

**Architecture:** Opt-in 26D velocity-integrated policy, six driven wrist joints, measured geometric feedback and actual object-height rewards. Reuse continuous isolated recovery and preserve legacy20D entry points.

**Tech Stack:** Torch2.7, RSL3.1.3, IsaacSim4.5/PhysX, pytest, existing hand Conda environment.

## Task 1: Six-axis physical carriage

Worker owns tabletop_palm_mount.py, tabletop_vec_isaac.py, tabletop_load_parameters.py and new tests/test_free_wrist_mount.py.

- [x] Add failing tests for intrinsic XYZ quaternion composition, joint target/reset dimensions and angular drive readback; run only CPU tests on server.
- [x] Add ANGULAR_JOINT_NAMES and ROTATION_CONTRACT; author optional XYZ angular chain. `author_mount(...,rotation=False)` retains default three joints; `MountView.move(pose,angles=None)` submits six targets only for rotating mounts.
- [x] Configure separate angular actuator when args.free_wrist_grasp; validate real SDK drive and velocity readbacks for complete six-joint mounts, reject partial mounts.
- [x] Confirm green tests and provide source paths, API details and limitations. No Isaac or optimizer launched by workers.

## Task 2: Geometry and dense lift feedback

Worker owns new tabletop_dexhands_feedback.py and tests/test_dexhands_feedback.py.

- [x] Add failing tests that real FK distance improves near the finite cylinder, all four fingers matter, net height below0.5mm gives no lift credit, upward object with bilateral load scores, empty hand ascent does not, zero/invalid quaternion fails.
- [x] Implement `geometry_features(m,radius_m=.008,length_m=.5)` and `reward_rates(m,requested_force,actions=None,previous_actions=None)` as pure Torch functions. Document feature order, scales and maximum rates in CONTRACT. Return finite N-row terms with no hidden stage or fixed pose.
- [x] Use actual hand/grouped normal loads; do not interpret FK proximity as measured contact. Include bounded force tracking, balance and action-change costs. Main integrates dt scaling once.
- [x] Run red/green CPU tests and self-review. Do not modify other modules or upload production sources without main signal.

## Task 3: Main policy/world integration

Main owns tabletop_free_wrist_grasp.py (new), TensorTask/World/Training, policy control dispatch, CLI and a new launcher.

- [x] Write tests first for26D shape, targets integrate from current command, rotation is independently controllable, reset isolation and observations include controller state.
- [x] Add `--free-wrist-grasp`, requiring continuous free-grasp policy tensor pipeline. Keep staged/manual/old-checkpoint mixing rejected.
- [x] Integrate joint/palm/rotation velocities into bounded targets; use original actual joint bounds. Compose world rotation from seed quaternion and intrinsic XYZ. TensorWorld sends drive targets, measures pose/velocity/angular joints, never teleports during ordinary motion.
- [x] Extend task measurement keys and dynamic action size, observation and reward hooks; preserve invalid-state recovery and signed force rewards. Clear old lift item only in new mode.
- [x] Make RSL algorithm/storage action size explicit with20 default,26 when new flag; stamp contract and initialization. New launcher starts bounded experiment with explicit output and budget.

## Task 4: Physics attempt and delivery

- [x] Snapshot deployed files before replacement and compare normalized hashes. One Isaac instance at a time.
- [x] Run targeted combined regression, native rotation/reset/collector smoke and short bounded learning/search attempt.
- [x] Obtain spec then quality review; fix findings with targeted rechecks.
- [x] Record observed height/contact/slip/reward and policy update counts; do not conflate wrist motion with object lift.
- [x] Update README/CHANGELOG/report, commit locally on existing10-29 working copy and verify deployed source identity. Preserve failed-trial logs outside Git.

## Result

Implementation and bounded measured attempts completed. Actual rod lift remains unsuccessful: maximum0.133572mm, zero10mm holds and zero formal successes. See ../reports/2026-10-30-free-wrist-validation.md. Further DexGrasp-Anything ideas are research only.
