# Integrated grasp control validation — 2026-10-26

The preserved iteration583 checkpoint is unchanged. v154 resumed4096-environment PPO from that checkpoint with the validated control transport. Training admission is separated from grasp success; no stable grasp success is claimed.

## Repairs under test

- Separate opt-in coordinated controller: measured finger motion drives palm compensation; pressure saturation stops positive integration; index recovery can retreat0.03rad below the policy envelope without crossing mechanical targets. Thumb near-limit release coordinates the opposing index. Existing physical guards remain authoritative.
- Fixed-root teleport transport was inconsistent: during actual commanded lift, base position finite differences were about0.008–0.010m/s while reported base velocity was0. A moving base needs physically represented motion; this was verified in saved v142/v143 traces.
- GPU `set_kinematic_targets` is unavailable in installed Isaac5.1 (v144 runtime error). The final diagnostic uses three driven prismatic joints inside each hand articulation. The original DH116 base becomes a moving link, and the wrist anchor stays fixed. Finger active DOF count/action space remain unchanged.
- Wrist drives: stiffness100000N/m, damping1000N·s/m, force cap100N, speed cap0.03m/s per local axis. These are simulation wrist support assumptions, not manufacturer finger force calibration. Original finger torque0.2Nm and contact material remain unchanged.
- Explicit USD linear DriveAPI is required in the new asset. Merely writing stiffness/damping/force values through the tensor interface gave poor tracking. Runtime drive declaration is now tested, along with coordinate conversion, readback and removal of the old world attachment.
- The carriage measures actual palm offset and checks tracking error≤3mm and palm speed≤0.06m/s. The CLI requires a control probe or coordinated tensor training for this transport. Startup and normal tensor steps share the same transport guard; actual base-link observations replace fixed-anchor observations.

## Recorded results

| Run | Result |
|---|---|
| v142 coordinated target control | Rejected at6.0625s: right thumb joint limit guard. |
| v143 index-only retreat and opposing release | Rejected at5.2875s: right thumb joint limit guard. |
| v144 separate kinematic body | Startup rejected: GPU kinematic-target setter not implemented. |
| v145 carriage startup | Strict readback rejected added joints; now verifies all eleven hand joints plus all three carriage joints separately, including units. |
| v146/v147/v148 carriage without USD DriveAPI | Contact test rejected by original velocity guard; exact source snapshots preserved. |
| v150 free carriage, without USD DriveAPI | Failed motion qualification; target30mm, large tracking error. |
| v151 free carriage, with USD DriveAPI | All16 hands passed: actual travel29.9mm, final velocity0.010m/s, maximum tracking error0.104mm. This is an empty-hand test, not grasp success. |
| v152 complete grasp with driven carriage | 8/8 completed, no fatal physical fault, success0, early3mm pass0; maximum actual clearance0.531mm, peak upward contact force1.410N. |
| v153 final startup/lead-governor review fixes | 8/8 completed, no fatal physical fault, success0, early3mm pass0; maximum clearance0.531mm. Candidate lift remains gated by coupling, insufficient load, pressure and slip. |

Evidence directories are under `runs/tabletop_rigid_20261026_v*/`; source snapshots and launch manifests are under `runs/control_probe_20261026_v135_preserved/`. `wrist_motion_comparison.json` contains the per-hand empty-hand comparison. Physics trace records actual palm velocity/position, carriage joint targets/positions, and PD effort estimates; PD estimates are not measured solver drive forces.

API reference: [PhysX articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html). Installed backend availability was tested locally; public API presence alone was insufficient.

## Final review and regression

- Policy-envelope floor no longer overrides measured position lead when external load displaces the actual joint below that floor. Mechanical bounds retain priority.
- The near-limit thumb preserves its previous request subject to the measured lead governor. A stale request at q+0.02 may decrease toward q+0.008: this retains a closing position bias, not a guarantee of net closing torque. Slew remains0.05rad/s. Opposing-index release and palm/lift pause remain active.
- Reset affects selected environments only; host startup preflight now uses driven carriage and actual base-link measurements, as tensor stepping does.
- Exact one-way checkpoint migration adds only coordinated-controller and driven-carriage contracts. Policy, Adam and counters are retained; unrelated changes or reverse migration are rejected.
- Current CPU regression:105 passed,2 skipped (CUDA unavailable by test configuration), using `CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest tests/test_dh116_coordinated_grasp.py tests/test_dh116_palm_mount.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py tests/test_dh116_load_parameters.py tests/test_dh116_solver_iterations.py tests/test_dh116_root_pose_writes.py -q`. Shell syntax and diff whitespace checks pass.
- `scripts/train_dh116_tensor_ppo.sh` now selects coordinated control and driven carriage. Rewards, finger effort and hard physical protection were not changed. Physical qualification is limited to the recorded bounded probes, not an all-policy guarantee.
- New training source hashes and launch metadata: `runs/control_probe_20261026_v135_preserved/v154_training_source/sha256.json` and `v154_launch.json`. Original checkpoint SHA256: `ebea11d21c32c475866d73ed4bc05348c85f3e821fc677ee481079043054bb41`.

## Production resume verified

`training_start.json` confirms initial iteration583, timesteps57311232 and13 Adam state entries; migration lists only `measured_motion_grasp_v1` and `driven_wrist_carriage_v1`. Actual monitor reached iteration586 with 57606144 transitions, success0. New run: `runs/tabletop_rigid_20261026_v154_coordinated4096`; TensorBoard live link: `runs/tensorboard_live/v154_coordinated4096`. Background observer reports problems and50-update trend windows; it does not edit or restart the trainer.

Batch reset verification: at iteration599, completed episodes increased from154753 to158763. Failure accounting: {"mimic_error": 4, "grasp_not_established": 4006}. Training continued after selected resets without a fatal stop. These are failed learning episodes, not grasp success.
