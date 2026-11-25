# Changelog

## 2048-environment Teacher grasp training — 2026-11-25

- Score newly achieved per-episode grasp progress using measured contact, support transfer, detached lift and continuous hold. Remove the time-proportional reward for maintaining a stationary grasp on the supports.
- Keep commanded world wrist poses consistent with issued mount targets when local travel clips the request. Reapply the original table clearance in the mount frame.
- Validate all three grasps in a 2048-environment headless Isaac probe with natural exit 0. Rebuild the bounded flat CUDA contact allocation for 2048 environments and retain the exact pair export and reset checks.

## 1024-environment native PPO — 2026-11-25

- Support 1024 native rows through scene naming, batch articulation, contact capture, shared observations, Teacher training and frozen actor loading. Keep lane names ordered at four digits for the 1024-row scene.
- Rebuild the SM89 ropesim extension with CUDA 12.8. Size the bounded contact buffer for 1024 lanes with 41 nodes, 17 meshes and 12 solver substeps. Flat contact indices and vectors match the legacy batched export exactly in the candidate CUDA build.
- Pass 231 CPU tests and 15 tests in the CUDA session. Pinch, Tripod and Power each complete two native control windows with 1024 environments and exit naturally. The second windows take 0.56, 0.60 and 0.50 seconds respectively.
- Launch 1024 updates per grasp with separate physical GPU locks and checkpoint directories. Retain the verified 64-update checkpoint from the cancelled 512-environment Power run.

## Power PPO learning rate comparison — 2026-11-25

- Lower only the Power full-grasp PPO learning rate from 0.0001 to 0.00003. Log the sampled KL measured on the next minibatch after the first optimizer step, using the existing evaluation.
- Compare two native 512-environment, 16-update runs with the same scene, reset seed and policy seed. Both exit naturally. Median first-step sampled KL drops from 0.061435 to 0.000589. The lower rate applies all 12 minibatches across three epochs in 13.10 s per update at the median. The baseline applied one minibatch per update and took 13.34 s at the median.
- The lower learning rate resolves early stopping in this comparison. Both Power runs report zero measured bilateral support and zero qualifying Lift. The preceding opposed-pad reward comparison also finishes with zero qualifying Lift in all three grasps.
- Pass 228 CPU tests, 15 tests in the CUDA session and a two-window native Power probe at 512 environments.

## Opposed pad approach shaping — 2026-11-25

- Estimate near-surface approach directions from the original trilinear pad and palm grids, on the correct material and world half. Use the existing grasp-specific opposition geometry to distinguish equally close, same-facing and opposed pads. Near-surface guidance decays to zero one rod radius away from the surface.
- Continue to derive contact, load, detachment, Lift, Hold and Slip qualification from the native physical state and force readout.
- Complete the previous 512-environment run set, with 64 Pinch, 32 Tripod and 16 Power updates and natural exit 0. Peak contact rates were 22.42%, 63.93% and 0% respectively. No grasp reached qualifying detached Lift. Each Power update stopped after its first minibatch.
- Pass 228 CPU tests and 15 tests in the CUDA session, including both devices for the new regression. All three grasps pass two native control windows at 512 environments and exit naturally.

## Continuous pre-contact distances — 2026-11-25

- Extend approach distances from the locked SDF boundary. Keep signed in-lattice gaps and their masks for the original depth guards. Nonfinite coordinates do not index the grid or count as approach.
- Replay the same 0.2 mm crossing against six locked pad and palm grids. Changes in the approach distance stay below 0.2 mm after the fix. Keep the original CUDA binary and update the verified source manifest.
- Complete R7's 512-environment PPO comparisons. Pinch reaches brief bilateral opposition. None of the three grasps reaches qualified detached lift.
- Pass 227 CPU tests, 13 CUDA tests and naturally exiting 512-environment native probes for the corrected recipe.

## Measured opposition and bilateral lift reward — 2026-11-25

- Use simultaneous load-weighted normal directions for continuous Pinch and Tripod opposition. Separate Power finger and palm opposition. Preserve the existing discrete contact checks.
- Shape prelift from the weaker hand's signed upward load and actual support unloading. Ramp the height factor out below the reset height and give zero potential below −30 mm. Preserve the strict total-load lift signal and detached Lift/Hold/Slip qualification.
- Finish the prior 512-environment PPO comparison for all three grasps with natural exit 0. None attains detached Lift. Tripod prelift reaches 1.248 without measured bilateral opposition.
- Pass 226 CPU tests, 13 CUDA tests and naturally exiting 512-environment probes for the new reward.

## Support-to-hand lift transfer reward — 2026-11-25

- Shape early progress using measured upward hand load, measured unloading of the supports and the less-loaded hand. Keep the lower of median and minimum rod displacement as the lift height.
- Keep strict contact, detached lift, Hold and Slip qualification. Detect invalid table, non-target and excessive reaction cases in the reward regression.
- Pass 224 CPU tests, 13 CUDA tests and naturally exiting 512-environment probes for all three grasps. Train this recipe as a candidate. Its prelift signal can still reward asymmetric support transfer or a catch at a low height.

## Balanced grasp pressure — 2026-11-25

- Shape approach progress toward the least-loaded required finger and the less-loaded hand. Preserve the force, opposing contact, support, lift, hold and slip criteria.
- Pass 224 CPU tests and 13 CUDA tests. Each grasp completes two native control windows with 512 environments and natural exit 0.
- A paired diagnostic sustains Pinch bilateral contact for 30 fixed-wrist windows. The lifted comparison loses bilateral contact in its first lift window. Tripod sustains contact for at most one window. No lift qualifies.
- Read eight live Power middle-link PhysX hulls. Both finger52 controls reproduce the locked SDF values exactly. The six new candidate SDFs and the pose fit remain offline evidence.

## Anatomical grasp scenes and first-person stereo — 2026-11-25

- Rebuild Pinch, Tripod and Power resets and references with the left wrist at −Y, right wrist at +Y, and thumbs inward. Verify actual PhysX link positions at startup and after a local Reset.
- Set the fixed, rectified stereo rig 0.32 m behind and 0.42 m above the wire center. Verify 12 same-clock native captures with all 41 material nodes in each eye's field of view.
- Gate Hold on target-only load and raw hand reaction below 40 N. Isolate nonfinite post-terminal samples from reward and observation calculations on active lanes.
- Pass 223 CPU and 13 CUDA tests. Each grasp passes the two-window 2- and 512-environment probes and two PPO updates on 2 environments, all with observed exit 0.
- All three stereo previews save their frames and return from App.close before exiting with SIGSEGV. Preserve the captures with diagnostic status.

## Faster Teacher control and native stereo — 2026-11-25

- Bind the 0.05 m/s wrist, 1.0 rad/s wrist rotation and 1.2 rad/s active joint commands to the shared action contract and verify all PhysX speed limits at startup.
- Apply continuous costs for mild guard excursions and log individual terminal causes. Keep measured bilateral load, detached lift, hold and slip as success conditions.
- Reuse native CUDA contact reductions, use FP32 pad sampling, defer reward validation to window settlement and apply minibatch KL early stopping. Save checkpoints at intervals, best updates and completion.
- Replay a source-bound frozen Actor in the supported cable scene and capture fixed stereo plus an overview at one measured simulation boundary.

## Teacher grasp reward — 2026-11-25

- Add bounded light-contact and assisted-lift potential from owned DeformX contact pairs for Pinch, Tripod and Power.
- Separate signed vertical loads from strict external contact magnitudes. Preserve geometry guards, 24D actions and the fully detached 30 mm Lift, 5 s Hold and 0.5 mm Slip criteria.
- Default full-grasp episodes to 720 control windows and retain the 32-window PPO rollout. Record arrival, light contact, supported height and prelift progress in each update.
- Verify 195 CPU and 13 CUDA tests. The three native 512-environment, two-window probes exit naturally with code 0 and perform zero PPO updates.

## SCENE-INIT-SPEEDUP R1 — 2026-11-24

- Clone one fully authored/validated USD template into the shared Teacher/Student batch; retain the independent reference builder for numerical comparisons.
- Preserve SDK tuple-to-float32 hand authoring and double-precision table/support authoring with identity environment transforms. Keep float64 origins, unchanged physical assets, material values, 480/60Hz, joint and 1µm guards.
- Replace quadratic collision exclusions with Cloner inverse groups, preserving external global colliders and rejecting pre-existing group reinterpretation.
- Reduce measured 128-lane preparation from205.716s to6.658s. Real N2/N128 fixed-action two-window comparisons preserve exact state/rod/reward/guard values and natural exit0; float64 force reductions retain only ~1e-15 roundoff.
- Pass182 regressions (169 CPU/13 CUDA). PhysX replication trial failed with outer-6 after close (GIL finalization); retain failure evidence, block training for that mode, no retry or forced termination. New changes remain local, not pushed.

## ini — 2026-11-24

- Consolidated the production runtime into src/dh116:24 implementation modules and6 namespace initializers. Removed retired code,26 historical tools, legacy package hierarchy, reverse script imports and obsolete path rules.
- Kept6 entry scripts and15 focused test files. Preserved assets/resources/vendor, historical models and raw-data recovery copies outside the live project.
- Fixed state-only heldout selection, exact frozen-Teacher label binding and same-pre-action batch completeness. Kept all numerical policy/PPO/reward/geometry baselines and physical limits.
- Restored the original two-target-finger Pinch SDF role while keeping five force routes and non-target-finger guards.
- Rebuilt local CUDA12.8 ropesim/FP64 geometry candidates. CPU166 passed; CUDA geometry13 passed. Real two-lane flat ABI comparison passed7 solver steps with exact indices/vectors.
- Final native128-env diagnostic passed2 control windows/18 SDK steps,0 PPO updates, normal App.close and externally observed exit0. Initialization wait is separately bounded; no forced exit was used.
- Replaced two stale hollow-straw generator paths with verified immutable Git blob references; physical asset bytes remain unchanged.
- Native windows:0.44650s/0.28645s. This is a bounded diagnostic, not grasp/Lift/Hold qualification.
- Failure evidence, source hashes and recovery state are retained in the external archive; generated logs/models/data are excluded from Git.
