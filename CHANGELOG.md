# Changelog

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
