# DH116 T2 implementation and first native smoke

This is a progress report, not a completed lift-regression or reward-ablation report. PR #1 targets `11-5` and remains Draft. No merge or readiness change is authorized.

## Original run and immutable models

The user initially selected waiting for the original run, then explicitly authorized its exit: “原进程可退出”. At 2026-11-06 08:32:36 UTC, the operator copied and validated the last complete visual v2 checkpoint before sending SIGINT to the verified PID150108. The process exited; GPU3 returned to9MiB with no compute process. The old monitoring automation was deleted after this intentional exit.

The preserved original completed1750 PPO updates,44897 episodes and0 native successes. Its new immutable copy has SHA256 `837791335e5f1c2153db763ed419381c708ec8b1f8ff2d4abfc6fa025aab249d`. It did not complete2048. Original files,1439/129 source weights and physical parameters were retained. Derived native tests use independent directories and one Isaac instance at a time.

Actual129 and1439 visual checkpoint contracts were compared recursively and are identical, including1024 training environments and224×96 fixed cameras. The269/482 historical lift peaks are log entries; corresponding weights remain unavailable and will not be substituted.

## Implemented infrastructure

- Evaluation-only v2/v3 deterministic actor loading, actual authored reset installation/readback and first-terminal physical telemetry. Production manifests require100 unique physical initial states per grasp. Startup, synthetic fixtures and repeat episodes cannot count as native trials.
- Invalid slip remains null with coverage/reasons. Applicability is cross-checked against raw prior-anchor/contact/measurement facts. Complete12-joint torque estimates/limits and24 actions are required; simulation torque estimates are not real hardware feedback.
- Evaluation checkpoint identity is checked at preparation, inference loading and completion. Native evidence binds manifest, raw reset state, RGB-D geometry, actor input and compressed pre-reset substep streams. A writable report flag does not certify native execution.
- Explicit1024→small-experiment checkpoint projection preserves full model/Adam/LR/Torch RNG and source counters. Exact training resume still requires the complete v3 contract.
- Completed v3 checkpoints are retained immutably at the configured interval and final update. Independent native validation selects models with distinct `best_validation` and `best_success_rate`; zero successes cannot produce a best-success model.
- Explicit preserve/reset-Adam warm starts. Reset clears only new-process Adam state, records source SHA and iteration, and requires a subsequent complete update. Actual loaded checkpoint budget is rechecked before metadata/reset/camera. Projection/reset/reward provenance survives subsequent saves.
- Separate R0–R5 reward metadata/derived checkpoints, experiment-v2: R0 unchanged; R1 adds qualified height hold, R2 adds uncredited qualified upward progress, R3 adds descending-segment penalty, R4 combines independently supported hold/up, R5 combines validated factors. Positive terms require actual valid contact, observable slip, sufficient signed support and measured low velocity. Positive terms use completed control windows; DROP has an independent live physical-substep ledger with actual dt, including mid-window terminals. Invalid live samples break continuity; inactive reset tails cannot erase or charge terminal motion. Reset references use the actual physical-reset sample. Any provisional experiment-v1 R1–R5 model must be explicitly re-derived.
- Physics identity canonicalizes only known per-grasp batch repetition and the explicitly versioned new reward metadata. Full source/execution critic SHA remains recorded. Camera, original base rewards, materials, geometry, controllers, limits and30mm/5s/0.5mm success remain bound.

Actor3844D / privileged Critic371D /24D physical action contract are unchanged. No unverified tactile/contact/slip feedback enters a real Student interface.

## Native smoke actually executed

The1439 v3 derivative was explicitly projected to16 environments,8 per grasp, with source SHA `002812709ac5e6b9a83ef64c66f4c60f205945e90edcb9afdc243ee5ddae4477`. A separate662-file execution snapshot was frozen before native launch. The original224×96 camera and physics/control settings were used, including the original renderer view limit.

| Native stage | Complete PPO updates | Timesteps after stage | Adam steps added from1439 | Update wall seconds |
|---|---|---:|---:|---|
| First process |1440,1441|47645696|80|24.915,28.194|
| Restored process |1442|47646208|120|25.697|

Both native processes exited0 and saved complete checkpoints plus immutable bundles. Initial RGB-D image/object validity was100%; actual geometry audits passed. Maximum frozen log-probability reconstruction error was `3.337860107421875e-6`. Actual commands passed the runtime bounded24D checks. Projection provenance was preserved. Native successes remained0.

A third native process loaded1441 at an already-complete budget and saved into another new directory. Independent recursive comparison found exact equality of policy, Adam, LR, Torch CPU/CUDA RNG, iteration, timesteps, episodes, successes, complete contract, action derivation and experiment projection. This verifies actual native restoration without attributing any extra PPO update or success.

Raw execution evidence and model files stay outside Git under `/mnt/mydisk/xk/dh116/runs/lift_repair_native_20261106T0834Z/`. The derived training smoke is not part of the paired historical-policy evaluation denominator.

## CPU verification and limitations

Test-first changes include terminal-before-reset, diagnostics-anchor selection, conditional slip observability, snapshot races, source replacement, complete torques/actions, reset-Adam budget races and physical identity. Independent reviewers found real defects, which were repaired before publication. Final reward delta review independently reproduced terminal substeps1/2/7/8: each receives−0.001041667 at its live terminal sample and0 throughout the inactive tail;50 focused tests passed. Warm-start/retention/reward-fork review passed73 related tests. The real-case physics-identity delta also passed independent specification and quality review.

Executed focused commands used `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `OMP_NUM_THREADS=2` and `/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider` in the isolated stage. Recorded runs include133 related tests, the final202 reward/window/reset tests with1 skip,102 identity/evaluation/selection tests and25 warm-start/reward-fork tests. They overlap and must not be summed as unique coverage.

Final full CPU regression repeated the original baseline command, including its exact `tests` scope and USD/DeformX module paths:

```bash
dh116_usd_root=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$dh116_usd_root:src:.:vendor/DeformX" \
 LD_LIBRARY_PATH="$dh116_usd_root/bin:${LD_LIBRARY_PATH:-}" \
 PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
 tests --continue-on-collection-errors -q --tb=short \
 --junitxml=/tmp/dh116-stage2-full-exact.xml
```

Actual result:5101 passed,96 failed,49 skipped,28 errors and91 subtests passed,267.28s,exit1. The known old failures/errors remain; this is not an all-green repository. JUnit comparison preserved all4801 baseline nodes with identical status and all5079 prior-P0 nodes with identical status. All473 nodes added since baseline passed; all195 added in this stage passed. An earlier broader collection incorrectly included vendor/scripts tests and lacked the baseline vendor module path; its131 errors are not used as the baseline comparison. No old tests were removed or altered to hide failures.

All1207 original source/resource files were rehashed after the authorized exit: zero mismatches. Models/raw logs remain outside Git.

The native smoke passes the action/RGB-D/PPO/save/restore gate only. Per-grasp100-case statistics for129 versus1439, cold-reset native certification, R0–R5 ablation outcomes, preserve/reset-Adam effectiveness and independent validation selection are still pending. Authored validation/test manifests have been frozen with distinct seeds,100 cases per grasp, and remain unverified until actual execution/readback. There is no evidence yet that lift degradation or reliable grasping has been solved. Success remains30mm,5seconds and0.5mm slip.
