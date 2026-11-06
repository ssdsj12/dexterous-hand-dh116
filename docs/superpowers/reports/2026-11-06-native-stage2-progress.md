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

## Publication follow-up: physical initial identity and real checkpoint bundles

The optional training initial-state audit reads all native joint positions/velocities, hand/base poses/root velocities and object pose/velocity directly from simulator data. Clone origins are removed; repeated training templates are allowed but do not become independent evaluation trials. It checks SimulationContext time, step index and TensorWorld physical-step count before/after camera sampling. A saved-source reference must match policy, source file, seed, physics, camera and groups within2e-5 before PPO construction; reference SHA is pinned and rechecked. Fixtures cannot issue the native-only sidecar. The default path performs no sampling and preserves the model/observation/action contract. Cross-reward-fork source references are deliberately not generalized by this R0-only audit.

A real1439→128 CPU checkpoint-bundle probe exposed a serialization defect: `privileged_critic_contract.phase_force_reward.excluded_sources` is a tuple in the actual source, while JSON stores it as a list. Valid bundles were rejected by selection and duplicate retention. Only this exact known label tuple receives a JSON-manifest view; unknown tuples, physical/numeric/boolean types and the original Torch contract's typed SHA remain strict. Exact training resume/projection comparisons are unchanged. Regression tests cover both positive cases, metadata tampering and changing the actual Torch contract type while attempting to update only the file SHA. CPU bundle probes are not native updates or grasp trials.

The evaluation collector now makes one blocking host transfer per tensor batch before the established row loop. Pure Torch CPU/CUDA input tests produced identical summaries/raw JSON and unchanged source tensors/RNG. This is transport evidence, not a native performance benchmark; the running129/1439 historical comparison keeps its original frozen code for both models. Subsequent validations use independently frozen reviewed code.

The fixed100-case manifests provide small local perturbations of the authored grasp templates (active angles within0.0005rad and object XY within0.00005m). Distinct native readback identities must still pass. These local-case statistics do not establish broad pose, material or real-hardware robustness. R0 Adam pilots and R1–R5 effectiveness remain pending; zero native successes cannot be described as reliable grasping or solved lift degradation.

The final publication follow-up repeated the same full CPU command after all three fixes:5141 passed,96 failed,49 skipped,28 errors,91 subtests passed in266.10s,exit1. All4801 original baseline nodes,5079 prior-P0 nodes and5274 nodes from commit `d796917` retained exactly their previous status. All513 additions since baseline passed, including235 T2 additions and all40 nodes in this follow-up. Independent review passed62 initial-state-related and76 real-bundle-related tests. A real1439-derived CPU bundle was retained twice/read by selection without changing its source SHA or typed contract hash. The serial controller passed11 CPU checks, including its real-checkpoint fixture; none are native grasp trials.

## Actual historical129: completed native cold evaluation

At2026-11-06 09:33 UTC the actual129 model completed200 first-terminal cases,100 per grasp, after1750 control steps. This is the actual old visual129 policy, not a269/482 peak model or the new tanh-Gaussian smoke. The same frozen runner/physics/reset/camera/test manifest is continuing serially with actual1439; the pair and reward comparisons are not complete.

| Grasp | Native cases | Lift P50(mm) | P90(mm) | P95(mm) | Maximum(mm) |30mm rate|Max qualified hold(s)|Native successes|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Pinch two-finger|100|0.000030|0.000916|0.003668|0.005627|0%|0|0/100|
|Tripod three-finger|100|0.009632|0.391500|0.501771|1.033019|0%|0|0/100|

Both groups have0% five-second qualified holds. Slip is NA with0% valid coverage: all854757 physical rows have `no_anchor` and prior latch=false. This is missing fixed-anchor slip observability, not zero slip or proof of meeting0.5mm. Opposed-contact alive fractions are0% and1.072823%, respectively; simulated joint-torque saturation fractions are6.913689% and15.902894%. These are privileged simulator diagnostics, not real motor measurements.

An independent read-only audit bound source checkpoint SHA `5c85bc7a9516f9124a5f6dc99d39fe185016c60c0dff6f1c97346b2d429ecd0c`, report SHA `b31f3affd8508ad787a46c39f44aa72a9115527511cea976b902ac16d19b8163` and native certificate SHA `8b0c7cdf10cca9a7bda5277f00d0b410b2a86c76b0398f83643638e2d9b1c291`. It verified200 unique actual readbacks, maximum writer/readback numeric error4.76837158203125e-7 in original field units, all camera geometry with maximum environment P95=0.342867mm, and real3844D inputs against points/masks/encoders/FK. A single full raw gzip replay (854757 physical samples,62.768s CPU) reproduced every first-terminal trial and conditional-duration/reason field. No repeated reset episode or fixture was counted. This held-out historical diagnosis does not choose new weights or validation models, and establishes no reliable grasping.
