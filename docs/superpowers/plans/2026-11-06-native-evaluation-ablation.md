# DH116 T2 native evaluation and controlled ablation plan

> **For agentic workers:** Use subagent-driven-development with test-first changes, independent specification and quality review. The user's later instruction “原进程可退出” superseded waiting for natural completion. The verified original PID150108 was stopped with SIGINT after preserving complete1750; the old monitor was deleted. Keep single-instance execution and separate outputs. PR remains Draft.

**Goal:** obtain actual native smoke/recovery evidence, matched129/1439 deterministic statistics for at least100 independent cold-reset cases per grasp, and controlled R0–R5/Adam warm-start comparisons without changing original physics or merging PR#1.

**Architecture:** opt-in pre-reset physical telemetry feeds the existing first-terminal aggregator; sensor-only inference reuses RGB-D/3844D history while371D remains privileged. Explicit offline experiment projections produce new immutable checkpoint bundles. A serial native controller enforces the old-run completion/GPU3/exclusive-lock gate. Validation and held-out evaluation manifests are distinct, fixed before policy comparisons.

**Tech Stack:** Python3.10, Torch2.7, RSL3.1.3, Isaac4.5, TensorDict, pytest; CUDA hidden for all preparation tests. Worktree remains `codex/dh116-lift-repair-20261106`, PR#1 remains Draft.

## Authority, resources and immutable inputs

- [x] User explicitly requested T2, R0–R5, validation selection and preserve/reset-Adam comparisons.
- [x] User explicitly selected single Isaac instance; GPU2 must not be used concurrently even though it was idle at inspection.
- [x] Original tracked run is `dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z`, PID150108 at inspection, native successes0. No stopping, restart, source or output edits.
- [x] Actual source1439 frozen SHA `d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9`; explicit v3 derivative SHA `fe04af732c205f37180ad8dd003f1a5f36d4edfa3a538fed78067827163a9abe`.
- [x] Actual129 source exists;269/482/1328 weights do not. No renaming surrogates.
- [x] Snapshot the final complete old1750 model before the user-authorized exit; SHA837791335e5f1c2153db763ed419381c708ec8b1f8ff2d4abfc6fa025aab249d. Original1439/129 unchanged. Do not call this2048 completion.

## Task1: opt-in pre-reset telemetry

**Own files:** `tabletop_tensor_env.py`, new `tabletop_evaluation_telemetry.py`, focused tests.

- [ ] RED terminal-before-reset, first terminal freeze, invalid slip versus0, qualified hold versus raw bilateral contact, estimated torque versus action saturation, disabled-path equality.
- [ ] Add `enable_evaluation_telemetry(callback)` with a batch callback after task consumption and before any physical/task reset, using the pre-consumption live mask. Keep the default path disabled and unchanged.
- [ ] Payload includes actual substep dt, pre-reset terminal/native-success events, episode-reference net bottom lift, native qualified hold, observable fixed-anchor drift, raw opposed contact, implicit-actuator estimated torque with original limits, submitted24D action and explicit validity.
- [ ] Read only; do not update reward/controller/success trackers, consume RNG or append Actor features. Torch/PhysX values are diagnostic privilege, never real hardware feedback.

```python
assert callback_terminal_peak_m == .031
assert collector.summary()['completed_count'] == 1
assert invalid_slip_result is None
```

## Task2: visual deterministic evaluation and cold-reset identity

**Own files:** new `tabletop_vision_inference.py`, `tabletop_vision_evaluation_contract.py`, `tabletop_vision_evaluation.py`; evaluation-only dispatch in `tabletop_tensor_training.py` / native CLI; focused tests. Do not edit Task1 files.

- [ ] RED inference loads actual v2/v3 actor topology without PPO/Adam/storage, restores construction RNG, uses eval/inference_mode and never feeds371D into3844D actor.
- [ ] Explicit inference projection accepts a smaller evaluation batch without pretending exact training resume. Require unchanged camera calibration/resolution/history/action/physics provenance; keep v2 deterministic bounded mean and v3 tanh(mean) distinct by version.
- [ ] Generate seeded authored cold-reset states from actual initialization templates, including complete named q/qd, roots/root velocity and object pose/velocity. Resolve active/mimic joints using the existing coupling equations, original limits and same controller reset convention.
- [ ] Freeze at least100 distinct float32 writer states per grasp in an evaluation manifest and a separate validation manifest; trial labels/origins/quaternion signs are not identity differences. Synthetic fixtures remain test-only.
- [ ] Apply the manifest through the existing actual reset hooks, clear visual history/previous action, read back the real state and camera before actions, preserve raw readback/evidence and verify uniqueness. Fail closed on insufficient valid independent cases, mismatched reset/readback, invalid camera or missing terminal samples.
- [ ] Consume Task1 callback into `FirstTerminalAggregator`, stopping once every independent trial has its first terminal or an explicit bounded timeout. Include terminal sample before reset, never count fast-environment repeated episodes.
- [ ] Emit per-style raw trial rows and P50/P90/P95/max lift,30mm rate, qualified hold, fixed-anchor drift/coverage, native success, raw bilateral contact duration and estimated torque saturation. Set native certification only with real native execution/readback evidence, not a writable fixture flag.

## Task3: explicit experiment projection, immutable snapshots and selection

**Own files:** new `tabletop_vision_experiment_checkpoint.py`, `tabletop_validation_selection.py`, offline CLI and tests. Root integrates training hooks after review.

- [ ] RED source immutability/no-overwrite/sidecar races and finite exact copied model/Adam/LR/RNG state.
- [ ] Explicit v3 projection from1024 to a small balanced batch for smoke, with a whitelist of count/group/per-env-reference/minibatch fields and recomputed existing semantic hashes. Preserve actor3844/critic371/action24, source file bytes and original physics; reject all other contract differences. Never relax exact v3 resume.
- [ ] Keep original1024 derivative immutable. Native smoke uses a separately named projected derivative, then exact same-small-contract resume to prove saved state reload.
- [ ] Retain periodic full checkpoint bundles by iteration/hash. A validation sidecar records manifest/report/checkpoint hashes, version and provenance; do not lose Adam/RNG/counters or overwrite an existing immutable candidate.
- [ ] Rank complete independent native validation results lexicographically by native success, qualified hold,30mm achievement, slip, lift distribution and action stability; report both grasp styles. Reject incomplete/missing/fixture-only evidence.
- [ ] Maintain separate best_validation and best_success_rate selections with immutable candidate history. With zero native successes, do not produce a best_success model.

## Task4: independent warm-start optimizer modes

**Root-owned integration:** `tabletop_vision_training.py` and reviewed CLI forwarding.

- [ ] Add explicit preserve/reset optimizer launch modes for independent experiments. Default preserve is unchanged. Validate the source completely before reset; preserve policy/critic, LR, existing Torch RNG and source provenance in both modes.
- [ ] Reset Adam only in the new process after loading the trusted source; record that this is a warm start, not exact optimizer resume. Preserve source global iteration separately from experiment budget and avoid serializing a falsely complete checkpoint with empty Adam at a no-op budget.
- [ ] Save reset provenance in subsequent complete checkpoints, and verify Adam step/moments after at least one full update. Compare both modes with matching initial policy, seeds, states and update/transitions budget.

## Task5: isolated reward factors

**Root-owned files:** a new lift reward-experiment helper, opt-in integration in existing reward-ablation module, focused tests. Legacy `support_deficit` R0 output must remain unchanged.

- [ ] RED R0 equality, height hold/up/drop terms, native invalid/contact loss/support deficit/slip gates, deadband, physical dt scaling and selected-row reset.
- [ ] Define R1=R0+height-hold, R2=R0+upward, R3=R0+drop, R4=R0+validated hold+up, R5=R0+all independently supported factors. Freeze all coefficients/config hashes before native experiments.
- [ ] Positive quality requires valid opposed contact, support, slip and speed. Use actual episode-reference net height,3mm entry/30mm target, real dt and a vibration deadband. No actor feedback or physical/success parameter change.
- [ ] Run short same-seed/state/source-policy/budget single-factor experiments first; combine only after results, and report absent improvements honestly. Include contact-loss and low-support diagnostic cases without making them positive success.

## Task6: serial native execution after original exit

- [x] Verify user-authorized original exit, frozen complete checkpoint, old PID gone and GPU3 idle; use an exclusive lock and separate output per native process.
- [x] Small v3 smoke: real RGB-D, two complete PPO updates1440/1441, immutable save, exact-small-contract resume update1442. Third no-op native restore preserved all model/Adam/LR/RNG/counters exactly. See native-stage2-progress report. No native grasp improvement claim.
- [ ] Paired129/1439 deterministic evaluation: same frozen manifest, physics/camera/reset/seed; at least100 real first-terminal trials per grasp per model. Maintain split/provenance/unique readback evidence.
- [ ] Compare preserve/reset Adam under matched budgets, then R0–R3 and supported R4/R5, using independent validation for selection and held-out evaluation for reporting. No best-by-training-peak shortcut.
- [ ] Publish actual native smoke report, every-grasp statistics, ablation and optimizer tables, selected model basis and unresolved issues. PR stays Draft; user decides readiness/merge after this stage.

## Validation and publication

- [ ] All feature changes RED then GREEN; spec review before quality review. No native launch during preparation.
- [ ] Targeted suites followed by full CPU regression with original4801-node baseline comparison (known96 failures/28 errors/49 skips retained separately).
- [ ] Check source/physics/camera SHA, old-source1207-file integrity, original run health and Git diff before publishing.
- [ ] Update README/CHANGELOG with actual CPU/native stage. Push only reviewed code/docs to the same draft PR; models, runs, raw logs and secrets stay out of Git.
