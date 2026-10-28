# Instant close completion force implementation plan

> **For agentic workers:** Use subagent-driven-development for the task gate and scoped review; the parent integrates telemetry and training metadata.

**Goal:** Use the current physics sample after mechanical readiness dwell for lift admission and display that exact sample per completed close.

**Architecture:** TensorTask accumulates non-force readiness independently of force sign, then gates its phase transition with current normal Z force. TensorEnv records accepted close-to-lift samples before any reset. The trainer publishes individual events, with identities, separately from rollout statistics.

**Tech Stack:** Python, PyTorch, existing Isaac tensor adapter, TensorBoard, pytest.

## Confirmed constraints

- User confirmed on 2026-10-27: non-force readiness stable 0.2 s, then current signed normal force >= 0; negative force continues close without restarting readiness dwell.
- Non-force instability still resets dwell. Hold compatibility retains 0.3 s dwell. Cap dwell at its threshold while waiting for force.
- Close timeout remains 4 s; no downward impulse reset. Keep reward coefficients, action/observation dimensions, physics guards and lift normal-plus-friction stability channel.
- Force is signed hand-to-object normal Z sum, excluding friction, gravity and table support.
- Main force points represent individual accepted close-to-lift events, not averages over environments, events or time. No completion means no force point, never a fabricated zero.
- Preserve environment index, one-based episode index, episode time, physical substep and the exact gate force. Preserve valid events even when a later substep fails.
- Existing dirty work and historical run data must be preserved. No commit/push or safety-limit relaxation.

## Task 1: Task gate

Files: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py`, `tests/test_dh116_close_instant_force.py`.

- [x] Add failing tests for negative history then nonnegative endpoint, positive history then negative endpoint, instability resetting dwell, opposite friction, timeout and inactive/failed rows.
- [x] Run `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_close_instant_force.py` and confirm behavioral failures.
- [x] Move `vertical_force_n >= 0` from readiness accumulation to transition selection; cap contact_time at the phase threshold.
- [x] Export `CLOSE_COMPLETION_FORCE_CONTRACT`; verify focused task and existing force-reward tests.

## Task 2: Exact endpoint telemetry

Files: `src/go2_pvcnn/control/dh116/tabletop_vertical_force.py`, `src/go2_pvcnn/control/dh116/tabletop_tensor_env.py`, `tests/test_dh116_close_completion_telemetry.py`.

- [x] Test two simultaneous endpoint events have separate scalar steps, IDs and normal forces despite opposing friction; no-event writes no force scalar.
- [x] Test a real TensorEnv task transition is captured from the same `normal_vertical_force` tensor passed to consume; subsequent force changes/reset do not overwrite it; later safety failure retains the event.
- [x] Add event writer with JSONL and TensorBoard event index. Compute normal force once in TensorEnv and pass it both to consume and endpoint recorder after guards and before reset.
- [x] Move `force_N` rollout curves to `rollout_force_N`; label transition-event aggregates explicitly. Use exact `force_N/close完成_*_瞬时` tags for main event curves.
- [x] Run focused telemetry and env suites.

## Task 3: Training contract and verification

Files: `src/go2_pvcnn/control/dh116/tabletop_tensor_training.py`, `tests/test_dh116_tensor_training.py`.

- [x] Test one-way exact migration from prior force contract to current-sample gate metadata; reject unrelated modifications and reverse migration.
- [x] Instantiate the endpoint writer after checkpoint resume, preserve event index in checkpoints, include counts in monitor, and close/flush via the existing writer finally block.
- [x] Run task, force, env, training and relevant reward regression tests; perform scoped independent code review and resolve findings.
- [x] Report offline verification separately from Isaac training validation. Existing training stopped at iteration 86 due to measured passive-joint overspeed; do not claim that cause is fixed by this change.


## Verification record (2026-10-28 Asia/Shanghai)

- New gate tests were run red first, then passed. Endpoint telemetry tests also failed before implementation; the combined legacy migration regression reproduced the independent review finding before its fix.
- Parent verification: `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_close_instant_force.py tests/test_dh116_close_completion_telemetry.py tests/test_dh116_vertical_force.py tests/test_dh116_lift_entry_force.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_close_downward_penalty.py` — **107 passed**.
- Independent reviewer rechecked telemetry/training after the migration fix — **36 passed**, no remaining important finding for the fresh-output-directory launch path. `git diff --check` passed.
- Event rows and scalar points are written immediately for each accepted close-to-lift substep, after guards and before reset. They survive a later physics fault even when that partial PPO rollout is discarded.
- Event numbering resumes from the completed checkpoint. Run directories scope event identities; the launcher refuses an existing output directory. Appending a resumed run into the same directory is not supported because a prior fault journal can be ahead of its checkpoint counter.
- A new 4096-environment/1024-episode-per-environment run was launched from the completed iteration-86 checkpoint: `runs/tabletop_tensor_20261027_000010_instant_close_force`. The directory prefix retains the pre-midnight task date; launch occurred at 2026-10-28 00:00 Asia/Shanghai.
- `training_start.json` confirms initial iteration **86**, timesteps **8454144**, and **13** optimizer state entries. The actual checkpoint migration contains exactly the new gate and endpoint telemetry contracts.
- TensorBoard at `http://127.0.0.1:6009` was repointed to the new run. Old averages remain in the old run; new rollout diagnostics use `rollout_force_N`, and primary endpoint values use `force_N/close完成_*_瞬时`. Set smoothing to zero to inspect raw individual values.
- The prior passive-joint velocity fault has not been fixed by this feature. The unchanged live velocity guard remains active; a successful startup does not prove long-term physical stability or lift success.


## User steering: restart from zero

- The user then explicitly requested fresh training. The short continuation above reached iteration 87 and was stopped at the user's request; its files and checkpoint were preserved.
- New run: `runs/tabletop_tensor_20261028_000214_fresh_instant_close_force`.
- Launched without `DH116_RESUME` / `--resume`, retaining 4096 environments, 1024 episodes per environment, current-sample close gate, rewards and physics configuration. TensorBoard port 6009 now points to this fresh run.

## Continuity audit and second fresh start (2026-10-28)

The user confirmed the continuity boundary: ordinary failures reset only the affected environment; serious physical failures retain global protection stops. The user then requested another fresh start after these fixes and TensorBoard inside VS Code.

Audit findings and disposition:

1. **Fixed — optional endpoint telemetry could abort physics.** `tabletop_vertical_force.py:21` now handles only sink `OSError` as degraded telemetry. Journal and TensorBoard failures are tracked independently, preserving existing rows and reporting the first potentially affected event. The tensor trainer shares the guarded writer. Invalid data, programming errors and checkpoint/monitor persistence failures remain strict.
2. **Fixed — logger cleanup could mask a physical error or prevent final outputs.** `tabletop_vec_training.py:645` attempts each cleanup independently, retains the original training failure and writes the checkpoint/report with logging diagnostics. `tabletop_tensor_training.py:796` publishes the original stopped monitor before cleanup and refreshes it if cleanup adds telemetry diagnostics. `watch_dh116_tensor_ppo.py:107` reports optional telemetry degradation without abandoning physical monitoring.
3. **Unresolved physical transient; protection worked.** The preceding fresh run stopped after completed iteration 8 / 786432 transitions. Environment 444, episode 1, right passive `finger22` reached 3.0056912899 rad/s above the unchanged 3.00001 rad/s guard. The 32-frame fault trace shows no recorded command/root jump or reset boundary; it does not contain enough solver/actuator information to establish the cause. The telemetry fixes do not repair or waive this physical fault. Before the stop, 1294 episodes had completed and 1234 recoverable physical episodes had been reset individually.
4. **Open requirement gap — actual contact heights.** `tabletop_tensor_task.py:170` leaves `contact_target_z_m=None` in formal PPO. Synchronized palm Z offsets do not guarantee equal measured contact-point Z; the authored palm heights differ by about 9.513 mm. This audit did not implement a new contact-height controller. Current settings must not be described as proven equal-height physical contacts.
5. **Open optional-radius issue — bottom calculation.** `tabletop_tensor_world.py:204` and `tabletop_gpu_isaac.py:227` use the original 0.008 m radius in cylinder-bottom telemetry. The optional 20 mm candidate would be wrong by 12 mm when horizontal. The active fresh run uses the original 8 mm radius.
6. **Open optional-radius issue — initialization geometry.** `configs/dh116_training/grasp_initialization_radius20mm.json:14` changes height without rebuilding the original finger geometry; the audit's USD/FK check found about 11.39–11.86 mm pad penetration for that candidate. It remains unqualified and is not used in this launch.

Regression maintenance updated stale test expectations and incomplete fixtures to the current pressure contract, removed dependence on an absent historical run, and supplied mount/contact buffers in mocked worlds. No physical thresholds, reward coefficients or reset classifications were relaxed to make tests pass.

Fresh verification:

- All 34 modified/new test files: **349 passed, 8 skipped** (CPU run).
- After the final stopped-monitor ordering change, parent focused telemetry/trainer/recovery/monitor suites: **96 passed, 1 skipped**.
- Independent final ordering review: **46 passed**, no new important finding in that scoped change. The implementation agent's broader focused suite: **286 passed, 1 skipped**. These overlapping counts are not additive.
- `git diff --check` and `bash -n scripts/python.sh scripts/train_dh116_tensor_ppo.sh` passed.
- New run: `runs/tabletop_tensor_20261028_001809_fresh_continuity_fix`; 4096 environments, 1024 episodes per environment, no resume argument. `training_start.json` records null resume source, zero initial iteration/transitions/episodes/successes and zero optimizer-state entries.
- TensorBoard port 6009 was repointed to this directory and the existing VS Code integrated browser was refreshed. Smoothing is 0. Startup is not evidence of completed training or successful lifting; severe physical stop protection remains enabled.
- Post-launch live check: iteration **2**, **196608** transitions, **109** completed/recovered physical episodes, status `training`, endpoint telemetry `ok`, observer `problems=[]`, and a nonempty `tensor_ppo.pt`. TensorBoard's first episode scalar is step 1 / value 8, followed by step 2 / value 109; the VS Code page shows the new directory and these new data. No close completion had occurred yet, so no endpoint force point was fabricated.

## Monitoring follow-up: reproducible passive velocity fault (2026-10-28 morning)

- At the user's next monitoring request, the 00:18 fresh run had stopped after iteration 8, at about 00:23. Its entire bounded fault trace is identical to the preceding 00:02 fresh run: environment 444, right `finger22`, 3.0056912899 rad/s. Optional telemetry remained healthy. The severe guard is doing its specified job; restarting the same deterministic configuration alone does not solve the fault.
- Hypothesis 1: the default articulation constraint order permits later constraints to change velocities after the solver's velocity-limit pass. The installed Isaac Sim 5.1 API and NVIDIA's [107.3 stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order) describe `solve_articulation_contact_last`, which moves contact and joint maximum-velocity constraints later in the solver. Added an explicit opt-in CLI flag, scene readback verification and checkpoint contract identity. Contract changes cannot silently resume older physics. The formal launcher default was not changed.
- Four new tests failed before implementation; focused tests then passed (54 passed, 1 skipped), and independent review passed 90 tests. These verify configuration/metadata/guard behavior, not numerical convergence.
- Controlled fresh seed-42 run `runs/tabletop_tensor_20261028_095810_solver_order_fresh` retained 4096 environments and all other contract fields; checkpoint comparison confirmed the solver-order metadata was the only difference. **This candidate failed** after iteration 4: environment 2354, right `finger22`, 3.0033273697 rad/s. Solver ordering alone is insufficient; it was not promoted to the launcher.
- Hypothesis 2: 16 velocity iterations leave a material residual in the competing contact/mimic/velocity constraints. Added 64 as an explicit CLI candidate, keeping the default at 4 and the formal launcher at 16. The new CLI test failed before the one-line parser change. Focused verification: 55 passed, 1 skipped. Fresh controlled run `runs/tabletop_tensor_20261028_100345_velocity64_fresh` uses the original solver order and only increases velocity iterations to 64. Live qualification pending.
- Every candidate keeps the 3.00001 rad/s measured guard, 2.97 rad/s passive solver cap, 480 Hz, PGS, 64 position iterations, mimic frequency 200, reward coefficients and isolated-reset rules. No measured velocity clamping or guard relaxation was introduced. TensorBoard port 6009 follows the active candidate; prior stopped runs and evidence are preserved.

## Monitoring follow-up: early joint-limit task boundary

- The velocity-64 candidate stopped after iteration 17 / 1,671,168 transitions. Its all-joint speed peak stayed at 2.9700000286 rad/s, but environment 291's right active `finger12` reached -0.0122538814 rad against a +0.0079126228 rad lower limit: 0.0201665042 rad outside, exceeding the unchanged 0.02 hard tolerance. Close event journal contains 97 events (93 in the last completed update); successful lifts remain 0. This is not a successful training qualification.
- The 32-frame trace shows the thumb command held at its mechanical lower bound while the measured joint continued outward. The controller was already pausing lift and releasing the opposing index. First outside-by->0.005 sample was sequence 3418 at 7.120833 s; the hard crossing was sequence 3444 at 7.175 s, 26 substeps / 54.17 ms later. At the earlier sample, all strict physical guards were satisfied. Underlying contact/solver drift is not claimed fixed.
- Identified continuity gap: task joint-limit failure and independent hard failure previously shared the same 0.02 threshold, leaving no earlier failed-grasp boundary for this gradual excursion. Added a stricter **PPO-only task boundary** at more than 0.005 rad outside the active angular joint's nominal range. This magnitude is a new conservative task threshold; it is not equivalent to the controller's 0.005-rad warning margin *inside* the lower limit.
- Eligibility requires a live environment, all strict physical guards satisfied, and no pre-existing joint failure. The existing `joint_limit` failure and -25 penalty are retained, only that environment is reset immediately, and recovery is counted after validated reset. Any same-sample hard violation still stops globally. Initial reset, immediate reset, frozen/pending next physics, and end-of-tick state must also clear the stricter bound; no repeated-reset escape from bad reset physics.
- Scope is the twelve active angular joints, excluding prismatic wrist coordinates. Recovery-disabled runs and control probes do not enable the boundary. The `training_joint_limit` checkpoint contract records it; missing, changed or reverse contracts are rejected. Next validation uses a fresh run, not a silent checkpoint migration.
- New behavioral tests first failed: **6 failed, 11 passed**. After implementation and scope/contract regressions, parent focused verification (`test_dh116_tensor_env.py`, `test_dh116_tensor_training.py`, `test_dh116_close_completion_telemetry.py`, `test_dh116_close_instant_force.py`, `test_dh116_velocity_limits.py`, `test_dh116_solver_iterations.py`): **146 passed**.
- Independent review found and reproduced a probe-loading incompatibility from the new training-only metadata. Added a failing probe regression before allowing only an exact known saved `training_joint_limit` contract to be omitted in inference probes; unknown thresholds/versions, physics differences and reverse omission remain rejected. Final parent seven-file suite (including `test_dh116_control_probe.py` and expanded both-hand, boundary and same-sample hard-fault tests): **173 passed**. Independent final scoped suite: **109 passed, 1 skipped**; also checked 72 active-joint/limit/substep combinations and tick-end reset fault injection. These counts overlap. `git diff --check` passed; no remaining blocker in the scoped review.
- Fresh actual validation started at 12:02 Asia/Shanghai: `runs/tabletop_tensor_20261028_120230_early_joint_fresh`, 4096 environments, 1024 episodes per environment, seed 42, velocity iterations 64 with original solver order. No resume argument; old runs preserved. TensorBoard port 6009 points to this run. Actual startup and fault-point passage are still pending at this entry.
- Actual startup verified: null resume, zero initial iteration/transitions/episodes/optimizer state. Checkpoint comparison against the previous velocity-64 run shows exactly one changed field, `training_joint_limit`. First five iterations matched the previous run's episode counts, rewards, close-event counts and physical peaks exactly.
- By iteration **19 / 1,867,776 transitions**, the run passed the prior iteration-17 stop point. **4** early joint-limit failures were isolated, with 828 recovered episodes total (4 joint-limit, 822 mimic, 2 force), 4,860 completed episodes, 100 close-to-lift events and 0 successful lifts. Speed peak remained 2.9700000286 rad/s. Observer reports no problems and telemetry is healthy. This validates passage of the earlier failure point, not long-term solver convergence or lift success. Evidence: `early_joint_contract_comparison.json` and `early_joint_live_verification.json` in the new run.
- At the user's dashboard check, server `/data/environment` and event 71 were matched to the live journal (environment 3725, control step 306, PPO iteration 13). Event 91 belongs to PPO iteration 17. Close-force plots use individual event indices, not PPO iteration numbers. TensorBoard now uses a visible named run (`tabletop_tensor_20261028_120230_early_joint_fresh/.`) and window title rather than `.`; the existing VS Code tab was refreshed. Training was not restarted for this display change.

- Continued live observation reached iteration **31 / 3,047,424 transitions**, with **11** isolated joint-limit failures and no global stop; all-joint peak **2.9700016975 rad/s**, below the unchanged 3.00001 guard. 133 close events, 0 lift successes. Trainer and 30-second read-only observer remain running; observer reports no problems. The observer records faults but does not automatically edit code or restart training. No further parameter changes were made during normal observation.
