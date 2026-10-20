# DH116 Grasp Speed Implementation Plan

> **For agentic workers:** Use executing-plans for the user's Inline Execution workflow. Do not spawn agents or restart the live Isaac instance during offline implementation.

**Goal:** Implement the approved staged-speed and simulation-throughput design, then verify real bilateral hold and lift.

**Architecture:** Keep physical rod integration separate from visual USD publication. Parameterize motion and coupling only after measuring baseline cost and validating same-physical-time numerical equivalence.

**Tech Stack:** Python, NumPy, USD, Isaac Lab, DeformX, pytest.

## Global Constraints

Use the existing user-selected isolated DH116 project and feature branch; preserve dirty changes. Use one Isaac instance. All CPU cores are authorized. Retain collision guards, real contact impulses and reaction forces; no attachment-based success. Spec: `docs/superpowers/specs/2026-10-18-dh116-grasp-speed-design.md`.

## Task 1 Visual publication without changing physics

Files: create `src/go2_pvcnn/control/dh116/wire_mesh_publisher.py` and `tests/test_dh116_wire_mesh_publisher.py`; modify `scripts/verify_fixed_dual_panda_dh116.py` and `tests/test_dh116_gui_render.py`.

- [x] Run baseline `bash scripts/python.sh -m pytest tests -q`: 651 passed in 26.81s.
- [x] Add real in-memory USD tests: publish an initial rod and translated rod, assert points match `rod_surface`, topology stays unchanged, and USD change notices do not report topology rewrites on the second call. Reject changed node count before altering mesh data.
- [x] Add a runner integration test extracting `if do_render` and asserting the loop's `publish_wire(snapshot)` is inside it, while `sim.step(render=do_render)` remains unconditional.
- [x] Run focused tests and verify failures before implementation: 4 expected failures, 1 pass.
- [x] Implement `WireMeshPublisher(mesh, radius)` with cached USD attribute handles and a fixed vertex count. On each call compute `rod_surface`; on first call only set counts/indices, then set points/extent. On later vertex-count mismatch raise before any writes.
- [x] Bind `publish_wire = WireMeshPublisher(mesh, .008)` after Isaac startup. Move `do_render=render_due(step,args.render_every,args.headless)` before the wire block, and use `if do_render: publish_wire(snapshot)`. Preserve initial/restored-state publication and all physics, force, metric and validation calls.
- [x] Run focused tests, full regression and `git diff --check`: 9 focused passed, 655 full passed in 32.09s. Live process still uses old code; total speedup not measured.

Task 1 implementation and all listed test cases are present. Offline in-memory USD benchmark, 1001 iterations and median of 3: baseline publication 3.25348 s, render-every-10 fixed-topology publication 0.05485 s. This is not an Isaac or whole-run performance result. Next checkpoint is Task 2, not actual grasp completion.

## Task 2 Performance measurement and control configuration

### Task 2a Contact speed governor

Create `speed_profile.py` with immutable `GraspSpeedProfile(far_m_s=.02, near_m_s=.001, loaded_m_s=.001, acceleration_m_s2=.05, slowdown_gap_m=.002, lift_m_s=.01)` and `ContactSpeedGovernor.update(time_s, force_speeds, loads, valid, gaps, active)`. All channel arrays are (2,2); reject nonfinite values before mutation. Free-space speed envelope is `min(far, sqrt(near**2 + 2*acceleration*max(gap-slowdown_gap, 0)))`. Slew from zero using measured dt. Contact or positive load uses bounded force feedback; overload retreat and inactive-channel stop override smoothness. Distances are measured mesh gaps, never approach goal error.

Add keyword-only `speed_profile=None` to `TouchFirst` and keyword-only `surface_gaps_m=None` to its update. With a profile, validate gaps before modifying phases, then apply the governor after ordinary touch/opposition/preload logic. Without a profile preserve existing results exactly. Tests cover braking, asymmetric gaps, load priority, immediate inactive stop, negative retreat, invalid inputs without state mutation and unchanged legacy behavior. Run tests red, implement the formulas, run green and full regression. Do not expose the candidate in CLI until allocator limits, sequence plumbing, lift and measured-gap wiring are complete.

Task 2a completed as a component, not runtime activation: 13 expected missing-feature failures preceded implementation; 35 speed/touch tests pass, full regression 675 passed in 27.31s. The governor preserves legacy default and requires finite measured (2,2) gaps in staged mode. Current runner still has no staged profile; next work is allocator limit/scaling diagnostics, sequence/restore plumbing, synchronized lift speed and explicit CLI configuration. Original joint caps and torque limits have not been increased.

### Task 2b Sequence and allocation integration

Pass `speed_profile=None` through TabletopGraspSequence and StageResetSequence; require state-triggered mode and finite 4-channel measured `surface_gaps_m` before changing sequence time in staged mode. Pass gaps reshaped (2,2) into TouchFirst. Parameterize SynchronizedTransport with `max_speed_m_s=.005`; staged initial lift uses .01, retaining the 1/2 mm synchronization thresholds and ramp duration .2 s. Reject staged lift above the allocator's current .01 m/s transport bound.

Extend `coordinated_bimanual_increment` with `normal_speed_limit_m_s=.002`, `force_speed_limit_m_s=.002` and `return_diagnostics=False`; validate positive finite limits <=.05 and <=.002 respectively. Keep Panda .05 rad/s and hand .02 rad/s caps. When requested, return `(increment, report)` including common scale and `einsum('shij,sj->shi', J, velocity)` before/after scaling; these are model-allocated velocities, not actual physical tracking. Loaded force speeds remain capped independently of the fast free-space candidate. Tests live in `tests/test_dh116_staged_sequence.py`; five RED tests observed before implementation.

Runner opt-in `--grasp-speed-profile staged` is only valid with guarded grasp sequence; default baseline remains unchanged. Pin the serialized profile in run_config, pass measured contact sample gaps to update, log allocation diagnostics each commanded step and pass profile through restore. Do not activate runtime until integration tests and regression pass; actual faster trajectory/effort verification and dt benchmark still remain.

Task 2b implemented and offline checked: 5 sequence/allocation RED tests and 2 restore/runner RED tests preceded edits; 58 focused passed, full regression 682 passed in 27.41s. CLI staged mode is now wired and pinned but not launched. At measured q from baseline step 43896, offline USD allocation of candidate left/right thumb speeds 9.327/1.000 mm/s was scaled by 3.019 to 3.089/0.331 mm/s. This is not measured tracking. Shared scaling can slow the near hand when the far hand saturates; keep that limitation visible and assess feasible speed/joint headroom before claiming action acceleration. The initial certified approach path still uses its existing timing; far-speed governor currently affects free-space touch, not retiming the whole approach.

- [ ] Extend separate rod, visual, source-guard, logging and Isaac timers; measure steady-state intervals excluding GUI pause and initialization. Add tests for time accounting.
- [ ] Add explicit speed configuration with initial approach 0.02 m/s, near contact 0.001 m/s, loaded closure cap 0.001 m/s, lift 0.01 m/s; include acceleration and braking distance. Preserve old baseline as an explicit profile.
- [ ] Test near-contact slowdown against actual surface gap, acceleration continuity, load-triggered progression and arm/hand rate allocation. Report allocated rather than merely requested speed. Do not increase torque limits.

## Task 3 Coupling interval comparison

Prediction optimization whole-backend parity check: three alternating 5 ms prescribed-moving real-pad runs using original public revalidation vs internal validated-input predictor. Median 1.31985→1.08843 s; sampled rod positions/velocities/directors/angular velocities, accumulated linear impulses and force peaks are bit-identical. `runs/forward_gui_render10/prediction_ab.{json,py}`. This uses inner 10 us only to establish optimization parity, NOT time-discretization accuracy or whole-Isaac speed. The stable integration design and state migration choices remain awaiting user response; no new integration method has been implemented.

Moving reference checkpoint: prescribed vertical mesh velocity 10 mm/s, 100 ms, outer 100 us, inner 2.5 vs 1.25 us yields max position difference 6.49017e-7 m, peak-force difference 0.000121685 N, accumulated impulse difference 1.55403e-7 N s. All tolerances pass for this prescribed-boundary fixture, not robot feedback. Evidence: `moving_dt_halving_probe.{json,py}` in current run.

Exact prediction optimization: reuse boundary-validated immutable motion internally via `_predict_validated_motion`, keep public revalidation, dt checks and full predicted-output validation. Include moving_contact_models.py in runtime backend source pins. Two predictor RED tests + one pin RED test preceded changes; 80 focused and 689 full tests pass. Paired function microbenchmark median .388637→.208689 s, not backend/Isaac proof. Existing live process unchanged. New implicit/semi-implicit contact scheme requires the pending design choice before implementation; current reference integrator is preserved.

Fine static reference established: original friction/material/geometry, 100 ms bilateral fixture, outer 100 us; inner 2.5 vs 1.25 us gives max centerline difference 6.24497e-7 m, max peak-force difference 0.00095003 N and max accumulated impulse difference 1.35468e-7 N s. All specified engineering tolerances pass for this fixture. Wall times 47.489/94.795 s, not a performance solution. Report/script `runs/forward_gui_render10/inner_dt_finest_halving_probe.{json,py}`. Use as the reference for a stable contact-damping integration experiment and moving-boundary checks, not as proof of full closed-loop convergence or grasp completion. Production dt unchanged.

Sensitivity localization: disabling tangential damping only in the standalone diagnostic reduces 10→20 us centerline discrepancy to 7.49e-7 m. Production friction remains unchanged. Frozen initial-geometry mass-normalized damping has largest eigenvalue 761473.75/s, fastest time constant 1.313 us (NOT a full nonlinear/Verlet stability bound). With original tangential damping 1000, 5→2.5 us gives centerline difference 0.123491 mm (position criterion passes) but right-thumb peak-force difference 0.058707 N exceeds 0.05 N, so overall convergence still fails. Reports/scripts `inner_dt_no_friction_probe`, `contact_stiffness_probe`, `inner_dt_fine_halving_probe` under the current run. Next compare 2.5→1.25 us or design a stable friction-damping integration preserving physical coefficients and reaction impulses; do not silently weaken tolerances or remove friction. No runtime migration/restart yet.

Numerical precision finding: static bilateral real-pad experiment extended to 100 ms, outer 100 us fixed. Relative to inner 10 us, inner 20/50/100 us produce maximum centerline differences 1.585/6.887/13.425 mm; reject all against 0.2 mm tolerance. Halving to 5 us still differs by 0.548 mm, so 10 us is NOT established as converged for this fixture. Production settings unchanged. Raw reports/reproduction scripts: `runs/forward_gui_render10/inner_dt_probe.{json,py}` and `inner_dt_halving_probe.{json,py}`. Next investigate contact/time-discretization sensitivity and finer-step convergence before claiming numerical qualification; fixed-fixture results do not prove actual closed-loop failure, but finite output alone is insufficient. State migration has been proposed to the user, not executed; source SHA/config/reference checks passed without modifying old pins.

Internal-query optimization checkpoint: conservative distance-to-AABB screening compiled with Numba (no fastmath), preserving float32 padding and exact candidate mesh queries. Measured current-pose candidate total 80→66. Four missing-feature RED tests preceded implementation; 49 focused tests, 686 full tests, and 43 vendor rod-mesh tests pass. Three paired alternating static bilateral backend measurements (5 ms physical each) gave medians 1.908289 s old vs 1.590967 s new with bit-identical sampled positions and accumulated impulses. Evidence/script in `runs/forward_gui_render10/broadphase_ab.{json,py}`. Concurrent live Isaac and regression load limits performance inference; not realtime proof or moving-robot feedback validation. Existing instance is unchanged; production interval remains 100 us.

Initial diagnostic (not Task3 acceptance): `runs/forward_gui_render10/coupling_probe.py` takes `falling`, `single`, or `bilateral`. Real pad meshes, unchanged 40-element rod and contact parameters; fresh starts. Falling lasts 60 ms, contact cases 10 ms, matched samples every 1 ms. Contact fixtures prescribe fixed isolated pads with nominal selected-point penetration 0.2 mm at rod center height 0.4 m; no robot feedback. Outer intervals 100/250/500/1000 us, inner 10 us. Falling maximum centerline difference 5.13e-8 m; contact states essentially identical, bilateral peaks 0.642/0.631/0.970/0.767 N. Falling wall times 8.022/7.657/7.635/7.455 s; bilateral 1.753/1.803/1.900/1.804 s. Each is one warmed timing with live Isaac competing for resources. No reliable performance or closed-loop qualification claim. Next prioritize internal contact-query overhead, then moving-boundary/robot-feedback comparison. Runtime dt unchanged. Focused regression this turn: 32 passed in 0.51s.

- [ ] Build separate falling-rod, single-contact and bilateral-contact benchmarks with equal initial states and physical durations for coupling dt 0.0001, 0.00025, 0.0005 and 0.001 s; retain rod dt 0.00001 s.
- [ ] Compare point error <=0.0002 m, force error <=max(0.05 N,5%), impulse error <=max(0.0001 N s,5%), and step-halving convergence. Reject unstable or collision-failing candidates.
- [ ] Audit every dt-dependent runtime guard, dwell calculation and impulse-to-force conversion before enabling any new interval; add regression tests for each changed constraint.

## Task 4 Actual validation and handoff

- [ ] Validate the faster requested trajectory, limits, actual tracking and drive headroom before grasp qualification.
- [ ] Preserve current run. Stop it cooperatively only when replacement configuration is ready; verify process exit before launching another instance. Restore a pinned stage only through validated compatibility or explicit migration, never by editing SHA values.
- [ ] Verify bilateral true contact, preload, sustained hold, 30 mm lift and 1 s hold with <=1 mm station mismatch and existing collision/slip limits. Record raw evidence, video and real-time ratio.
- [ ] Update README, CHANGELOG and resume information with actual results and unfinished items. This plan remains incomplete until dynamics and performance requirements are verified; Task 1 alone is not overall completion.

## Review

All four spec areas remain in scope: motion, throughput, numerical equivalence, actual grasp. Task 1 is independently testable; detailed code changes for Tasks 2–4 must be planned after profiling and dt results establish the valid runtime parameters, without dropping those tasks.
