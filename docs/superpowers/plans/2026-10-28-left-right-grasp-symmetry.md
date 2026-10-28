> Update 2026-10-28: the close dwell described below is superseded by [instant per-hand close hold](2026-10-28-instant-per-hand-close-hold.md). Current coordinated control has zero close dwell and uses both current per-hand readiness signals.

# Left/right grasp symmetry investigation and correction

**Goal:** Use the better left grasp as a physical reference for the right hand, as requested by the user. Shared Z commands alone do not establish symmetric contact.

**Constraints:** Keep the real left/right source assets, mechanical limits, existing affine coupling, motor torque budget, physical guards, rewards and success predicate. Do not substitute a second left hand or relax right-hand limits to manufacture symmetry. Preserve existing dirty work. No commit or upload requested. Only one Isaac process at a time.

## Investigation and implementation steps

- [x] Compare authored roots, initial and closed joints, opening distances, pressure directions and real contact traces.
- [x] Derive world/root reflection with actual source USD frames. Test whether a strict whole-hand mirror is mechanically feasible.
- [ ] Produce a reproducible, feasible right-hand candidate matching left functional pad points and normals, with equal initial opening distance. Recompute pressure direction from the new FK, rather than copy joint angles or use stale calibration.
- [ ] Retain the left seed unchanged; write separate candidate initialization and matching policy envelope. Audit source hashes, coupling, bounds, full-hand table clearance, radial gaps and contact-frame residuals before any simulation.
- [ ] Compare against the existing synchronized eight-case run using unchanged controller/physics settings and the same depths/force targets. Record per-side thumb/index contact, measured loads, force feedback, release states and actual lift result.
- [ ] Report separately what is geometrically aligned, what is verified dynamically, and any unresolved grasp failure. No imitation/PPO launch without qualified successful teacher trajectories.

## User clarification and completed lift correction

The user explicitly clarified that exact copied behavior is unnecessary: both sides must complete close before lift, and their lift heights may differ. Functional geometry matching remains a diagnostic way to improve the weak right grasp, not a whole-hand symmetry requirement.

The current measured-load readiness already requires both hands to reach at least 1 N, meet the existing mechanical conditions for 0.2 s, and have nonnegative current aggregate pure-normal Z force at transition. Regression tests now explicitly reject 4 N on one side with 0.2 N on the other, including after extended dwell. Thresholds were not increased or weakened.

Fixed a real mismatch in TensorWorld: shared-Z safety projection had continued through lift and overwritten unequal accepted palm offsets. It now shares heights only before lift, then preserves the two captured offsets with independent per-hand clearance correction. The existing common upward increment remains; this does not add independent lift-velocity policies. Contract version `accepted_close_grasp_force_servo_v3` records the change.

Offline validation: 145 focused tests passed after the correction; 93 tests passed after contract metadata integration. Scoped review found no P1/P2 issues. These counts overlap and are not additive.

Right functional-fit candidate is stored separately in `runs/right_functional_fit_20261028/`, with the left seed and source assets unchanged. Offline coupled limits, source hashes, 17 mm initial pad span, positive initial hull/object gap and table clearance were checked. Actual eight-case validation `runs/success_teacher_20261028_173808_right_fit8` completed with zero successful lifts. Right thumb now participates in measured contacts; full-episode right mean loads range 0.56–1.50 N versus approximately 0.10–0.16 N in the preceding original-pose run. This is not a stable grasp certificate. Candidate geometry remains separate from the production default.

## Contact feedback continuity and actual comparison

Trace evidence showed repeated substep load dropouts clearing accumulated explicit effort. Added a 20 ms loss grace to the opt-in contact-effort controller: only already acquired contact may retain its existing integral during a short missing sample, without further integration. Sustained loss clears it; caller safety invalidation and zero closure direction immediately clear both effort and contact history. Initial no-contact produces no effort. Gains, total torque cap, physical guards, rewards and close readiness are unchanged. Selected environment reset clears only that environment's new state. Contract version is `contact_load_integral_effort_v2`.

The same eight-case physical comparison `runs/success_teacher_20261028_174833_right_fit_grace8` completed normally. All eight first episodes failed close; zero lift steps/successes. First-episode effort-clear events decreased from 2028 to 364, and 1794 substep/hand samples retained nonzero effort below the contact acquisition threshold. The maximum continuous duration with BOTH measured hand loads >=1 N remained 0.05417 s, below the unchanged 0.2 s dwell. Seven first episodes ended after exhausted reposition attempts, and one reached the four-second close timeout. The environment continued automatically after early resets, so its total episode count 10 is not the number of independent first-case trials (8).

Final combined command: `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_coordinated_grasp.py tests/test_dh116_lift_grasp_hold.py tests/test_dh116_contact_effort.py tests/test_dh116_measured_readiness.py tests/test_dh116_close_instant_force.py tests/test_dh116_tensor_world.py tests/test_dh116_loaded_palm_motion.py --tb=short` — 186 passed. Scoped continuity review found no P1/P2 issues; `git diff --check` passed.

Remaining problem is physically continuous bilateral grip/reposition behavior. Neither normal-force dwell nor actual success was bypassed. No qualified demonstration, actor imitation training, or new student PPO was launched. Both Isaac comparisons have exited; there is no ongoing monitoring process from these tests.

## Evidence already established

Current initialization opens left pads to 17.0 mm and right pads to 17.6 mm. The calibrated position-closure vectors also become explicit-effort directions: thumb/index effort ratio is 0.903 on the left and 0.464 on the right. These are not symmetric physical commands.

In `success_teacher_20261028_165523_synchronized_effort8`, case z=-0.5 mm / 4 N per side, reclose/idle samples after 0.5 s show left thumb and index contact, whereas right normal forces come only from the index link. This is a measured contact mismatch, not merely different force labels.

Source geometry investigation found that strict reflection of the current left thumb would require right finger12 near -0.119 rad and a different follower relationship, outside the currently supported right-hand coupled model. Exact whole-hand mirroring is therefore not an admissible repair under the existing limits. Use feasible functional alignment and quantify its residual; do not call it exact physical symmetry.
