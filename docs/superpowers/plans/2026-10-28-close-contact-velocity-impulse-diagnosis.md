# Close contact velocity impulse diagnosis

**Authorization:** The user requests continuous monitoring and repair of confirmed problems, then explicitly requests finding and fixing the root cause. Normal failures remain isolated resets; severe faults retain the existing hard stop. Initial work added diagnostics; the subsequent controlled experiment verifies a specific clearance-control repair as detailed below. No unrelated files, commits or historical run deletion.

**Evidence:** `tabletop_tensor_20261028_125219_lift_hold_fresh` stopped after iteration 66 at environment 479. Right active finger21 jumped from -0.061184313 to +3.051659822 rad/s in one 480 Hz step; all preceding retained steps were below .26 rad/s. Left hand load jumped 4.068053 to 35.419132 N while right load dropped 1.412759 to .073928 N. Individual-link maximum was21.71775 N. Command slew, lead and palm targets were within current contracts; the event was in close, not lift. The 3.00001 rad/s stop correctly fired before reward/reset. A 2.x rad/s early reset cannot intercept the saved event.

**Design:** Add bounded optional tensor snapshots to the existing32-frame fault ring: force-control state, applied stiffness/commands, object position/orientation/velocity, and existing contact geometry. Preserve ring capacity, copy semantics, selected-env serialization and fatal-guard priority. Do not create sensors or label implicit PD estimates as measured solver torques. Use the latest valid checkpoint in a separate output directory with identical physics/reward/control contracts and retain all source artifacts. Capture actual control actions in a local diagnostic journal to support later controlled replay; do not declare it a replayable simulator state.

- [x] Read saved fault and inspect control/guard paths independently. No evidence-supported command or reset defect found.
- [x] Add failing snapshot/optional-field tests, implement diagnostic fields, verify affected tests and strict guard behavior.
- [x] Review changes; verify source checkpoint and strict unchanged contract.
- [x] Launch one same-contract continuation from iteration66, with richer fault trace and bounded action capture, and repoint current TensorBoard after startup verification.
- [x] Inspect actual runtime results and test a single controller change using identical recorded actions. The signed-clearance repair prevents the captured environment785 failure without resetting that environment. No hard-guard relaxation, fabricated grasp success or universal stability claim.

## Implementation and verification

The trace extension changes only `tabletop_physics_fault_trace.py` and its tests. RED:2 failed/5 passed for missing diagnostic fields; GREEN:7 passed. Independent trace+env regression66 passed, reviewer found no actionable issue. Root command `DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_physics_fault_trace.py tests/test_dh116_tensor_env.py tests/test_dh116_velocity_limits.py tests/test_dh116_lift_grasp_hold.py` passed109 tests; `git diff --check` clean. Additional bounded trace memory is at most58.125MiB for4096 environments.

A local diagnostic-only wrapper `runs/velocity66_diagnostic_tools/action_capture.py` records the exact clamped float32 actions before each TensorEnv step, including a failing step, with a4096-control-step cap. It does not change actions, PPO, random state, physical parameters or contracts; smoke validation checked exact copies, cap and unchanged RNG. The prefix is not a replayable simulator state.

Same-contract continuation launched14:11:10 local time as `runs/tabletop_tensor_20261028_141110_velocity_trace_resume66`, PID1920612. Source is the valid iteration66 checkpoint; SHA256 `601cd936d2f5326500d121814e016bafce89e2b6fbf1f557cef6fb3ec8465e50`, finite policy tensors and13 optimizer state entries. Original failure artifacts remain intact. Runtime validation pending; no physical fix is claimed.

Actual continuation passed startup and strict contract equality. `training_start.json` reports iteration66,6488064 transitions,24770 episodes and13 optimizer entries; `resume_contract_verification.json` records zero differing contract fields. At iteration72 new isolated failures are handled normally and no hard fault trace exists; successful lifts remain0. The action journal contains finite bounded control actions and no recording error. Root continues observation through the initial4-second close window.

Latest actual observation at 2026-10-28T14:18:28.269811+08:00: iteration78, 7667712 cumulative transitions; successes0, hard fault recurred=False. Source checkpoint SHA remains unchanged. Recorded action-prefix status has no error, named TensorBoard and observer match this run. See `runs/tabletop_tensor_20261028_141110_velocity_trace_resume66/velocity_diagnostic_continuation_review.json`. Physical impulse prevention remains unverified; no physics candidate was promoted.

## Continued root-cause repair after the user's explicit request

The continuation subsequently failed after completed iteration89. Environment785,
in lift, reported left passive `finger22` velocity -6.4491672516rad/s. The
existing hard stop fired. The complete action journal contains556 control
steps for all4096 environments (182190080 bytes), SHA256
`3bd71cbdafd75f8e3cf7fddbbd1c24bcee3f964beab202a204c0e4b371bebe48`.

The richer trace establishes a controller defect upstream of this fault:

- All32 retained frames have negative left-hand mesh clearance, approximately
  -0.150 to -0.360mm. Exact authored-USD forward kinematics using actual passive
  joint positions identifies the lowest collider as hard `finger22_link`;
  its compliant child `finger23_link` remains above the table. Computed
  clearance agrees with the measured value within approximately50nm.
- The world projection uses `measured_z - max(clearance-floor, 0)`. When the
  measured clearance is already below the0.3mm floor, it cancels further
  descent but requests no upward correction. Finger motion can therefore
  maintain hard-link/table overlap even while palm exploration is held.
- In the retained lift frames, the common lift offset is frozen at4.0625mm.
  The right-hand load is insufficient for continued lift; holding the accepted
  palm request does not restore clearance lost through finger movement.

The first controlled candidate changes only the safety projection to
`measured_z + floor - clearance`, taking the stricter hand's target for both
hands. It retains common lift coordinates, authored base-height differences,
ordinary exploration slew, force feedback and every physical guard. A static
finger pose gives one stationary absolute safe target even as the driven
wrist catches up; the expression does not accumulate correction from lag.
No solver, material, force-target, reward or reset threshold changes are part
of this candidate. Drive force/speed/transport limits remain active. The full
restorative target is an experimental control correction, not a guarantee
that contact impulses or physical grasp failures are eliminated.

`palm_target_safety` becomes `synchronized_measured_clearance_v2`, explicitly
recording signed-deficit restoration. Formal training rejects old or absent
safety contracts rather than silently migrating them. Any subsequent formal
training must start fresh; the diagnostic replay permits this one declared
contract difference and writes no training checkpoint.

Focused world tests: RED10 failed/17 passed; GREEN27 passed. Additional
contract tests: RED2 failed/1 passed (legacy contracts were silently upgraded).
After removing that migration, root's combined training/world/lift-hold/env/
fault-trace suite passed145 tests. An independent adjacent world/controller
suite passed146 tests. Review and actual paired replay remain necessary.

Baseline replay started14:28:26 as `runs/velocity66_replay_v64`, PID1923789,
with original64/64 solver settings and zero contract differences. At360
completed control steps all153 close-completion events are exactly equal to
the original captured run, including force values and event timings. This
supports reproducibility; final fault comparison is still pending.


Baseline finished at14:42, reproducing the original failure on attempted
control step556 (555 completed), environment785, left passive finger22
-6.4491672516rad/s. All32 retained frames match exactly for joint positions,
velocities, control state, object state and loads. Only two diagnostic centroid
Y values in frame22 differ by1.9073486328125µm; no other fields differ.
`reproduction_verification.json` records this limited reduction-rounding
variation rather than claiming byte-identical JSON.

Candidate `runs/velocity66_replay_clearance_v2` started14:42:30, PID1926273.
Runtime contract comparison confirms exactly one changed key,
`palm_target_safety`;64/64 solver and every other contract remain identical.
The helper tracks environment785 every physics substep, plus minimum hand
clearance and all angular-joint finite-difference motion, excluding resets.
No training checkpoint is produced by either replay.


## Paired physical validation result

Candidate completed all556 recorded controls with all original guards active.
Environment785 stayed in the same episode6 throughout4448 physics samples,
with no pending reset, and remained in phase3 at the original fault frame.
At sequence4442, left finger22 velocity changed from-6.4491672516rad/s to
-0.2180342972rad/s, while left clearance changed from-0.247671331mm to
+0.273415648mm. Its minimum candidate clearance was+0.196915741mm.
The matching32-frame window had positive clearance throughout. This verifies
prevention of the captured failure without evading it through an early reset.
See `runs/velocity66_replay_clearance_v2/paired_validation.json`.

The first restorative action is directly observed: at sequence1789 the
clearance undershot0.3mm by3.275408µm; the next target added that deficit,
and the next measured clearance was0.340351mm. The candidate therefore
intervened near the positive boundary rather than jumping out of the later
large penetration. A separate agent independently reviewed the comparison.

Limits: no successful grasp/lift was observed in this finite replay. The
right-hand load remained weak and the object stayed near table height. This
comparison does not establish the cause of the earlier environment479 fault,
whose saved clearance was positive, or guarantee future numerical stability.
Global angular finite-difference speed reached3.678932rad/s while measured
instantaneous joint speed stayed below the existing hard stop; the available
baseline ring is insufficient to compare that particular finite-difference
peak. No claim that every positional derivative stayed below3rad/s is made.

Reproducible offline geometry audit is in
`runs/velocity66_diagnostic_tools/geometry_audit.py` with JSON/TXT reports;
input assets and both fault archives are SHA256-pinned. It preserves actual
passive joint coordinates and agrees with runtime clearance within56nm.

Fresh formal training launched14:56:19 as
`runs/tabletop_tensor_20261028_145619_clearance_restore_fresh`, PID1927487,
4096 environments and1024 episodes per environment, with no resume option.
Solver settings remain64/64, original reward/force/guard contracts unchanged.
Startup, dashboard routing and early-rollout verification are pending.


Fresh startup verified at 2026-10-28T15:03:39.634047+08:00: iteration13, 1277952 transitions, 4259 completed episodes; successful lifts0. All initial counters and optimizer state entries were zero; no resume source. The only contract difference from the captured run is palm_target_safety. Ordinary failures remain isolated resets across the complete4-second close window; hard fault trace absent and observer problems/warnings empty. TensorBoard6009 exposes only the fresh run; the existing VSCode TensorBoard tab was visually verified displaying its name. Detailed evidence is in the run's clearance_fix_startup_verification.json. Training and its read-only observer remain running.
