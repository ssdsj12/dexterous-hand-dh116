# Dual mounted SDK action/readback boundary: CPU-only audit

Date: 2026-11-13. Base f76e1214e6fdc46006761795bedc6ba7e0ab589a.
Only this report and one new test module are added. No frozen runtime, table
contact source, controller, geometry, native recipe or physics parameter changed.
No Isaac, native attempt, GPU solver, training or publishing was performed.

## Actual archived input and first unresolved native blocker

The v3 attempt 90d75220592f6551a389f5c9d8bbab55624492821115e41e05660a183307c943/
attempt-n2pwu0f3 completed two setup steps and reset, then exited outer/inferior 1
without a signal at open_bimanual_table_contacts: actual original table collider
required. Thus this package does not claim a clean 2+8 native result.
The table hierarchy correction belongs to a separate package.

The test pins and reads the actual v3 physical.jsonl at:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-zero-v3-20261113-aYoTfF/native-runs/flex.f7-dual-pinch-zero-interval-v3-native/attempt-n2pwu0f3/native/physical.jsonl
SHA256 36cff198922a48525cedfcbf5bd68c3fff7d2cdbbe44eec2ca1ba9c948bb0168.

The archived complete reset supplies both actual joint/body orders, 17 DOF,
23 bodies, distinct /leftHand/wrist_fixed and /rightHand/wrist_fixed paths,
measured q/qd, target arrays, root/link transforms, and integer native DOF types.
The single rod/reset/shared clock and original reset seed are retained.
The archive does not contain link COM velocity or SDK limit buffers; their zero
velocity and finite limit arrays in this test are explicitly CPU shape fixtures,
not reconstructions or claims about the original native values.

## Installed SDK boundary executed on CPU

The new test extracts the unchanged installed Python method ASTs only for test
execution. It executes ArticulationData joint/link/root/COM-velocity cache
properties and timestamp update, the actual tensor API getters, original
convert_quat, and Articulation.set_joint_position_target. CPU tensor allocation,
backend copy endpoints, timestamp containers and the write flush are doubles.
There is no SDK class constructor, app, native import or physical step.

This is more specific than manually assigning cached body poses: the real
installed cache cloning/aliasing, XYZW-to-WXYZ conversion, buffer shape and
getter behavior are exercised before calling the unchanged capture_mounted_hand,
adapt_mounted_hands and BimanualStepConnection begin/submit/finish. The source
files and archive have exact SHA assertions in the test.

The nine cases cover:
- Native-order mounted observations retain 66 valid and 305 unknown channels.
- Installed getters allocate (1,17) DOF types and (1,17,2) limit buffers.
- Zero and +/-0.001 normalized thumb-master actions use original target kernels.
- Original target setter plus CPU flush changes backend targets while measured
  joints remain unchanged: submission is not measured tracking or contact.
- Stale cloned body pose, invalid DOF type, wrong body path and nonfinite
  synthetic velocity fail closed without a target write.
- Out-of-range normalized action fails before target writes.

Counter updates in finish tests simulate one completed boundary only. They do
not count as SDK or rod execution. The tested getter/observation/target boundary
has no additional reproducible blocker under these explicitly limited inputs.

## Verification

With DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python,
DH116_NUM_THREADS=2, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1:

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_mounted_sdk_boundary.py tests/test_dh116_teacher_mounted_wrist_source.py tests/test_dh116_deformable_bimanual_step.py tests/test_dh116_flexible_physical_exchange.py

68 passed in 4.59 s (9 new, 59 existing), exit 0.
Evidence: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-mounted-sdk-boundary-cpu-ldy7i8wh.
The earlier focused run was 9 passed in 3.15 s.
Repository.verify_protection() passes (528+2 protected files). All preexisting tracked files are unchanged.
New test SHA256: 0afbbd2d1535b1c20dd4c38ea73a27a3342a64c92d2f7c6a44732315dbc2d9a6.

## Existing route for a separately bounded nontraining action

The reusable scheduling boundary is DualHandPhysicalEpisode.pre_action_state()
then control(action) in tabletop_deformable_dual_hand_episode.py:85/114.
FlexiblePhysicalExchange.control at :162 validates one finite normalized (1,24)
action, checks whole-interval remaining budget, and holds that owned action over
exactly eight 480 Hz ticks unless a terminal guard ends the episode. It calls the
existing BimanualStepConnection begin/submit/finish and MountView, preserving the
original policy-owned finger and wrist kernels. No force action is introduced.

Canonical order: left six fingers 0:6, right six 6:12, left/right world XYZ
12:18, left/right intrinsic XYZ 18:24. ACTIVE order is finger11, finger12,
finger21, finger31, finger41, finger51. Contract maximum rates are 0.3 rad/s,
0.01 m/s and 0.5 rad/s. Over eight ticks their unguarded magnitude bounds are
0.005 rad, 0.000166667 m and 0.00833333 rad respectively; these are request bounds,
not achieved motion or clearance guarantees. A future no-lift action schedule
must separately pin exact action values/directions and interval count; no such
native schedule is created here.

Current prepare_dh116_dual_pinch_interval.py:46 constructs exactly zero action;
its inspect_interval:93 rejects nonzero actions. A new bounded operation and
matching independent audit are required for any nonzero experiment. Rebinding
that old zero-only validator to accept nonzero would invalidate its scope.

## Safety/evidence gaps before closure

Even a clean zero-action 2+8 run would establish the measured execution and
ownership chain only. It would not establish dual contact or Teacher readiness.
Static hand-table readback remains NA, original 371D state remains incomplete,
and source envelopes/historical locked pads are not proof of native attachment.

A concrete additional gap is in FlexiblePhysicalExchange._step:147: after-SDK
geometry is read for pad poses and COM, but its full-hand measured clearance is
not included in the returned physical row or checked by _guard_failure (:93).
That guard checks joint/mimic state and per-pad 30 N only. The existing 2.1 mm
wrist projection is applied to the preceding boundary and is not a post-step
clearance termination condition; its floor field alone is not a measurement.
Before nonzero closure, persist and check same-boundary post-step source-envelope
clearance (still explicitly distinct from native contact/cooked geometry).
Do not reduce the 2.1 mm floor or infer safety from empty static contact buffers.

The previously observed source-envelope floor margins are only about 10.8 um
left and 16.3 um right. Choosing positive thumb/index actions from action indices
alone is not a proof of a safe closing direction. Native hand pairs, per-link
loads and signed motion must be measured under a separately frozen bounded
diagnostic. same_rod_contact_qualified, grasp and Teacher remain false.
