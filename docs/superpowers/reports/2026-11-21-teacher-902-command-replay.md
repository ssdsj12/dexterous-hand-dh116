# 902 controller-command replay and rejected approach timing

The earlier geometric route used measured joint positions and quaternion Slerp. The real controller integrates issued backend finger targets and seed-relative intrinsic XYZ angles, with table projection and float32 SDK writes. A valid end pose does not prove that intervening commands are safe.

## Implementation and independent review

Added scripts/screen_dh116_902_command_path.py. command_step reuses the original finger_pre_guard, surface_final_guard, wrist_requested_targets, project_policy_wrist_targets and local_translation functions. It requires measured q, backend targets, reset roots, current roots, table clearance, integration state and mount anchors. Carriage limits are checked before returning targets. No new controller, safety threshold, dynamics model or action contract is introduced.

The scope is one live policy-owned boundary with zero common Lift offset. Native safety, PD tracking, mimic behavior, contact forces, moving rod and stopping conditions remain the native owner's responsibility. This tool does not grant native or Teacher admission.

audit_boundary loads the existing sealed natural-exit boundary and checks eight consecutive physical transitions for each grasp. replay_errors compares finger requests, actual float32 finger targets, requested wrist positions/angles, projected roots/angles/positions and actual float32 wrist targets. Raw-file SHA and decoded-content digest are separate identities.

Antigravity read the same Ubuntu worktree at d0a71679 and matched source SHA. R2 identified missing carriage limits and projected-state comparisons. Codex adopted both and added actual wrist-write checks. R3 corrected the suggestion that wrist-angle limits were missing: they already exist in the shared projection. It also agreed that nominal kinematics cannot certify real joint tracking. R1 encountered model unavailability; no review result is attributed to it.

## Verification

- 60 related CPU tests passed, 3.43 seconds; red tests and original failed diagnostics retained.
- Pinch archived SDK 425–432: eight physical transitions, all eight command/projection comparison fields have maximum error 0.
- Tripod archived SDK 401–408: same result.
- These are CPU replays of existing native records, not new Isaac runs.
- Both fixed 640x480 camera profile hashes remain unchanged.

The replay reports bind their source files and original native receipts; hashes are in readiness. An earlier temporary Pinch path typo failed FileNotFoundError before reading data and was corrected to the existing approach-native-pinch receipt, without fabricating a new run.

## Actual rejected candidates

Future-state comparisons explicitly assume ideal target tracking after their first measured input and a fixed archived rod. They preserve actual finger float32 writes, but their geometric wrist poses use float64 projected targets; they are rejection diagnostics, not a qualified prediction of PhysX motion.

| Timing | Minimum continuous lower bound, mm | Result |
| --- | ---: | --- |
| Simultaneous, 66 control windows | -0.266130 | Reject |
| Index delayed 16 windows | -0.263340 | Reject |
| Index delayed 32 windows | -0.263340 | Reject |
| Wrist first | -4.025896 | Reject |
| Fingers first | -0.862299 | Reject |
| Rotation first | -7.136872 | Reject |
| Translation first | -9.300886 | Reject |

The first three check all 528 nominal substeps per candidate and took 114.32 seconds total. The last four use 132 windows / 1,056 nominal substeps and geometry every eight substeps; sparse screening can reject a candidate but cannot certify its full path. Translation first also gives a -3.680945 mm support lower bound. These are conservative gap lower bounds, not exact penetration depths or actual native contacts. Existing 0.15 mm contact penetration, 0.25 mm noncontact and all support/table standards remain unchanged.

The first phase script stopped at all-NaN SDF values during rotation-first. Its two completed failures and traceback remain. A separate continuation ran only rotation-first and translation-first, marking 17 and 46 SDF samples unknown; it did not zero-fill or treat unknown as clear.

A prior command comparison hit the finger12 lower bound after float32 writes: 0.007912621833384037 versus original lower limit 0.007912622218025698 rad. Preserve the failed candidate. Only the new candidate goal received 0.0001 rad inward margin; neither joint bounds nor protection changed.

## Reproduction

Use the existing Ubuntu CPU environment from docs/contracts/dh116_wire_joint_pregrasp_v1.json:

    python -B -m pytest -q tests/test_dh116_902_command_path.py tests/test_dh116_902_flexible_contact.py tests/test_dh116_deformable_bimanual_step.py tests/test_dh116_tripod_supported_reset.py
    python -B -m scripts.screen_dh116_902_command_path --receipt "$PINCH_APPROACH_OUTER" --grasp pinch --output "$FRESH/pinch.json"
    python -B -m scripts.screen_dh116_902_command_path --receipt "$TRIPOD_SAFE_OUTER" --grasp tripod --output "$FRESH/tripod.json"

Comparison scripts and full evidence remain in the private ref902 evidence directory. Readiness records their output hashes. No raw assets, checkpoints, videos or logs are uploaded.

## Remaining work

Move wrist alignment outside the rod envelope before final fixed-wrist closure, and screen the complete issued-command route. A valid path would still need native tracking, continuous geometry, opposed load and reaction delivery verification. Pinch, Tripod and Power remain separately unqualified for full grasp success; Power still needs its palm/fist envelope qualification. Both cameras, flexible-wire physics and 24D/371D contracts remain fixed. There were zero new native runs, RGB captures, Teacher queries and PPO updates in this package. No change to Lift >=30 mm, Hold >=5 s or Slip <=0.5 mm.

Registry regression: 60 passed in 15.34 seconds, exit 0.
