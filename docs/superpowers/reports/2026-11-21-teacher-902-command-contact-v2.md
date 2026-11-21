# Source-bound 902 Tripod command contact v2

Baseline: cfbc2147e66baa7a0688ccda562db54c3b921149, Ubuntu branch 11-17.

## Actual change
The historical v1 remains capped at 24 control windows. A separate opt-in v2 route uses the sealed real Tripod SDK408 boundary, unchanged right-hand terminal pose and original 24D command kernels. Its 66 windows contain 528 physical command substeps. Every substep checks BOTH hands, each with 17 source collision shapes and five cooked pads. This is an ideal command-tracking prediction, not a replacement for measured native geometry.

- scripts/screen_dh116_902_command_contact.py::prepare executes the real float32 finger/wrist target submissions through command_step, rather than a pose Slerp shortcut. Original joint bounds, safety projections and action ranges remain.
- apply_schedule delays right index closure by 16 windows and applies temporary index/middle relief of 0.001/0.0005 rad, returning to the original terminal joint values. Left-hand commands remain exactly zero.
- quantized_window_commands preserves each double-precision joint waypoint under the existing repeated float32 target writes. Correction is confined to right-hand action generation and bounded to 1e-6 rad per control window. It cannot compensate physical tracking error or change the controller.
- support_bound uses the existing full convex-hull/box separating axes when the conservative AABB bound is insufficient. It does not alter support geometry or physical margins. The v2 CPU candidate additionally requires 1 mm support reserve.
- The existing complete nine-frame rod envelope, SDF labels, 0.150 mm contact limit, 0.250 mm noncontact limit and original table clearance are retained. Surface points do not become material nodes.
- load_stage verifies request, archive, source and exact action SHAs; independent_entry checks same-clock native boundary, both hand reports, all 528 samples, station identities and cooked hulls. Existing native entry recomputes the whole CPU proof in an isolated process before Kit and requires exact equality.

## Failure preserved and correction verified
The first full CPU run exited 1 after 74.61 s. Geometry checks passed, but repeated float32 commands caused 1.141368e-5 rad terminal error in the right middle-finger channel, exceeding the unchanged 1e-5 threshold. Its proof SHA is 82a1867df9bb8bab5f2ffcad418ce8b66c055aa8084d1600444b34b7a7d1ee51.

After bounded numerical command compilation, the full run exited 0 after 73.02 s:
- 528 command substeps / 1056 hand observations, SDK408→936 prediction.
- Joint terminal error: 2.162143e-7 rad; translation: 1.141517e-8 m; rotation: 7.207939e-8 rad.
- Minimum full-envelope contact lower bound: -0.149920701 mm (threshold remains -0.150 mm).
- Right noncontact lower bound: 2.622141 mm; left: 1.118115 mm.
- Minimum support lower bound: right 1.387092 mm, left 25.523286 mm.
- Terminal original SDF pad gaps: -0.029024, -0.021571, -0.033164 mm.
- Proof SHA: 1454e120b4a4f94d1b98b6f953c2fe39bdbd91800de036d5cdc80d6dded043c4.

These contact margins are small, and ideal tracking is not a real-contact certificate. Actual tracking, dynamics, reaction forces, hold and lift require native measurement. Neither CPU run advances Isaac or trains a model.

## Tests and preservation
191 related CPU tests pass, including 20 v2 tests. Regression log SHA: d140adee08f0ebd963ec1eeed3af8d2dd01b430d0f931ff9af264b08d09743ee. Negative cases cover missing substeps, wrong hand/clock, stale source/request/action seals, mid-path collision, unqualified success flags and invalid float32 compensation. Fresh-process imports retain the USD-before-Kit boundary. Registry result is recorded in readiness.

All 528 protected files, three historical checkpoints and both fixed RGB configurations are unchanged. Raw geometry, native receipts, proof arrays and logs remain outside Git.

## Antigravity review
The shared full-envelope design was reviewed in earlier R1/R2. Final R3 was quota-blocked; a later stale R3 request was cancelled after no new reply. A new 902-command-contact-v2 R1 request has been sent to the configured codex-dh116-ssh-reviewer against the current live Ubuntu file SHA. Its result remains pending and is not an approval. Codex independently checked numerical behavior and ran the listed tests.

## Remaining native work
No new native run, Teacher query, PPO update or successful grasp in this package. Next freeze one bounded Tripod right-contact experiment: preserve original 240-step settlement and 21-window approach, then 66-window contact; maximum SDK936, 11208 solver substeps, 87 control windows, one environment, zero PPO. Record actual source/cooked guards and reaction force and require natural outer exit. Stop at any original guard violation. This is contact evidence only, not 5 s hold or Lift admission.

Pinch still requires force-loaded fixed-wrist hold; Power requires palm plus four-finger enclosure and opposed thumb. Both stereo camera configs remain fixed. All three must separately meet original support-free 30 mm / 5 s / 0.5 mm success criteria.
