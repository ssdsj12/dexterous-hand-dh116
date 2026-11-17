# End supports and complete native-video recording components

This package publishes independently testable components, not new Teacher training or a success qualification.

## Implementation
- `plan_flexible_end_supported_scene`: independent Z translation of seed copies and two narrow end supports. Original joints, rotations, table and hand materials are retained.
- `bind_native_end_supports`: preserve original mesh slots 0..10, append support slots11/12, real box SDF and independent PhysX materials with matched friction. Native PhysX registration remains an execution check.
- `support_load_records/evaluate_support_clearance`: actual support wrench slots and full-segment clearance, stale/invalid sources remain NA.
- `NativePreviewRecorder`: immutable CPU snapshots, strict run/episode/env/model identity and monotonic source clocks, SHA-checked NPZ without pickle.
- `capture_frame/encode_frames`: render-only frozen-clock guard, bounded updates, complete PNG sequence, existing FFmpeg, overwrite refusal.
- `measured_annotation`: baseline-relative height is separate from support-cleared net Lift. Supported/unknown values remain NA; video never qualifies a grasp.

## CPU evidence
59 tests cover these four components: 15 end-support USD/SDF,34 recording,6 capture/encoder,4 delivery labels. The deployed runtime/display regression totals79 passed with existing installed USD and actual ropesim.sdf. The encoder fixture is explicitly synthetic CPU data, not native evidence.

Command: LD_LIBRARY_PATH=<installed omni.usd.libs>/bin PYTHONPATH=<installed omni.usd.libs>:<existing teacher-pair-build>/python:src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_deformx_end_supported_scene.py tests/test_teacher_preview_recording.py tests/test_teacher_video.py tests/test_teacher_video_delivery.py

## Native attempt1: preserved failure
Local evidence: runs/teacher_supported_video_20261117T052012Z. The run completed60 zero-action settling controls, last SDK tick482 at1.004166719s. Actual DeformX support wrench norms at the final sample were0.596367/0.320555/0.325091 N for Pinch/Tripod/Power. This does not qualify PhysX-side contact identity or full grasp success.
The next baseline callback failed on an inference tensor in-place update outside InferenceMode. No Teacher control or full video was produced. Real-tensor CPU regression reproduced the exception and passed after adding the context. Natural shutdown independently SIGSEGV'd in the livestream present thread; GDB stack and library map remain in server logs. No signal kill or forced exit was used.
A separate v2 single-run budget is user-approved; its output is pending and must not be inferred from CPU tests.

## Protection and boundaries
528 protected source hashes and all three candidate Teacher weights match. Two old protection-inventory checkpoint paths were already missing and remain blocked. No historical assets, joints, physics parameters, model bytes or current uncommitted trainer changes are published by this package.
The local preview driver wraps a SHA-bound, uncommitted deployment source and private style seeds; those are excluded pending a separate source review. Public components do not imply a portable, qualified Teacher training environment.
Power requires bilateral palm/proximal collision and five-finger enclosure evidence; five pad contacts do not satisfy it. Full success remains30mm/5s/0.5mm plus bilateral grasp and no external support. No Student, BC or PPO success is claimed.

## Next
Finish the separately approved video attempt, verify actual frame sequence/identity, transfer only final videos and provenance to D:/视频, and report real shutdown status. Proper Power palm geometry and full grasp validation remain separate tasks.
