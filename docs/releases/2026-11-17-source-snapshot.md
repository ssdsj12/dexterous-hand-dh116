# 11-17 source snapshot

This branch freezes current Ubuntu runtime sources plus published diagnostics. It is not a trained Teacher release.

## Provenance and integration
- Published development base: ccd87fd282901f05387a566d254bca7850c66a4d.
- Ubuntu checkout:11-16 at e85a3ef032ef5316cb947372ab6d7dcfbc60bd7d, with current local Teacher/Student source changes.
- Source paths and exact byte SHA256 are in docs/contracts/dh116_11_17_source_snapshot.json.
- Imported19 current runtime/config/test paths and9 local preview/seed/wiring paths by explicit allowlist. Byte-compared each to Ubuntu; original checkout and index are unchanged.
- Intentionally retain809 historical paths deleted on the divergent11-16 branch. No delete, merge reset or force push is part of this snapshot.
- Exclude logs, streamed telemetry, runs, model binaries, raw image/trajectory data, caches, local launch helpers and unrelated user files. Frozen model identities are hashes, not model uploads.
-11-10 and codex/dh116-stereo-privileged-dagger remain unchanged. PR#2 remains Draft.

## Verification
87 targeted CPU regressions passed,0failed,0skipped. This includes actual offline USD/SDF tests, recording integrity, delayed writer regression, named-link replay, controller source extraction and curriculum-unit tests. Every imported Python file compiles; every imported JSON parses.
Command: LD_LIBRARY_PATH=<installed omni.usd.libs>/bin PYTHONPATH=<installed omni.usd.libs>:<existing teacher-pair-build>/python:src PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_deformx_end_supported_scene.py tests/test_teacher_preview_recording.py tests/test_teacher_video.py tests/test_teacher_video_delivery.py tests/test_teacher_cached_replay.py tests/test_teacher_read_only_display_usd.py tests/test_teacher_grasp_preview_runtime.py tests/test_teacher_supported_runtime.py tests/test_dh116_curriculum_and_fixes.py
No Isaac instance, physics step, solver step or PPO update was started for this release.
528 protected original-source hashes match; all three existing candidate model hashes match. Snapshot comparison details are explicit in the manifest. Two previously missing historical checkpoint paths remain blocked, with no replacement invented.

## Actual qualification and known gaps
Current Teacher candidates are NOT grasp-qualified. In the latest end-supported cohort Pinch palms move~80mm while fixed rod nodes move~2mm; Tripod palms move~67mm while the rod is nearly stationary. Neither has observed bilateral pad-load threshold samples or full support clearance. Power reaches21.009mm off-support height but is not a qualified palm-enclosing, five-finger fist grasp.
The current GPU candidate uses67D privileged Teacher input, while this local Student path uses fixed stereo RGB plus39D proprioception. This snapshot does not claim the final equal-dimensional Teacher/Student task-state contract or successful distillation.
The Student source creates StereoLeft/StereoRight; the separate Teacher preview/video source uses an observer viewport, without paired RGB sensor display. A rendering-enable flag alone is not a stereo camera.
flexible_train_curriculum_stage1 contains10mm/0.5s/2mm candidate shaping thresholds. They are not the formal native success standard and must never be used to qualify expert data. Formal acceptance stays net Lift>=30mm,continuous Hold>=5s,Slip<=0.5mm,effective bilateral opposed contact,and no external support/attachment.
Native shutdown SIGSEGV remains unresolved. CPU passing and complete video export do not grant natural-exit, contact, Teacher or Student qualification.

## Resume work
Prioritize Teacher: establish opposed contact, fixed-wrist stable retention, then support-cleared Lift. Implement proper Power palm/finger enclosure with real qualified collision surfaces. Preserve historical PPO/checkpoints and use independent versions. Do not enter BC/DAgger success pools from these failed candidates.
