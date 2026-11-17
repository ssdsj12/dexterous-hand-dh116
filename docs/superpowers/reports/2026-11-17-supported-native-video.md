# Failed contact/Lift diagnostics from supported native motion caches

## Result and published scope

The user watched the Windows compilation and explicitly reported: “根本没抓住，
只是灵巧手本身往上走”. These are failed contact/Lift diagnostic videos, not
successful Teacher demonstrations. Hand motion is not rod grasping, and the
Power21mm reading is not evidence of sustained grip. The existing cache and video
bytes are preserved; per-finger load and hand/rod relative-motion analysis is a
separate cache-only diagnostic work package.

Three complete native motion caches and three MP4s, plus a sequential compilation,
were produced from the same cohort. Four videos were decoded and saved under
D:/视频/DH116_20261117T052535Z with independently rechecked Windows SHA256 values.
Video artifacts are complete; natural process exit and full grasp success remain
failed/unqualified. Finger12 event windows, Power palm contact and Stereo/BC/DAgger
training-data admission remain unqualified.

This work package publishes exactly six code/test paths plus report/readiness,
Registry, ReleaseManifest, README and CHANGELOG. It excludes local native drivers,
supported runtime, uncommitted trainers/style recipes, runs, logs and model files.

## Code review and CPU verification

A read-only review found no important issue blocking this scoped publication.
audited_link_poses rejects duplicate/changing link identities and invalid XYZ/wxyz.
The CPU preparer copies original visual geometry and strips physics from the new
render-only scene. All 1191 saved scene prims were independently inspected on CPU:
no residual physics/PhysX properties were found. The renderer uses no
SimulationContext, solver or Teacher, keeps timeline playback disabled, checks the
source ledger/scene/recording hashes and source clocks, and captures bounded
render-only updates. The PNG helper now waits up to three seconds for background
disk writes after the screenshot future completes; its delayed-writer tests pass.

The publication worktree passed 74 tests, zero failures/skips, in 4.75s:
end supports, recording, video capture/encoding, delivery labels, cached replay and
read-only USD display. The existing installed USD and ropesim.sdf were used without
dependency changes. A first probe omitted the ropesim Python path and reported
71 passed/3 skipped; rerunning with the existing path exercised all74 tests.

Command (subprocess-only USD LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES empty):
PYTHONPATH=<installed omni.usd.libs>:/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python:src
python -B -m pytest -q -p no:cacheprovider tests/test_deformx_end_supported_scene.py tests/test_teacher_preview_recording.py tests/test_teacher_video.py tests/test_teacher_video_delivery.py tests/test_teacher_cached_replay.py tests/test_teacher_read_only_display_usd.py

## Preserved native failures and actual cache evidence

Native attempt1 at runs/teacher_supported_video_20261117T052012Z completed60
zero-action settling controls, SDK tick482, then failed during baseline reset
outside InferenceMode. It produced no Teacher control/video and independently
SIGSEGV'd during shutdown. This earlier failure remains a failure.

Separately authorized native v2 at
runs/teacher_supported_video_20261117T052535Z completed60 settling plus480 Teacher
controls: total540 controls and SDK tick4322 at9.004167136270553s, zero PPO updates.
Each Pinch/Tripod/Power cache has481 source frames including the initial state.
The first cached frame is SDK tick482 after settling; initialization accounts for
the two SDK ticks preceding540 eight-substep controls.

After caching, the original native screenshot helper raised
"viewport did not write a frame" before the asynchronous writer produced PNG.
The actual first PNG later appeared. Native outer_exit.json records GDB255 and
natural_exit_passed=false; shutdown SIGSEGV is not relabeled as success.
No Teacher, DeformX or physical step was rerun to recover the videos.

All481 native ledger frames contain the same102 env/hand/link identities.
The earlier independent manifest verified49062 named records against the NPZ
poses, with exact XYZ and normalized-wxyz equality. Rendering consumes those
named records directly and does not invent a23-body/wrist index mapping.

## Cached rendering, videos and shutdown

Earlier render_only_v1 also failed on the asynchronous PNG write and exited
GDB255. The completed render_only_v2 uses the bounded background-write wait and
produced videos_complete.json after11567 render updates, with zero new physical
steps, zero solver steps and zero Teacher queries. Its current authorized batch
attempt1 completed artifact generation; attempt2 was not used by this package.

Each style selects source frames0,2,...,480 without interpolation:241 video
frames,1280x720,30fps,8.033333333333333s. source_frames.json preserves the original
policy/SDK clocks and recording identity. MP4 SHA values and full FFmpeg decoding
were checked independently. The compilation is Pinch then Tripod then Power:
723 frames,24.1s; delivery_verified.json confirms full decoding and source hashes.

Actual final rendering exit is also GDB255/natural_exit_passed=false, with
SIGSEGV in the saved log and no forced cleanup. videos_complete.json certifies
completed artifact generation; it does not certify natural shutdown.

## Measured grasp limitations

Pinch baseline-relative height peak/final:6.45032525062561mm.
Tripod height peak:0.010669231414794922mm, final:-0.0006854534149169922mm.
Neither has a qualified support-cleared Lift (the delivered value remains null).
Power support-cleared peak/final Lift is21.008789539337158mm, below30mm, and its
bilateral palm/proximal contact qualification remains false. Five pad contacts
do not establish palm enclosure. No complete30mm/continuous5s/0.5mm Slip/
bilateral-contact/no-support grasp or Finger12 event-window success is granted.
The user-confirmed failure classification overrides any informal interpretation of the
video filenames or hand motion. The annotations explicitly say candidate/native replay and full-success NOT
qualified. These caches/videos do not enter any successful demonstration or
Stereo training pool.

## Protection and release boundaries

All528 protected source/script/config SHA values and the three frozen model
hashes match. Six published code/test SHA values match the executed live checkout.
Two historical teacher128 checkpoint paths in the protection inventory remain
missing; verification stays blocked, with no replacement bytes or symlinks.

Source/evidence/video hashes, actual exit JSON, qualification flags and local
delivery status are recorded in readiness and release metadata. Prior component
commit c8d3629379a5b46f4189ef0fd21bc5f49d60e1ae was read back from GitHub while
PR2 remained Draft. The containing new commit is imported/pushed through the
Windows bare relay; its exact verified remote SHA belongs to the external
publication receipt and final task response. No merge or force push is allowed.
