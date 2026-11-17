# Read-only Teacher display: CPU and native evidence

## Result

The independent display owner was published as b5fbc3efb07c635894b184e2c77371a5587e415c
(parent 86a3a88a2c750a27380b14c45f31d42a0dd653f8). Windows verified the ten-file
commit, normal push and remote content readback; GitHub PR #2 remained Draft.
This metadata attestation changes only this report, readiness, Registry and
ReleaseManifest. It starts or stops no Isaac process and excludes the separate
support and Power palm-wrap development work.

Ten CPU USD tests passed in the clean publication worktree; seventeen passed
with the Ubuntu live-viewer integration. An independent reviewer reproduced the
17-test suite and found no blocking issue. These are CPU tests, not native runs.

## Native display qualification

The separately authorized run at
/mnt/mydisk/xk/dh116/runs/teacher_visual_copy_20261117T043644Z
completed 480 control steps across three environments with zero PPO updates.
Its display_no_physics.json records 826 prims, 102 display links, zero rigid
bodies, zero colliders and zero joints. Both saved SDK/display audits record
unchanged physics clocks and position error 0 m. First/final maximum rotation
errors were 2.7957056521316965e-07 / 2.1900144899933154e-07 rad, below the
0.001 rad gate; the position gate remains 0.1 mm.
The final physics clock was [3842, 8.004167084116489].

Windows WebRTC observation, reported by the parent task, confirmed Pinch and
Tripod following the physical hands. switches.jsonl also records button-driven
mode changes; it does not certify Power palm contact or grasp success.
The native display is qualified only for visual copies following measured SDK
body-link XYZ and wxyz poses without adding physics components.

## Scope and remaining qualifications

The owner copies original instanced visual subtrees, preserving mesh points and
resolved materials. Every preview pose and rendering-visibility opinion lives
in its removable anonymous session sublayer; source asset RootLayer bytes and
the user's edit target are unchanged. The existing published visual_pose_sync
coordinate algorithm writes only display copies. Original articulation,
controllers, checkpoints and DeformX physics remain outside this module.

Final net minimum lifts were Pinch 36.00853729248047 mm, Tripod
34.182369232177734 mm and Power 35.18328094482422 mm. This was approximately
eight seconds of short-episode playback. It did not verify continuous 5 s Hold,
0.5 mm Slip or the complete bilateral-contact acceptance contract.
Teacher grasp qualification therefore remains false. Current Power uses five
fingertip meshes and has no palm collision; full Power/palm-wrap qualification
remains false and is separate work.

At the parent handoff, PID 3455794 retained the final frame without new control
steps. Natural process exit was not observed or qualified. Stereo RGB/instance-ID
supervision and training-data admission remain unqualified. No successful
demonstration or training pool admission is granted.

Only the independent owner and CPU tests were published as implementation.
The local live viewer depends on other uncommitted Teacher/Student/style-seed
work, excluded from this package. The earlier stale-USD and SIGSEGV failures
remain historical failures and are not relabeled by this successful display check.

## Protection and publication attestation

All 528 protected source/script/config SHA values were rechecked against the
locked inventory and match. The owner/test SHA values match both the published
worktree and the native run's live checkout. The three current preview model
SHA values match their frozen configuration and ready.json.
The two historical teacher128 Checkpoint paths named by the protection inventory
remain absent; the earlier bounded read-only alternate-location search found
no matching bytes. Historical checkpoint verification remains blocked. No
replacement, symlink or substitute bytes were created.

Readiness and ReleaseManifest retain the immutable saved native evidence SHA
values, measured errors, frozen model identities and qualification boundaries.
The progressing viewer log is treated as a snapshot rather than an immutable
finished-run artifact. The metadata commit is published by the same Windows
bare-relay route; its exact remote SHA is recorded in the external publication
receipt and the final task response, avoiding a self-referential commit hash.

## CPU command

Use the installed SDK USD extension in subprocess-only LD_LIBRARY_PATH and
PYTHONPATH, CUDA_VISIBLE_DEVICES empty and PYTHONDONTWRITEBYTECODE=1:

    python -B -m pytest -q -p no:cacheprovider tests/test_teacher_read_only_display_usd.py

This loads USD geometry APIs without Kit/AppLauncher/PhysX simulation.
No environment, dependency, module, test, asset or checkpoint is changed.
