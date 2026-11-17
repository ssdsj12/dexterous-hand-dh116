# Read-only Teacher display: CPU implementation

## Result

Ten CPU USD tests passed in the clean publication worktree; seventeen passed
with the Ubuntu live-viewer integration. An independent reviewer reproduced the
17-test suite and found no blocking issue. No Isaac simulator was started by
these tests, and no native or grasp qualification is granted.

The owner copies only the original DH116 visual subtrees into a separate display
namespace. Actual left/right instanced USD meshes preserve points and resolved
materials. The display tree has no rigid bodies, colliders, articulations, joints
or physics properties. It follows measured body-link world XYZ and wxyz
quaternions using the existing GitHub visual_pose_sync composition convention.

All preview opinions are in an owned anonymous session sublayer. RootLayer bytes
and the user's edit target are unchanged. Removing that layer restores rendering
without overwriting concurrent user edits. Sync checks the SDK clock is
unchanged. Source/target/hand/env/link and per-link position/rotation errors are
returned for the viewer gate (0.1mm / 0.001rad).

## Earlier failures

The original live viewer displayed stale USD link transforms. Enabling
updateToUsd and requesting native transform export did not resolve the raw
per-link discrepancy. The last bounded check stopped after one control step
and later exited with SIGSEGV (-11), not natural-exit or grasp success.

## Scope and qualification

Only the independent display owner and CPU USD tests are published in this
package. The server 11-16 checkout has unrelated uncommitted Teacher/Student
and style-seed changes, which are excluded. Preview integration remains local
until its source/model dependencies and native result are separately validated.
Display copies are preview-only, not qualified RGB/instance-ID supervision
sources for BC/DAgger.

The 30mm Lift / continuous 5s Hold / 0.5mm Slip and bilateral-contact
requirements remain unchanged. The three frozen Teachers are candidate models;
stage flags and rendered grasps do not establish full success.

## Protection audit

All 528 protected source/script/config SHA values match the locked inventory.
The two historical teacher128 Checkpoint paths named by that inventory are
currently absent; a bounded read-only alternate-location search found no matching
bytes. This pre-existing artifact verification is blocked, not marked passed.
No replacement checkpoint, symlink or zero-filled data was created.
The three current preview Checkpoint SHA values were checked unchanged.

## CPU command

Use the existing SDK USD directory in subprocess-only LD_LIBRARY_PATH and
PYTHONPATH, with CUDA_VISIBLE_DEVICES empty, then run:

    python -B -m pytest -q tests/test_teacher_read_only_display_usd.py

This loads USD geometry APIs, not Kit/AppLauncher/PhysX simulation.
No environment or dependency is installed, upgraded or replaced.

## Next

The approved bounded native viewer check must measure display-only components,
position/quaternion errors, contacts/lift and outer exit status. Native execution,
visual correctness, natural shutdown and true success remain separate evidence.
