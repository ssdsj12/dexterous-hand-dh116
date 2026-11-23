# Shared CUDA mesh Reset and real DH116 continuation

The isolated solver adds reset_env_mesh_poses(env_ids, positions_world_m,
quaternions_xyzw). It reuses native commitSlot to update current, target, solver
pose and World AABB, clearing selected motion/velocity/contact accumulators
without advancing time or altering neighbors. Staged targets, invalid IDs,
malformed poses and non-unit quaternions are rejected. Snapshot v2 exposes
actual CUDA motion, velocities, AABB and pair counts. Physics remains in World.

## Actual native evidence

GPU 3, two environments, four real DH116 articulations: 226 SDK steps, including
216 shared solver exchanges. Original archived 24D approach is issued through
the original action adapter; the neighbor receives zero action. Nonzero reaction
writes: 14/0. At SDK 218 only lane 0's hand, composer, rod and collider are reset.
Current/target/solver poses agree, motion is zero, and neighboring actual hand,
rod, mesh state and pending reaction remain bitwise unchanged. Reset advances
zero SDK or solver steps. Eight subsequent SDK steps complete. Final Reset
counters: 2/1; rod ticks: 96/2592. Natural exit 0 without signal or forced exit;
app.close returns. Outer elapsed: 57.003 s.

Outer SHA: 2e00adb144d16377eb684d932e761aa37b4eaf33cbd048e581c1a2c5c5379c33.
Journal SHA: 7ce731d22238af55d921e5b7484f99f46b0aeebf8fd37aa9bb08f0581021ddb5.
Full argv/frozen sources are in the local attempt.json. Raw evidence stays
outside Git at shared-reset-isaac-gpu3-bb998gcy under the existing
teacher-collision-ordered-fk-20261119 evidence directory.

All three grasps independently pass selected Reset at 128 solver lanes: five
invalid requests rejected; neighbor state/pending reactions bitwise preserved.
These are solver-only cases, not 128 Isaac environments or PPO.

## Build, regression and preserved failures

The source and binary are an isolated candidate; original backends are unchanged.
scripts/patches/deformx_shared_mesh_reset.patch applies after the existing shared
pair patch. Dry-apply output matches all three compiled files. Explicit build:
CUDA Toolkit 12.8.61, architecture 89, PYBIND11_FINDPYTHON=ON, actual DH116 Python
3.10.12. PyTorch stays 2.7.0+cu128; no driver/runtime/Isaac/Python upgrade.
Binary SHA: 19de772f73beb082a1f20e9b2337d2d593bfc6021121db18e4a3e15dfc0ff6bb.

API-absence test failed before implementation. Final CPU: 57 passed, one CUDA
skip, zero failures. Command:
python -m pytest -q -p no:cacheprovider
tests/test_dh116_shared_native_commands.py tests/test_dh116_shared_rod_owner.py
tests/test_dh116_batched_pair_capture.py tests/test_dh116_flexible_wrist_projection.py
tests/test_dh116_deformable_bimanual_step.py

Initial packaging omitted upstream sdf.py; unchanged package files were added.
An initial probe driver duplicated the grasp keyword after one successful GPU
case; corrected driver completed all three. Failed harness logs remain local.

All 528 protected files match inventory. Two historical /tmp checkpoints were
already absent and remain absent; their hashes cannot be verified. No models,
binary, raw logs, or data are uploaded.

## Scope and next work

Mechanical Reset qualifies, not Teacher state/reward/context Reset, grasp
success or PPO. Zero Teacher queries and zero PPO updates. Next integrate the
shared solver parent/row source identity, actual Teacher state/reward/context
and original PPO collector, then requested three-grasp 128/512 Isaac training.
Original Pinch clearance failure remains open. All guards, R2 closure and expert
pool admission are retained. Antigravity R4 agreed on World coordinates and
neighbor pending reaction preservation; its MCP reads an older tree, so current
source excerpts and all native results were independently verified.
