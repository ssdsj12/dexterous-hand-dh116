# DH116 Contact Coordinate Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the user's selected Inline Execution. Steps use checkbox syntax. Do not dispatch subagents.

**Goal:** Build and test the coordinate/point-Jacobian primitives needed before selecting real DH116 pad surfaces.

**Architecture:** Keep pure geometry separate from USD extraction and physical validation. Consume existing DH116 coupling by name, preserve seven Panda columns, and expose explicit world-frame Jacobian origins. This is D1/D2 foundation only, not complete pad calibration or a grasp qualification.

**Tech Stack:** Python 3.11, NumPy, pytest; existing pxr-enabled environment for the unchanged USD regression.

**Execution checkpoint (2026-10-15):** Tasks 1–3 implemented/reviewed in the working tree. Baseline52, new39, combined91 tests passed without skips. Actual USD instance-proxy geometry inspected read-only; real pad selection and PhysX remain open. Code not committed because this task consumes existing untracked DH116 foundation; no blanket staging. See [evidence](../../../notes/log/2026-10-15-dh116-contact-coordinate-core.md).

## Global Constraints

- Follow [approved design](../specs/2026-10-15-dh116-pad-jacobian-wire-grasp-design.md); upper pinch first, sliding deferred.
- Existing linked worktree, preserve unrelated dirty work, no O6 changes.
- CPU core0, OMP/BLAS/MKL one thread; no physics launched in this numerical task.
- No invented pad offsets, force limits, or physics pass claims.
- D1 actual source-surface selection, D2 actual PhysX validation, D3 table/wire and D4 grasp remain separate sequential integration tasks after this foundation. Their acceptance criteria remain those in the approved design, not waived by this plan.

## File structure and interfaces

Create `Go2Pvcnn/go2_pvcnn/control/dh116/contact_frames.py` with:

- `surface_reference(triangle, weights) -> (point, normal)`, finite (3,3) triangle in body-local metres, length3 nonnegative unit-sum weights. Normal follows supplied winding; no claim this alone proves outwardness.
- `palm_frame(origin, outward, wrist_to_middle) -> (4,4)`, column-vector rigid transform in base_link coordinates, +Z outward, +Y projected wrist-to-middle, +X=Y×Z; reject zero/parallel direction.
- `transform_reference(transform, point, normal) -> (point, normal)`, validates proper rigid transform and unit normal.
- `point_jacobian(spatial_jacobian, offset_world, joint_names, side) -> (3,13)`, input finite (6,18), rows linear then angular, columns exactly named Panda7+DH11611. Offset from the input Jacobian origin to contact point, same world frame. Output columns Panda1…7 then ACTIVE order.

Create `Go2Pvcnn/tests/test_dh116_contact_frames.py`, all tests use real NumPy functions, no simulator mocks.

### Task 1: Surface and palm coordinate primitives

- [x] Write triangle tests for point (0.25,0.25,0), unit +Z normal and reversed winding. Reject malformed/nonfinite/degenerate triangles, invalid weights. Write frame tests for handedness, orthogonality, translation, zero/parallel directions, and round-trip transforms.
- [x] Run `taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_dh116_contact_frames.py -q` from Go2Pvcnn with PYTHONPATH=$PWD and numeric threads=1. Expect missing-module RED before implementation.
- [x] Implement the primitives with these numerical definitions:

```python
point = weights @ triangle
normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
normal /= np.linalg.norm(normal)
z = outward / np.linalg.norm(outward)
y = wrist_to_middle - z * np.dot(z, wrist_to_middle)
y /= np.linalg.norm(y)
x = np.cross(y, z)
R = np.column_stack((x, y, z))
p_world = R @ p_local + translation
n_world = R @ n_local
```

Inputs must be finite and exact shape. Reject triangle cross norm≤1e-14, invalid weight sums (>1e-9), direction norm≤1e-12, improper transform (RᵀR or det outside1e-8), nonunit normal (>1e-8). Check outputs finite to reject overflow; don't silently repair bad transforms or clip weights.
- [x] Rerun tests, require GREEN; no actual surface selection is implied.

### Task 2: Explicit-origin, named-column point Jacobian

- [x] Add tests for rotation lever-arm sign, reordered columns, left/right names, coupled follower contribution, and actual rigid rotation finite differences. Reject missing/duplicate/unknown joint names, invalid side, wrong shapes and nonfinite values.
- [x] Run tests and observe new feature RED before implementing.
- [x] Implement using the following reduction (validation as above):

```python
shifted = spatial_jacobian[:3] + np.cross(
    spatial_jacobian[3:].T, offset_world).T
arm_names = tuple(f'{side}_panda_joint{i}' for i in range(1, 8))
arm_columns = shifted[:, [joint_names.index(n) for n in arm_names]]
hand_columns = shifted[:, [joint_names.index(n) for n in JOINTS]]
result = np.concatenate((arm_columns, fold_jacobian(hand_columns, JOINTS)), axis=1)
```

- [x] Confirm central-difference error≤1e-8 m/rad for a synthetic revolute point, including coupling with a nonzero follower column; this tests math, not actual USD geometry or PhysX origin.
- [x] Run new tests plus existing DH116 tests with pxr and real table sources; report any skip explicitly.

### Task 3: Review and checkpoint

- [x] Read the new module/tests, verify the code does not claim normal winding proves the palm side or that any actual contact calibration was obtained.
- [x] `git diff --check`; record test counts and exact command in `notes/log/2026-10-15-dh116-contact-coordinate-core.md`.
- [x] Update T500 dashboard/branch/log index and CODEX_RESUME: design approved; pure foundation versus actual surfaces/PhysX gates distinguished.
- [x] Preserve unrelated files; only checkpoint these new task files if committing. Next action is inspect actual DH116 mesh components/frames and select outward pad surfaces, then author the concrete USD/PhysX integration plan using that evidence. Do not ask again whether to use upper pinch.

## Self-review

Coverage: this scoped plan implements the reusable coordinate and reduction contracts in D1/D2, with tests. Real surface identity/picking, source pins, actual model finite differences/PhysX, scene gravity and contact physics are explicitly not satisfied here and remain on the parent design's sequential work list. All four interfaces above are defined in this plan. No runtime or external dependency changes.
