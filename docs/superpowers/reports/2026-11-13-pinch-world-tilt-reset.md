# Independent Pinch world-tilt reset — 2026-11-13

## Runtime change scope
`solve_dh116_pinch_direct_sqp.py::Queries/solve` now accept explicitly selected 7D or 9D coordinates; the default 7D path, hard margins, finite-difference step, selection, caps and failure semantics remain unchanged. No PPO/controller/physics code changed.
`solve_dh116_pinch_orientation_reset.py::decode` adds two STATIC reset rotation coordinates:
`R = Exp([0.03*x7, 0.03*x8, 0]) * R_world_z_yaw * R_original`.
This is world-frame precomposition about the unchanged translated base origin, not an intrinsic Euler action. Zero tilt preserves exact old 7D pose values. Original joint, translation, yaw and closure bounds are retained. The new coordinates do not change wrist commands or actuator amplitudes.

The new driver verifies old run bytes against immutable source commit `fd9e632f9a82dbb2a8ff2a1afea672bb172bff48`, rather than falsely comparing an old hash with the newly dimension-parameterized helper. Original hand geometry, physical helper files, SDF and source identities must still match current files. Left/right are loaded independently.

## One actual CPU diagnostic
Private result `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-orientation-20261113-qiicfu/run01/result.json`.
SHA `6149a6b5b82b39a18407b9b6de431335fd244ecc9610e884db9bae3ef8a1c6fb`.
7.543025582 seconds, 866/866 complete states. Both sides consumed exactly 200 unique queries; no retry or budget expansion. Reserved 33-state complete paths were retained. No Isaac/physics/PPO execution.

| Side | Thumb gap mm | Index gap mm | Continuous table lower mm | Continuous non-target lower mm | Qualified |
|---|---:|---:|---:|---:|---|
| Left | -0.016617 | -0.006738 | 2.422430 | 0.122400 | No |
| Right | +0.231705 | -0.164020 | 2.963322 | 0.174810 | No |

Left retained its parent; no feasible lower-loss iterate was found within the budget. Right obtained only 0.002079 mm thumb-gap improvement, still no contact. Selected right static tilt vector [0.000167083451,0.000187108283,0] rad. These CPU SDF predictions are not actual contacts. Both require the unchanged preflight target gap <=-0.1 mm, bounded depth >=-0.3 mm, original continuous table floor 2.1 mm and non-target separation. No native recipe was emitted.

This does not prove global infeasibility or establish that adding more iterations would solve the pose. Failure evidence is retained to avoid repeating this local experiment. Next geometry work must reconsider the reset coordinate/contact construction, not silently expand this consumed search.

## Tests and historical compatibility
- New-dimension red: six failures/eight passes before the optional interface; retained `pinch-orientation-dimension-red-*/pytest.log`.
- New driver red: missing module, `pinch-orientation-red-pJvQq0/pytest.log`.
- Full regression: **67 passed**, five existing HiGHS option-forwarding warnings, 3.96 s.
- Log: `evidence/pinch-orientation-regression-gJxpKI/pytest.log`.
- Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_dh116_pinch_orientation_reset.py tests/test_dh116_pinch_direct_sqp.py tests/test_dh116_pinch_coupled_lateral.py tests/test_dh116_pinch_static_yaw.py tests/test_dh116_pinch_trust_region.py -q`.

Tests cover exact zero-tilt compatibility, world-frame quaternion composition, unchanged base position and finger motion, no mutation, all old bounds, historical byte verification, explicit dimension rejection, nine-dimensional hard feasibility and existing seven-dimensional fail-closed behavior.

## Readiness
CPU implementation validated; two-pad reset/contact qualification failed within this finite budget. Teacher 0/3; no training. All historical original assets and models remain untouched. Independently continue solver parameter-source work: current radius/gravity/stiffness fields remain configured values with no native getter, so they cannot be silently represented as measured parameters or used to claim complete R0 availability.
