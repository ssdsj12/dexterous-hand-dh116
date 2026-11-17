# Pinch direct nonlinear reset solve — 2026-11-13

## Code and unchanged contracts
`scripts/solve_dh116_pinch_direct_sqp.py` replaces neither runtime nor previous search. It is a separate CPU-only direct SLSQP experiment on the same original seven-coordinate box, using the previous coupled result as starting coordinates. Actual independent left/right FK, original source/cooked collision envelopes, locked SDF and original joint coupling remain authoritative.

`Queries` deduplicates exact coordinate bytes and retains each unique attempt/error; no failed value is replaced with zero. `solve` optimizes actual target deficits and hard constraints, uses an explicit normalized difference step 0.001, at most 16 iterations and 200 unique queries per side, and retains only actual hard-feasible iterates. The optimizer success flag cannot override full geometry. A 0.002 mm stricter design reserve helps numerical constraint handling; it does not lower runtime margins. Each query consumes two complete states. Final 33-state paths are reserved per side; shared cap 866 states/60 seconds. Exhaustion returns retained best evidence without restarting or expanding budget.

## Actual run
Private result `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-direct-sqp-20261113-MS6xpU/run01/result.json`.
SHA `ef6e8d68f036f0a3a87389b6c3227cdaf403eae42f7e95d0b509378a8ae8d1c7`.
One bounded run: 808 complete states of cap 866, 6.845512363 seconds, no timeout, 0 physics/PPO.
- Left: 171 unique queries, 16 iterations, optimizer reports iteration limit reached.
- Right: 200 unique queries, per-side budget exhausted; no retry. Unused shared states do not grant extra queries.
- Both final 33-state paths completed from retained hard-feasible iterates.

| Side | Thumb SDF gap mm | Index SDF gap mm | Continuous table lower mm | Continuous non-target lower mm | Initial rod separation lower mm |
|---|---:|---:|---:|---:|---:|
| Left | -0.016617 | -0.006738 | 2.422430 | 0.122400 | 0.734847 |
| Right | +0.233784 | -0.174770 | 2.897987 | 0.137711 | 0.540757 |

Both left target pads enter the CPU SDF negative-gap predicate, but do **not** satisfy the unchanged -0.1 mm preflight contact target. Right thumb remains separated. Neither side is qualified for a new native recipe. Source/SDF values are not measured PhysX contact or grasp success. Global infeasibility is not established.

## Retained pose changes
Relative to the ORIGINAL independent pair:
- Left X -1.014045 mm, Y +1.149672 mm, Z +0.391103 mm, world yaw +0.003927818 rad, initial thumb spread -0.000375425 rad; thumb/index closure 0.014493728/0.040587435 rad.
- Right X -0.750489 mm, Y +2.147625 mm, Z +0.018988 mm, world yaw -0.001790410 rad, initial thumb spread +0.041617460 rad; thumb/index closure 0.006959475/0.043729742 rad.
No initial configuration or native recipe was overwritten. Complete q, root quaternion, all collider constraints and node identities remain in the private result.

## Verification and failure evidence
Red missing-module log: `evidence/pinch-sqp-red-DoC4au/pytest.log`.
Eight new tests passed. Regression **54 passed**, five existing HiGHS option-forwarding warnings.
Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_dh116_pinch_direct_sqp.py tests/test_dh116_pinch_coupled_lateral.py tests/test_dh116_pinch_trust_region.py tests/test_dh116_pinch_static_yaw.py -q`.
Log: `evidence/pinch-sqp-regression-FAAw6a/pytest.log`.
Tests cover exact cache/cap, true nonlinear feasibility despite false optimizer success, evidence on query failure, no restart on cap, counted failed states, per-side reserved path budget, and partial timeout without native qualification.

## Readiness and next work
CPU implementation and finite evidence collection passed; optimizer convergence/contact qualification did not. Teacher remains 0/3, PPO updates 0. Old/native budgets remain consumed. The current fixed-pitch/roll reset box has not yielded robust two-pad contact. Next investigate a separately bounded full orientation/contact-aware reset, using these exact retained poses and all non-target/table constraints, or advance independent reward/reset data contracts. Do not simply rerun this solver with a larger iteration count, lower the SDF margin, turn off table collision, or infer successful contact from negative source SDF alone.
