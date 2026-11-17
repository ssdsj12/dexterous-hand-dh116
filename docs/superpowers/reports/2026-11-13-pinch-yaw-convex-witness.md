# Pinch archived yaw witnesses — 2026-11-13

This CPU-only work package rechecks the sixteen nonzero-yaw endpoints already saved by the previous 102-state screen. It does not add candidate poses, change reset configuration, launch Isaac or establish contact qualification.

## Implementation
`scripts/audit_dh116_pinch_yaw_witness.py::witness` reconstructs the selected source convex envelope using the actual independent hand geometry and endpoint FK. It selects one actual segment of the archived 41-node rod polyline, computes a constrained closest-point candidate, contracts the hull point into the convex envelope, projects the rod point back onto that finite segment and records a feasible-pair distance upper bound. Negative radius-adjusted upper bounds prove overlap of this convex envelope and a 5 mm capsule segment only. A positive upper bound cannot prove separation, nor global closest distance. Source convex envelopes are not claimed to be exact PhysX cooked collision surfaces.

`evaluate` verifies the complete parent SHA, independent hand source SHAs and active q/delta identity. Every reconstruction has a retained attempt/status and selected actual segment. `main` caps this fixed cohort at 16 witnesses/30 seconds and preserves unexpected failure evidence. No radius, guard, action, mesh or physical parameter was modified.

## Actual CPU evidence
- Parent: `pinch-static-yaw-20261113-sqwO6J/run01/result.json`, SHA `d434fbd069325dbaf71195a0452717667a43be9c043e7e57601b77ffc1407890`.
- New private result: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-yaw-witness-20261113-Nhixzu/run01/result.json`.
- Result SHA: `61130374e62c82aa796b5b0bfbce95a92e015d0ae12ee3ad4435131bc37df715`.
- Actual wall time 3.763251369 s; 16/16 reconstructions; 10 convex-envelope overlap witnesses and 6 inconclusive cases. No new geometry-search candidates; 0 physics steps; 0 PPO updates.
- Left negative yaw: finger13 overlap bounds -1.435/-1.025/-0.613/-0.196 mm at -0.030/-0.0225/-0.015/-0.0075 rad. Left +0.030 finger22 -1.330 mm. Left other positive angles remain inconclusive with this one-segment witness.
- Right -0.030/-0.0225 finger22 overlap bounds -0.482/-0.250 mm. Right +0.015/+0.0225/+0.030 finger13 -0.224/-0.449/-0.668 mm. Right -0.015/-0.0075/+0.0075 remain inconclusive.

These are upper bounds from feasible source-envelope pairs, not penetration-depth measurements, actual rod contact, or a proof that every nearby pose is infeasible. No hidden bridge across rod segments was used.

## Tests and failure evidence
Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_dh116_pinch_yaw_witness.py tests/test_dh116_pinch_static_yaw.py tests/test_dh116_pinch_initial_table_clearance.py tests/test_dh116_pinch_trust_region.py -q`.
Actual: **56 passed**, three pre-existing HiGHS option-forwarding warnings, 47.04 seconds.
Private log: `evidence/pinch-witness-regression-nxITs3/pytest.log`.
Red tests are retained: missing new module at `pinch-witness-red-4CYeBO`; unexpected-exception evidence test initially failed at `pinch-witness-exception-red-O6vYxx` (1 failed/7 passed), then passed after exception evidence preservation.
Frozen source digests and run stdout remain alongside the result; large native/model artifacts are not published.

## Gates and next
CPU witness implementation passed. Static reset candidate, native contact, grasp and Teacher admission remain false. The old native budgets remain exhausted. Next work must couple initial palm placement and finger motion with actual independent hand FK; pure-yaw screening alone cannot solve the surrounding non-target interference. Preserve original runtime table/joint protections, independently validate any new reset/path before freezing a new bounded native recipe. Teacher PPO remains 0/3; no force/reward qualification is inferred.
