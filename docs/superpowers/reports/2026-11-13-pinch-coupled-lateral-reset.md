# Coupled Pinch lateral reset and yaw — 2026-11-13

## Change
`scripts/solve_dh116_pinch_coupled_lateral.py` extends the existing five-coordinate static reset/closure search with independently bounded world-X translation (+/-3 mm) and world-Z yaw (+/-0.03 rad). `decode` always uses the ORIGINAL pair origin and absolute limits; the previous trust candidate is a coordinate, never a rebased origin. Only initial pose changes; runtime wrist commands, active/passive coupling, geometry, material, actuator limits and protected assets remain unchanged.

`model` uses 29 full-query points for two-scale seven-dimensional differences. `refine` performs at most two directions per hand, one LP and one actual nonlinear candidate per direction, preserving hard constraints and old acceptance ratio. Rejected candidates cannot advance the current point. `run` uses independent left/right actual source geometry, cooked hulls and frozen SDF, both reset/endpoint constraints and a final 33-point path with displacement-derived continuous lower bounds. The shared cap is 306 complete geometry states and 50 seconds, not a new native budget.

## Actual CPU run
One run only: private `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-coupled-lateral-20261113-MIOUYp/run01/result.json`.
SHA `d6032b457debc9f1962dfe556a1367e1fd38afc463d1f901292a0fa46cd9f157`.
306/306 complete states, 4.674479436 seconds; 0 physics steps/0 PPO updates.

| Side | Accepted directions | Final thumb gap mm | Final index gap mm | Continuous table lower mm | Continuous non-target lower mm |
|---|---:|---:|---:|---:|---:|
| Left | 0/2 | +0.269221 | -0.106181 | 2.437380 | 0.155903 |
| Right | 1/2 | +0.339444 | -0.181363 | 2.866782 | 0.122870 |

Neither side meets two-pad contact. No new native reset recipe was created; Teacher admission remains false. Negative predicted gaps are CPU SDF predictions, never measured contacts.

Right accepted static offsets relative to ORIGINAL pair: X -0.300 mm, Y +2.100 mm, Z 0 mm, yaw -0.003 rad, initial thumb spread +0.042 rad; bounded closure thumb 0.018210446/index 0.039466763 rad. Left remains the old trust candidate. Both full paths preserve the original 2.1 mm runtime table floor and non-target separation lower bounds. No runtime protection was changed.

## Failure analysis
- Left first/second trial violated the 0.2 mm non-target design margin by 0.000584/0.000122 mm and were rejected. Their contact merit also failed or was insufficient for hard feasibility.
- Right first trial preserved hard constraints but actual/predicted improvement ratio was only 0.080892 (< unchanged 0.1 acceptance requirement), and was rejected.
- Right second trial preserved hard constraints and ratio 0.508768, reducing maximum contact deficit by 0.099037 mm; it still did not establish thumb contact.
- Two-scale infinitesimal Jacobian agreement (relative difference <=0.001516) did not predict finite candidate quality sufficiently. This is evidence to use actual nonlinear constraint optimization, not to relax contact/table guards or claim global geometric infeasibility.

## Tests
Red missing-module evidence: `evidence/pinch-coupled-red-6YiCXo/pytest.log`.
New tests 11 passed; full relevant regression **69 passed**, six existing/consistent HiGHS option-forwarding warnings.
Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_dh116_pinch_coupled_lateral.py tests/test_dh116_pinch_static_yaw.py tests/test_dh116_pinch_trust_region.py tests/test_dh116_pinch_spread_axial_direction.py -q`.
Private log: `evidence/pinch-coupled-regression-e0OTHF/pytest.log`.
Tests cover original absolute mapping, world quaternion composition, joint/speed/search limits, all seven derivative columns, nonlinear rejection, hard cap, and partial evidence preservation.

## Next
A separately bounded direct nonlinear optimizer may reuse this exact source evidence and variable box. It must retain rejected/partial attempts, validate all source/cooked/SDF constraints and continuous paths, and must not reuse this exhausted diagnostic run. Only a geometry-qualified reset/path may produce a new frozen native recipe. Real hand contact/reset/R0/Teacher gates remain blocked, with no Student or Teacher training performed.
