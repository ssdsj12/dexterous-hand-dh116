# Right Pinch relative Reset orientation — CPU counterfactual, blocked

Base: b3ba69674e748aa917bfbd9f00a9568da0cf6d7e. The previous frozen Reset candidate had a feasible-distance upper witness showing source-convex/capsule overlap. This package tests an actual change in relative hand/rod orientation, not another rotation that preserves the continuous rod geometry.

## Changes

- `solve_dh116_right_pinch_relative_orientation.py`: independent right Reset world X/Z rotations around (.43,-.2,.305)m, bounded to ±0.05rad each; composition Rz @ Rx after the existing fixed Reset. Actual recorded right FK poses, USD, own-side SDF and unchanged41rod nodes remain the source.
- `solve_dh116_right_pinch_reset_direction.local_pass`: accepts9coordinates as well as existing4/6/7; no old-domain behavior or physical limits changed.
- Every nonlinear endpoint and optional161-sample path retains both additional fixed angles.
- CPU budget246endpoint measurements,120seconds; path only if endpoints pass. No native recipe, no Isaac, no PPO.

## Actual result

Directory: `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-relative-reset-20261113-9shn7ze3/run01`.
Result SHA256: `532c6461b1ed30dd59b316b5e2718ef57a1ad9ba7ed3f47d17f50620fed86f86`.
Parent result SHA: `bd311cc064544f0b6c4ee295ac49d7b5e72f883dacd22c9f36ea6daa8787b865`.
Original physical ledger SHA: `170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf`.

119/246endpoint measurements;8.200207s; diagnostic process exit0. The first two local passes reduce the maximum normalized violation from0.148557 to0.140321. The third pass stops because two finite-difference scales disagree: relative Jacobian discrepancy0.325979. This is a local nonsmooth-constraint failure, not proof of global infeasibility. No automatic repeat or full path evaluation.

Selected fixed Reset changes: thumb spread +0.031175rad from measured initial q; rod-axis roll -0.0525rad; translation(-0.000144,+2.522241,-0.799363)mm, then X=-0.0025rad/Z=+0.0025rad about the stated pivot. Closure deltas thumb0.020734586rad/index0.031083766rad. These are counterfactual initial-pose parameters, not applied controls or larger actuator limits.

Endpoint failures: thumb gap+0.040321mm; index gap-0.075421mm, both miss the unchanged -0.1mm candidate margin; source finger13 rod lower bound-0.035878mm. The latter remains an unsafe/inconclusive separation certificate for this new candidate, not a measured native penetration depth. Previous V4's separately proven overlap must not be transferred to this new pose as proof.

Initial table clearance, source/cooked, mm:

| Right pad | Source | Cooked | Initial table penetration |
|---|---:|---:|---:|
| finger14 |3.779238|4.246292|0|
| finger23 |2.768066|2.768066|0|
| finger43 |24.052603|24.052603|0|
| finger53 |52.656782|52.946046|0|

All-shape initial table minimum2.364019mm. Initial clearance does not imply a safe/contact-capable closure path. Fresh authored-root consistency, self-collision and actual Reset tracking are still separate qualifications.

## Verification

RED6new tests before implementation; targeted19passed. Full regression111passed,0failed in6.62s:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_right_pinch_relative_orientation.py tests/test_dh116_right_pinch_reset_direction.py tests/test_dh116_right_pinch_measured_direction.py tests/test_dh116_polyline_closest_support.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_segment_geometry.py
```
Log: `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-relative-reset-full-20261113.log`.
SHA256: `93619369a9ebbc14accf0ee7cd89c5cb523698a09adcef89da993d42ba2c368e`.

CPU implementation passes; geometry, native contact and Teacher admission remain blocked. No new native physical step or PPO update. Preserve original physical data,528protected files,2historical checkpoints, previous failures and consumed recipes. Next: inspect the active nonsmooth constraint before a different frozen geometry experiment; independently resolve actual contact-opposition/stability/slip evidence needed by R0. No formal training or success claim.
