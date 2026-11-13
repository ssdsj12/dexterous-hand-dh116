# Right Pinch source-surface registration: rejected mirrored direction reference

Base: cc680a8df4e73db8be5625b2943af284c4d97c2f. CPU only; 0 new Isaac steps, 0 PPO updates.

## Implementation and evidence

`scripts/solve_dh116_right_pinch_surface_registration.py` resolves an actual locked-SDF zero near each audited **right** USD thumb/index patch; uses the right FK and original affine passive coupling; fits a proper SE(3) root, never a reflection quaternion. The left SDK234/substep11 pair directions are explicitly diagnostic references. Material stations35/34 map to unchanged Reset nodes5/6. No right joint angles are copied from the left.

Each fitted candidate is independently checked against all17 source convex envelopes,5 locked cooked pads, tableZ0.3m, 2.1mm clearance floor, non-target rod separation, 41-node SDF depth/margin and opposition. A good point/normal fit does not qualify collision safety, actual contact, full closure path, or Teacher admission.

Frozen read-only native ledger SHA: 170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf.
Result: `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-surface-registration-20261113-NWCSlfLI/run01/result.json`
SHA: ba47c60cd99ac88dd818ff41bc8de65990e1db9a262f35ea1e8515c461c1c8c4.
All three distinct starts converged to materially identical unsafe candidates. 103/768 FK evaluations,8.483097s/120s. Maximum60 solver nfev per seed; no budget extension.

## Actual failure

Seed0 final q: [0.499139854,0.048080965,0.705440050,1.15,1.15,1.15].
Root xyz: [0.522191494,-0.168653393,0.201956456]m.
Root wxyz: [0.666181355,-0.274107197,-0.494123510,0.486733606].
This is a new CPU-proposed Reset; it was **not executed**. The reference right Reset and original assets remain unchanged.

| Right collision link | Source table penetration mm | Locked cooked penetration mm |
|---|---:|---:|
| finger14 |11.739441|11.619289|
| finger23 |18.547631|18.543802|
| finger43 |44.631922|44.630296|
| finger53 |51.569218|51.535935|

Worst source shape: base_link,118.335749mm below table. Right finger13/rod certified lower separation improves to1.488391mm, but that cannot compensate for table collision. Thumb/index SDF gaps are -0.373678/-0.387568mm, also beyond original -0.3mm depth guard. Normal dot=-0.813063 and registration RMS0.328455mm are not success criteria.

The evidence rejects this particular mirrored-direction reference construction. It does **not** prove that every independent right-hand grasp is infeasible. Next: retain right-owned feasible palm-side geometry, constrain table side explicitly during target construction, and derive right target normals/stations independently instead of imposing the left mirrored pair.

## Validation and protections

Initial RED: missing implementation import, collection failed as expected; preserved `surface-registration-red-20261113.log`.
GREEN/regression:93 passed,0 failed. Includes exact proper rotation recovery, reflection rejection, undefined geometry, true locked-SDF root/gradient, identity/substep validation, non-replenishing FK budget, and rejecting collision of the seventeenth source shape even with perfect pad margins.
Command:

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_surface_registration.py tests/test_dh116_right_pinch_measured_direction.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_segment_geometry.py
```

CPU run: `bash scripts/python.sh -m scripts.solve_dh116_right_pinch_surface_registration --output <new-exclusive-directory>`.
No native recipe emitted; full path remains unqualified. No source mesh, physical coefficient, actuator limit, success threshold, controller, checkpoint or historical data changed. Publication verifies528 protected files and2 Checkpoints.
Teacher qualification remains Pinch/Tripod/Power0/3; R0 stability/material slip and full native Reset/action/state gates remain blocked. Existing measured left short contact is not a5s grasp.

## Other work / recovery

This package does not consume native budget or rerun C5. Previously measured234-step Pinch natural outer exit0 remains its original narrow scope; no new Tensor View ownership or exit claim is made. Existing WireSeg diagnostic counts remain128 segmentation and128 depth steps; no new training in this package. Tripod/Power independently retain their geometry blockers. New native budget is not requested for these unsafe candidates.
