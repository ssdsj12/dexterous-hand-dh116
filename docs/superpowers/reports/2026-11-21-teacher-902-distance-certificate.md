# 902 distance certificate and complete command path audit

Baseline: `115c8e9e8e9ff236f2e995c331b0857adfc6d7a7`; Ubuntu worktree, branch `11-17`.
This package repairs an actual numerical acceptance bug. It does not qualify grasping or start Isaac.

## Source change

`scripts/audit_dh116_certified_segment_distance.py::segment_hull_bounds` now requires independent feasible-witness upper and full-vertex support lower bounds for both optimizer status 0 and eligible status 8. Previously a success flag returned the raw distance estimate, which can overestimate clearance. The legacy `segment_hull_distance` returns only the certified lower bound.

`_polish_segment_distance` proposes at most 64 nearby active-feature KKT candidates when the original estimate is insufficient. Every candidate must pass the original certificate. The selection tolerance is not a physical allowance; the original 1 nm bound width, rounding reserve, SLSQP tolerance, 0.150 mm contact guard, all noncontact/support guards remain unchanged. Singular/nonfinite/unproved candidates do not pass.

Tests first reproduced false-success acceptance. Strict certification initially exposed four real Reset regression failures: the old successful solve left 1.54–80.65 nm uncertainty. A smaller optimizer tolerance alone did not resolve all cases. Bounded active-feature polishing restored the original Pinch/Tripod CPU geometry screenings without relaxing the certificate. All failing logs are retained privately.

## Existing measured data, new CPU computations

No new native data. The geometry source is the sealed Tripod SDK400–408 boundary. The old source collider, cooked pads, measured FK correction and original action kernels remain intact.

- Direct path, ordinal 462, source index: complete-envelope lower -0.227055 mm; feasible subset upper -0.224478 mm. This is a proven geometric counterexample to the -0.150 mm guard.
- Delayed path, ordinal 374, source middle: full nine-frame, three-node envelope lower -0.149168 mm. This proves only that one point's geometry bound, not the whole path.
- A subsequent complete 66-control-window/528-substep ideal command replay includes actual float32 finger and wrist writes. It finds another index failure at ordinal 478: lower -0.172797 mm, feasible upper -0.172726 mm. Noncontact minimum 2.623057 mm; support clearance minimum only 0.002514 mm. The route remains blocked.

The refinement recomputes support over ALL nine frames and each original three-node group. It does not replace that envelope with a last-frame or per-frame segment union. The maximum saved node displacement is 84.22137 micrometres; simply dropping old frames would silently weaken coverage.

These signed capsule-envelope gaps are geometric tests, not measured DeformX or PhysX penetration depths. Future q/root states are ideal tracking hypotheses. There is no new contact load, hold, lift, reward or training evidence. The native guard has not been changed to use the diagnostic refinement.

## Verification

- Final combined CPU run: **208 passed**, 56.08 s, exit 0 (148 related tests including 19 numerical certificate cases, plus 60 Registry tests). Log SHA `1d14ddc82fd0e17841f217d5c6a1c9f44487ca9e22bb990d917fad133ead82f0`. Earlier red and call-site failures remain retained.
- Original geometry parser remains byte-identical (regression enforced).
- Both fixed camera configurations, 528 protected files and three original checkpoints are verified before publication.
- Antigravity performed four read-only rounds against the same worktree; Codex corrected the temporal-envelope assumption and independently verified the final source. Final review matched source SHA `9af88d4f563d25c8bf63e566801397c76a310fcdffb01cf5ffdd54cc38adc5be`.
- Zero Isaac runs, RGB capture, Teacher queries or PPO updates.

The executable test command is `python -m pytest -q` followed by the explicit `cpu_tests.test_files` list in readiness; Registry checks use `tests/test_dh116_work_packages.py`. Private diagnostic commands and raw log hashes are recorded under `evidence_sha256`; raw geometry and logs are not uploaded.

## Next work

Keep the delayed candidate rejected. Adjust the actual middle/index command trajectory or outside clearance, then check every substep and live guard consistency before a new native recipe. Pinch still needs qualified fixed-wrist hold; Power still needs the requested palm/fist wrap and support-free Lift. All three retain the original 30 mm / 5 s / 0.5 mm criteria and remain outside the success expert pool.
