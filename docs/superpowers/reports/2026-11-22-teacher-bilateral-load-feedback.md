# Bilateral force feedback before Lift

The user priority is real grasp and rod Lift. Historical Pinch seed replay repair is deferred and is not a dependency of this package.

The previous actual Tripod experiments reached only one consecutive SDK step with both hands qualified. The 1704-step single left-index dose did not stabilize the grip (minimum left total force worsened from 0.186 to 0.163 N). This package replaces neither experiment nor their evidence.

## Actual runtime change

- `BilateralLoadFeedback.decide` in `scripts/verify_dh116_supported_force_seating.py` selects at most one deficient pad per hand from the preceding completed sample's 12 solver substeps. A qualified hand holds its existing targets. When all pads exceed the controller target but total hand load is insufficient, a scalar opposing-side heuristic chooses one side; this is not proof of vector equilibrium.
- `run_supported_action` in `scripts/verify_dh116_power_shared_rod.py` issues the selected canonical 24D command to the existing `FlexiblePhysicalExchange.control`, with a separate command ledger. The existing exact SDK1296 pose/contact-pair/composer-wrench boundary check remains active.
- `--bilateral-feedback` in `scripts/verify_dh116_tripod_supported_reset.py` is opt-in and replaces serial seating doses only for its independent config. Wrist channels remain zero. The original single-hand paths remain available.

The 0.085 N controller target does not replace the original qualification: each required pad >0.05 N, hand >0.30 N, correct material side and effective opposition. Per-finger cumulative requested target travel is capped; it is not described as actual measured joint displacement. The new thumb-flexion dose is an unvalidated native candidate.

## Verification

115 relevant CPU tests passed (53.74 s), including 27 owner tests. Tests first failed for the absent feedback, wrong opposing-side selection, JSON freeze round-trip, inconsistent loaded flags and unknown/incomplete runtime variants. A fake physical driver exercises the real runtime function, verifies issued actions reach control, and checks all eight SDK callbacks. This is CPU integration, not Isaac validation.

The real-data CPU freeze succeeded with the intact original replay: 196 total control windows, 1808 SDK steps including two setup steps, 21672 solver substeps, 600 s soft wall, zero Teacher/PPO/model operations. Native execution and stable grasp/Lift remain pending.

Antigravity read the actual Ubuntu sources in Teacher-GraspLift R1–R3. Codex rejected the proposed arbitrary new success thresholds, per-control phase ticks, and the assertion that monotone potential implies positive discounted shaping. The new feedback is a diagnostic candidate, not a PPO Teacher.

## Remaining work

Run the bounded same-replay contact-hold comparison. Require actual sustained bilateral reaction before adding wrist lift, then measure rod net height and removal of support load. Full Hold/Slip still needs real sources. Pinch and full-fist Power require their own contact conditions; no Tripod result qualifies them. Formal BC/DAgger/Student training remains closed.
