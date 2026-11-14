# Flexible R0 applicability and reset baseline implementation plan

Goal: remove the pre-grasp undefined-slip dead end without filling unknown measurements or changing legacy R0.
Execution is covered by the user's continuing-development approval; this package does not launch native physics or train a model.

Design: preserve the original episode and reducer files byte-for-byte. An opt-in diagnostic v2 episode reuses original physics/success calculations, adds explicit term applicability, and optionally binds a same-clock measured reset approach distance. A v2 reducer preserves exact-once 480/60 Hz settlement, failures and budget accounting. Raw missing slip stays null/invalid. Only known no bilateral grasp before any bilateral contact permits an inapplicable slip contribution of zero. Any bilateral contact latches applicability for the entire episode, including missing anchor data and later contact loss. Unknown contact never establishes inapplicability. Production coefficients and native/PPO admission remain unqualified.

Alternatives considered: zero-fill all missing slip would conceal sensor loss; changing v1 would silently alter frozen evidence. Both are rejected.

- [x] Red tests: pre-grasp diagnostic is finite with measured reset approach and all other required sources; missing raw slip is still null; all three grasp pad contracts; unknown contact blocks; contact-loss latch; fresh reset; exact timestamp; no double settlement; early terminal; original threshold and sources.
- [x] Implement independent v2 config, episode and reducer. Include version, source/config SHA, distinct measured/applicable/contribution counters; do not change coefficients or v1.
- [x] Run new tests and legacy reward/control/reset/native-parameter regressions. Preserve failure logs outside Git.
- [x] Offline read only the completed 18-step native parameter task; identify which real terms still lack sources, without adding physics or pretending the CPU reward is native-qualified.
- [x] Update report, Registry/readiness, protect 528 files/two checkpoints, commit on Ubuntu, import and publish via Windows, verify GitHub SHA and Draft PR.

Verification: 122 + 9 distinct CPU tests passed. One contact-history regression failed red then passed after fail-closed ambiguity latching. Native parameter ledger replay remains reward-blocked; no new physics. Publication completion is recorded by the scheduler and Windows release manifest, not inferred from this source plan.
