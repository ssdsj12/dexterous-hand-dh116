# Coordinated loaded grasp controller

User requests an integrated repair in this turn, with full close-to-lift validation, rather than stopping after each parameter experiment. Existing authorization covers local reversible implementation and bounded simulation. Training resumes only after actual control validation; no reward edits.

Evidence: v141 left index recovery targets are below policy envelope floor for 94% of active recovery samples at 4 N baseline and 100% at 8 N baseline. This clips away the intended recovery. v140 shows 0.008 rad position lead maintains load at unchanged 0.2 Nm. Palm compensation currently follows requested pressure offset even when joint motion is blocked.

Design: opt-in separate controller, default legacy path bitwise compatible. Keep action dimension20, policy action envelope, measured readiness, 480Hz, effort0.2Nm, original independent physical guards and reward unchanged. A bounded physical governor may retreat the index only at most0.03rad below the policy envelope, never beyond mechanical bounds. All targets use 0.008rad position lead and 0.05rad/s slew. Master targets also satisfy a 0.008rad follower-error budget; incompatible limits prioritize bounded recovery, while hard guards remain authoritative. This avoids repeatedly opening a loaded finger solely for exceeding the old0.004 soft tolerance, while also preventing unlimited closure.

Lift and close readiness use measured coupling <=0.008rad (loaded fixture gives0.0053/0.0074rad). Severe errors >=0.012rad invoke recovery and pause palm/lift; 0.02rad hard protection remains unchanged. Pressure release retains priority. Positive pressure integration pauses at coupling budget or position saturation; negative release stays available. No automatic torque increase.

Palm compensation uses actual midpoint displacement relative to each palm: mean(two pad-relative-to-object positions) + object-relative-to-mean-palm - (individual palm offset - mean palm offset), compared with its reset anchor. Thus unrealized force-offset requests do not move the palm, common lift is not canceled, and existing palm bounds/speed/table clearance remain enforced.

Implementation/validation:
- [x] Pure controller tests for blocked recovery floor, measured compensation, saturation, pressure release, mechanical bounds/slew, inactive and reset slots.
- [x] Integrate opt-in controller and paired inference probe (4/8 N, legacy/candidate, two spatial repetitions); preserve exact default behavior.
- [x] Run CPU regressions and one Isaac instance, record source hashes, contact forces and physical failures.
- [x] Measure actual3mm/0.1s and30mm/0.5s criteria, not commanded height: both remain unsuccessful. Training admission follows stable reset/control and reachable phases; full grasp success is a learning objective, not a prerequisite.
- [x] Persist results, provenance, training status and checkpoint compatibility; never report complete grasp success without measurement.


Integrated validation findings:
- v142: rejected at6.0625s; right thumb crossed mechanical lower bound minus hard tolerance.
- v143: index-only recovery and near-limit opposing-index release still rejected at5.2875s. Target limits do not prove actual loaded joint limits. No success claimed, no PPO deployment.
- New transport defect: during permitted lift, fixed base pose derivative is approximately0.008–0.010m/s while solver-reported base velocity is exactly0. Fixing rewards cannot supply this missing base motion.
- Separate kinematic wrist was rejected by installed GPU backend (setter unavailable). Final `--palm-transport driven_carriage` uses three prismatic drives inside each hand articulation, explicit USD DriveAPI, measured base-link observations and3mm/.06m/s tracking/speed guards. Normal motion is driven; only resets teleport the anchor.
- v151 empty-hand validation:16/16 hands track30mm trajectory with approximately0.103mm maximum error and0.010m/s measured velocity.
- v152/v153 full probes each completed8/8 episodes without fatal physical fault; neither passed3mm or30mm grasp criteria. Latest maximum clearance0.531mm. No reward changes, no relaxed physical guards.
- CPU regression105 passed,2 skipped; exact migration retains iteration583 policy/optimizer. v1544096-env training resumed in a new directory. Actual updates verified past583; old checkpoint preserved. Stable grasp remains a learning objective.
