# DH116 Tripod contact readiness at the actual SDK 568 endpoint

The right thumb, index, and middle witness gaps at SDK 568 are 1.083, 0.900, and 1.335 mm. They remain 1.104, 0.809, and 1.201 mm when the original backend wrist and finger targets replace both measured poses. The remaining separation is largely geometric. The tested wrist-only endpoint worsens one of the six model/pad gaps by 20.3 µm. Simultaneous finger closure helps, but it consumes the right thumb source-hull clearance unless the wrist moves with it. A two-hull constraint changes the limiting collider to the right index at useful doses. The existing 1/16 candidate retains a better combination of gain and continuous reserve than the tested alternatives.

## Evidence and attribution

The read-only input is the sealed 902 two-window native run ending naturally at SDK 568. Its outer SHA-256 is `3f404d60345a420e2d0e48141344f25b3eaa23c2531f871b4a8cc0c7faa385ca`. The audit verifies 639 inventory entries and the finished SDK 568 interlock for Reset seed `9d16393ecdcc02d007c7518d2d25e79b16d14b268e8e228765a86935b0799704`. The controller starts at SDK 240 and the completed SDK 568 state is the pre-action boundary for policy 41 (SDK 569–576). The audit checks the same-reset clock and preceding solver wrench delivery on each of the 16 transitions from SDK 552 through 568. It retains the true solver-substep loads by owning hand and pad. All six pads recorded zero positive load over 204 solver substeps, and all 16 delivered hand wrench packets were zero. The two supports stayed loaded, with minimum force norms of 0.156788 and 0.156775 N.

The three fixed material witnesses are right thumb node 3, index node 2, and middle node 4. The signed attribution below evaluates all eight combinations of measured wrist, palm-relative finger pose, and rod node position at the two endpoints. Averaging factor orderings allocates interaction and makes the sum equal the observed gap change. The attribution compares geometry states at the archive endpoints and does not infer a dynamic response to each factor in isolation.

| Pad | Observed 552 → 568 gap (mm) | Wrist (µm) | Finger pose (µm) | Rod (µm) |
| --- | --- | --- | --- | --- |
| Right thumb | 1.158 → 1.083 | +54.471 | −132.247 | +2.764 |
| Right index | 0.939 → 0.900 | −27.458 | −9.418 | −1.698 |
| Right middle | 1.424 → 1.335 | −51.782 | −31.169 | −5.694 |

Replacing the SDK 568 measured wrist and finger joints with their backend targets uses original FK. Across both substitution orders, the wrist contributions are +115.785, −84.421, and −119.418 µm, and the finger contributions are −94.655, −6.525, and −15.454 µm, in thumb/index/middle order. The rod remains at its measured SDK 568 position in this substitution.

## Geometry controls

The diagnostic fits the worst of all six signed witness reductions across measured-offset and absolute-backend models on the original 24D action. The tested wrist-only direction uses world translation and intrinsic XYZ and the finger-only direction uses the four original right finger targets. The finger-only endpoint improves every witness by at least 215.4 µm and consumes about 557 µm of the thumb source-hull gap. The unrestricted coordinated endpoint improves all six by at least 367.3 µm and consumes about 624 µm at the thumb source hull. These are endpoint comparisons.

Constraining the signed thumb segment 2 and middle segment 4 source-hull derivatives retains both source hulls in the endpoint model. The candidate replay uses the original controller projection and finger guards, an 8-step 480 Hz window, unchanged left targets, and the continuous bilateral screen for 44 shapes against the wire/table/supports under both pose models. The only joint action channels are right 6–9 (thumb finger11 and finger12, index finger21, middle finger31). Right ring and little (10–11) and all left joint channels have exact zero action. The original kernel confirms the fixed targets for these joints at each modeled step. Their measured joint positions can still move under dynamics. The rod radius and selected-pad depth gate come from the original freeze. The allowance is 43.809 µm, the larger of the original same-node prior eight-step drift and eight times the last displacement. The per-shape history allowance is sixteen times the measured prior-step movement, from the existing two-window verifier. These extrapolation allowances cover the modeled execution only.

| Candidate | Scale | Smallest pad-gap reduction (µm) | Minimum modeled swept rod margin (µm) | Limiting right source hull |
| --- | ---: | ---: | ---: | --- |
| Existing unguarded direction | 0.0625 | 28.965 | +50.100 | Middle |
| Thumb constraint | 0.25 | 50.275 | +5.115 | Middle |
| Thumb + middle constraints | 0.5 | 60.060 | +12.668 | Index |
| Thumb + middle constraints | 0.25 | 31.060 | +32.392 | Index |
| Thumb + middle constraints | 0.125 | 16.628 | +75.561 | Middle |

The matched existing direction uses this diagnostic's larger per-shape history allowance. Its original independent one-window certificate reported +73.056 µm. The double-constraint 0.5 candidate leaves measured-offset endpoint gaps of 1.009, 0.583, and 1.275 mm. Its continuous proof at scales 0.75 and 1.0 fails on the right index source hull. The result supports retaining the existing action for the separate guarded continuation. The modeled endpoints have positive witness separation, and the replay provides no evidence of achieved contact load or dwell.

The frozen argv, numeric output, logs, hashes, and regression result are recorded in [the contact-readiness evidence contract](../../contracts/dh116_teacher_tripod_contact_readiness_v1.json).
