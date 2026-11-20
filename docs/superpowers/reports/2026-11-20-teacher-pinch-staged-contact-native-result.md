# Native staged Pinch contact observation

The frozen runner at `2b55ea620aa81441c38e912cf9dc8597270d2ee2` completed two setup steps, eight zero-action quiet steps and 64 close steps. All 72 measured SDK rows, all nine control windows and 864 solver substeps are covered by the retained clock ledger. Isaac closed naturally with exit code 0 in 77.35 seconds. The quiet window passed with table loading in all 96 solver substeps and no hand load. There was no Lift, Hold or PPO update.

## Actual contact and tracking

| Quantity | Left | Right |
| --- | ---: | ---: |
| Minimum measured table clearance | 2.365186 mm | 2.326097 mm |
| Minimum non-target geometry lower bound during the run | 0.023871 mm | 0.034955 mm |
| Peak wrist position error | 0.027195 mm | 0.049542 mm |
| Thumb peak normal load | 0.05398 N | 0.56954 N |
| Index peak normal load | 0 N | 0.47274 N |
| Thumb substeps with measurable load | 246 | 17 |
| Index substeps with measurable load | 0 | 30 |
| Substeps meeting same-hand loaded opposition | 0 | 15 |

The right hand met the same-hand loaded opposition thresholds in its last 15 solver substeps (SDK 73 substep 9 through SDK 74 substep 11). Bilateral loaded opposition occurred in zero substeps. The left index remained 0.232327 mm from the rod at the last measured boundary. There was no Z projection clipping or original guard termination. The measured right finger12 never fell below 0.008949057 rad, keeping a minimum hard-limit margin of 0.001036435 rad. Left index tracking error was 0.003744 rad at the final sample. The maximum rod node displacement from the end of the quiet interval was 0.480179 mm.

## Replaying the measured endpoint

All counterfactuals use the actual final SDK link poses, joint state and rod nodes from one clock. Ideal finger tracking with the measured wrist held fixed predicts nominal left thumb and index gaps of -0.206 and -0.013 mm. Ideal tracking of all existing targets predicts +0.031 mm on the left thumb and -0.580 mm on its index. Simply waiting for all targets to converge cannot be assumed to restore left opposition. Interpolating only 25% of the remaining left wrist movement while restoring the existing finger targets predicts -0.150 and -0.157 mm, with a 0.213 mm non-target endpoint lower bound. The complete left interpolation passed the static table and non-target checks for the final rod snapshot.

On the right, keeping the measured wrist while moving the fingers all the way to their targets fails the static non-target check. Moving wrist and finger state together to the current command passes on the same snapshot. The next diagnostic should therefore treat wrist/finger tracking separately by hand and measure loads during the bounded corrections. These snapshot calculations do not establish how the real rod or joints settle, and they confer no grasp or Teacher qualification.

## Artifacts

- Frozen recipe SHA-256 `2949dc46e6ed043dcf07e373b016990f73a829caede1fdca366d0907ed4486d6`.
- Native ledger SHA-256 `2c3f9693ec9110c5dddf1e225585beb6b57b12f293dd9f86fc0414c767200202`.
- Independent measured metrics SHA-256 `2f408a04034855ebf68a8cac9081d3b818891b3999524a7cf8526f0046033310`.
- Endpoint exact-target comparison SHA-256 `2998411a6cc4be0898e83ec1c71eca45ff96fb828121b7a288f05e77b85b32db`.
- Endpoint fraction sweep SHA-256 `0ec6df95cb9b1a9e0538221c9a4e77f4cb0b4714e364652d0fb0f68c5f57b51f`.

The unchanged worker source passed 158 focused CPU regressions before launch. The native ledger was independently reopened, its sources rechecked and its solver loads recomputed. No model or raw ledger is included in the public report.
