# Power surface pair fitting on the frozen supported wire

This CPU diagnostic changes the pose search after the original preload failed the continuous source and cooked envelope. It uses the same SDK 6504 rod, reset wrist bounds, joint limits, calibration offsets and SDFs. The earlier closed target remains an input for ranking nearby pad surfaces. No controls are exported for native execution.

The search generates local rays from hull vertices and cooked face centres. A bracketed intersection with the locked trilinear field supplies a node point at the −0.050 mm target. A source supporting plane with a small reserve prioritizes ray seeds. That choice narrows the search and does not exclude physically valid surfaces beyond the sampled seeds. The final certification uses the original geometry.

The thumb ray is placed on its assigned material node by solving for wrist translation. The optimizer changes the wrist orientation, thumb abduction and flexion and the selected partner flexion. It fits the second ray to its own node and allows the palm to find a contact point on its SDF. The optional axial residual favors pad normals perpendicular to the real secant of the adjacent material nodes. It changes ranking and posture selection and imposes no new acceptance threshold. Each fitted end pose is rechecked on its actual locked SDF and owned nodes. The continuous source and cooked hull checks, other fingers, table, both supports and a common Y=0 separating plane govern acceptance.

| Fitting mode | Hand | Ray combinations | Reaching final gates | Rejected by rod envelope | Accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| anchored thumb | left | 100 | 100 | 99 | 0 |
| anchored thumb | right | 100 | 93 | 56 | 0 |
| transverse residual | left | 100 | 91 | 61 | 0 |
| transverse residual | right | 100 | 39 | 2 | 0 |

The initial fixed-palm prototype admitted only two left fits and one right fit to its subsequent geometry and ownership checks. Allowing translation from the thumb anchor removed that residual as the primary cause in the baseline. Nanometre-scale partner fit was possible, but close fit still left original and cooked solids inside the rod capsule.

The axial residual moved the main left failure to other links. Its 61 rod-envelope rejections included 26 first failures at the thumb middle link, 21 at the little-finger proximal link, 11 at the ring-finger proximal link, and 3 at the cooked thumb pad. Both right rod-envelope failures were at the source ring pad. All 63 have feasible upper bounds below their original limits. For the closest fitted left rod rejection, an independent sweep of each other digit over its original active range at 18 sampled positions found no passing little-finger setting. The highest tested feasible upper bound for that digit was −0.322 mm against its +0.250 mm requirement. This result applies to that wrist configuration.

Additional searches covered adjacent and coincident owned-node pairs and alternative partner fingers. The station search evaluated 180 new ray combinations and rejected one station choice at external support before fitting. The partner search evaluated 125 combinations over three additional left and two right measured-wrist seeds. Neither group produced a complete static candidate.

Antigravity read the exact server implementation and verified how the pad calibration is applied. The diagnosis separates fit residual, node ownership and continuous capsule clearance. It leaves no evidence of live opposition or Lift, and no evidence that the three grasp policies are trained. A usable alternative must pass both original envelope checks at reachable wrist and joint settings before any continuous command certificate or native contact experiment.

The related Ubuntu CPU regression passed 180 tests. Both actual searches returned the expected exit 1.
