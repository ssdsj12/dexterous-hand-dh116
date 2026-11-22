# Continuous target path for partial Power opposition

This diagnostic uses the exact recorded 10 mm wire boundary at SDK 6504 and rod tick 78024. The earlier 33-pose samples selected a left palm retreat of 1.5 mm and a right thumb flexion lead. The new script increases the left retreat to 1.8 mm along the same measured normal and converts those curves into 60 Hz 24D commands with the existing finger velocity, wrist vector-rate, translation and intrinsic-XYZ bounds. The three command stages contain 284, 202 and 88 control windows, or 4592 additional SDK steps and 55104 rod solver substeps at the existing cadence if executed in the simulator. No part of the sequence was executed in this release.

The certificate evaluates the actual piecewise-linear target state within each command window. Geometric margins subtract conservative link travel around each interval midpoint. Wrist rotation uses the sum of the absolute intrinsic-XYZ rates, and digit bounds include affine mimic joints and the maximum slope of each compiled joint path. The two hands must clear opposite sides of the same world Y=0 plane throughout the path. Every original source hull and baked pad is checked against the table and supports. Non-palm hulls also use the wire-polyline distance lower bounds, keeping the original target/non-target margins. The original endpoint check enforces palm proximity at the stage boundaries.

The palm uses the registered trilinear lattice. Projecting a node onto the closed box is nonexpansive and the trilinear field has a bounded slope over that box. Each node can clear by distance outside the box or by the SDF margin of its projected point. The certificate also requires at least one guaranteed in-lattice node. The high grid face is evaluated from an interior point with its displacement deducted. A blocked lower-bound proof alone cannot identify penetration.

The 1.5 mm route verified five of six hand-stage certificates. The left wrist reached 4096 samples with no negative sampled lower bound and about half the stage covered. A bounded 1025-point palm-only comparison found a minimum margin of 0.205 µm at 1.5 mm and 115 µm at 1.8 mm. The final audit uses conservative per-node velocities and rechecks every certified interval of the 1.8 mm route. The six-face signed distance minimum is over 12.0 mm for each original lattice.

| Stage | Command windows | Left intervals | Right intervals |
| --- | ---: | ---: | ---: |
| wrist reseat | 284 | 330 | 355 |
| thumb first | 202 | 19 | 346 |
| two pad closure | 88 | 15 | 10 |

All six target-geometry intervals have complete disjoint coverage. The related Ubuntu CPU suite passed 174 tests.

This scope assumes a frozen rod shape and the ideal controller targets. It provides no reserve for tracking error, rod motion, force transients or GPU lattice arithmetic. The archived native start target differed from the measured root by less than 0.04 µm and from the active joint readback by at most about 0.60 µrad. Those are one-boundary measurements. Guard projection, target lead, native force and action ownership still require verification on any live continuation. No partial opposition can stand in for all-finger and palm Power closure, and no closure can stand in for measured support separation and Lift/Hold/Slip.

Antigravity read the same server code and identified the mismatch between the smooth curves and the 60 Hz held action kernels, as well as the lattice boundary and simultaneous hand-motion gaps. Codex checked the command compiler against the existing Torch kernels and audited the proof on the bound archive. The open PR stays Draft. The original rejection limits and acceptance metrics apply to Pinch, Tripod and Power. Qualification across independent wire specifications remains separate from this reference-wire run.
