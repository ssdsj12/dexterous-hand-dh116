# Power preload and the continuous wire envelope

The original 574-window Power approach proved a partial-opposition target path on the frozen 10 mm wire. Reading its exact final state at the archived SDK 6504 boundary showed that all four intended distal pads still had positive node SDF gaps. Appending only a zero command would provide no expected pad compression in that fixed geometry.

The new diagnostic first translates each palm about 0.05 mm in its own measured world normal direction. It then seeks a −0.050 mm signed distance by increasing only the original thumb and selected-partner flexion on each hand. Wrist orientation and abduction stay fixed, and all original actuator and geometry bounds apply. The final pairing is left index and right ring. The approved origin, asset identities and all target material stations are read from the previous binders. FK applies each measured pad offset once to the SDF pose and once to the separate geometry vertices used by the hull checks.

| Hand and pad | Original nodal SDF gap (mm) | Candidate flex increment (rad) | Continuous source gap upper bound (mm) | Continuous cooked gap upper bound (mm) |
| --- | ---: | ---: | ---: | ---: |
| left finger14_link | 0.329 | 0.010516 | -0.495 | -0.123 |
| left finger23_link | 0.284 | 0.003922 | -0.357 | -0.104 |
| right finger14_link | 0.268 | 0.003783 | -0.408 | -0.195 |
| right finger43_link | 0.179 | 0.002971 | -0.364 | -0.060 |

Both palm translations passed the continuous geometry verification. The four pads reached the nominal −0.050 mm nodal target with same-station SDF checks at every proposed SDK instant, but the distal phase failed the existing −0.150 mm guard. The upper bounds come from feasible points on the hull and wire segment. They demonstrate a violation of the geometric envelope. Even the cooked right thumb has a feasible gap below the limit. Neither signed SDF nor this envelope test measures physical contact load.

To separate between-node effects from a difference between the envelopes, a separate point-to-convex diagnostic replaced each adjacent wire segment with only its selected node. The original source convex gaps remained below −0.150 mm for all four pads at about −0.457, −0.322, −0.216 and −0.297 mm. The cooked node-only gaps were about −0.05 mm. Aligning the segment witness to the selected node can reduce the segment contribution, but leaves the measured source/cooked discrepancy in these configurations.

The same-station wrist exploration tried 95 offsets per side, with rotations up to ±12° about three axes and slides up to ±4 mm. Of the candidates that reached the intended signed gaps and passed the station and normal checks, all 23 left and 7 right failed an original or cooked gap by upper bound. The best necessary upper bounds in that cohort were −0.428 mm and −0.369 mm respectively. A second search tried the selected partners from the 38 left and 23 right previously clear wrist candidates at their distinct recorded allocations. All 61 reached the partner node target and each violated a hull gap. Neither sample search excludes untested poses or surfaces.

A follow-up face-normal screen examined all ten distal pads in their registered local frames. It intersected locked SDF target rays with a supporting-plane lower bound at one node. Within 15 degrees of the four tested normals it retained 40 left-thumb, 15 left-index, 30 right-thumb and 27 right-ring directions. The nearest options deviate by 5.5°, 10.2°, 6.0° and 7.4° respectively. They identify alternative surfaces for the next pairing search. They do not establish simultaneous reach, whole-segment clearance or a live contact normal. The geometric ray directions and the measured rod tangent have distinct roles.

Antigravity read the actual server source and independently checked phase ownership, continuous hull constraints and the distinction between signed distances and recorded force. The results prioritize finding a different contact surface and pairing that has small source/cooked discrepancy and enough continuous margin before generating an execution path. The rejected extension starts no native execution, grants no local opposition, and unlocks neither Power Lift nor Teacher PPO. Native qualification still requires measured bilateral load and the exact applied following-tick wrench; full grasp and held-out wire performance have their original criteria.

The related Ubuntu CPU regression passed 177 tests. The preload diagnostic returned exit 1 as expected.
