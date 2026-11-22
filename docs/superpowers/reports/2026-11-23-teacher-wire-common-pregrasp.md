# Shared pregrasp reset selection for flexible wires

The screening command uses the 11 training wire specifications and nine fixed training placements. It selects the least opening fraction and shortest admissible wrist shift for each hand. Opening changes the five original flexion masters towards their source limits. Wrist orientation and Mimic are preserved. A Power-only option permits a different initial thumb opposition angle. It screens 34 source convex envelopes and ten locked cooked pads, uses the original wire, table and support guards, and checks both original wrist return limits and a conservative separation between hands. The initial one-direction wrist sweep failed for all three grasps.

| Grasp | Training geometry evidence | Frozen static holdouts |
| --- | --- | ---: |
| Pinch | 99 layouts, left +4 mm X and 25% open, right +12 mm X / +3 mm Z and 50% open | 104/104 |
| Tripod | 99 layouts plus one sealed measured curve, left -4 mm X / -6 mm Z, right -12 mm X, original finger angles | 104/104 |
| Power | 99 layouts, left +18 mm X with full flexion opening and 75% of original thumb opposition, right +8 mm X / +3 mm Z and 75% flexion opening | 104/104 |

The held-out cases reuse the previous 104 specimen and placement keys exactly. Only the selected resets alter their physical initial condition. Keeping the original Power thumb opposition failed across 441 left-side candidate results. A left pose that passed the first 10 training layouts failed support separation at a lower bound of -1.91 mm on the 500 mm, 9 mm-diameter specimen. Reducing the left starting opposition to 75% cleared the 99 layouts with the rematched right reset. The smallest world-Y separation between the two hand envelope groups was 260.5 mm. Native Power must still prove that the palm and five pads load the same local wire region through closure.

A pure 9 mm Tripod quiet run provided 48 solver-aligned records and a natural outer exit. The selector verified the outer, plan, result and record hashes, the training specimen and reset identity, and the original support-only quiet result before using the archived curve. The previous statically selected Tripod pose would require a nominal station-aligned descent beyond the original -4 mm limit by about 2.7 mm. The new selection keeps the station targets inside the unchanged workspace for the archived curve and the 99 nominal training layouts. The minimum component return margins were 3.315 mm left and 0.358 mm right. These are endpoint estimates. Contact, trajectory clearance and tracking of the new poses still require a same-episode native Reset, settlement and approach.

Run `python -B -m scripts.select_dh116_wire_common_pregrasp --scene ... --output ... --left-material-id ... --right-material-id ...`. Power uses `--power-left-thumb-scale .75`. Tripod's measured screening additionally supplies `--settled-run ... --settled-outer-sha256 ...`. Replay the frozen scene in `scripts.prepare_dh116_wire_holdout_resets --audit-hulls` with the existing held-out catalog and placements. The readiness contract binds the receipts and raw artifact hashes.

This package ran 109 CPU regressions, geometric preflight and sealed-data replay. No new Isaac attempt, off-support Lift, Teacher training or across-wire grasp acceptance occurred.
