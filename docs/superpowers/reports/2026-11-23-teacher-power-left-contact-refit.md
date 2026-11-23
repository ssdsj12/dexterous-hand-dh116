# Power left contact and the bilateral partial endpoint

Both hands pass the original endpoint checks on the same archived 10 mm wire at SDK 6504 and rod tick 78024. The right side matches the endpoint of the existing 82-window [right-hand approach](2026-11-23-teacher-power-right-concurrent-route.md). The new left side keeps the original palm, thumb and index material nodes. The artifact records the pair in left-then-right order.

The closest prior left seed had an owned thumb gap of +76.95 µm. Its limiting cooked index interval crossed the −0.150 mm guard. Separate reevaluation also found two source hulls below that guard. The bounded refit included 14 separating planes across six nearby hulls and used both endpoints of each represented rod segment. It fitted shallow contact on each original material node. The first development run had acceptable contact geometry but left only 5.54 µm of table clearance. The final local fit explicitly includes the original table clearance for every hull.

| Metric on the left | Recomputed value |
| --- | ---: |
| Palm at node 29 | −14.59 µm |
| Thumb at node 37 | −2.00 µm |
| Index at node 36 | −2.56 µm |
| Thumb–index normal dot product | −0.935874 |
| Minimum table clearance | 2.105 mm |
| Minimum support clearance | 12.406 mm |
| Source index conservative lower bound | −145.21 µm |
| Cooked index conservative lower bound | −67.90 µm |

The numerical optimizer reached its iteration limit. The independent original endpoint gate accepted the resulting pose after recomputing workspace and coupled limits, original SDF stations and normals, full source and cooked hulls against the full archived rod, placement against the table and supports, inter-digit clearance and bilateral separation. The optimizer result itself grants no qualification. The table margin above the original 2.100 mm guard is 5 µm, with no demonstrated tolerance to native tracking error.

The left wrist-first, finger-first and simultaneous straight sampled paths all failed. The simultaneous chord reaches a palm gap of about −2.109 mm. This points to a coordinated palm retreat and thumb re-opening during the left transition. The existing right extension remains a fixed-rod CPU proof. Left continuous motion, all-finger wrap, real forces and independent held-out performance are still unverified.

The Ubuntu related CPU suite passed 203 tests. The recorded archive refit exited 0 after both partial endpoints passed. No Isaac, solver steps or PPO updates ran. See the [readiness record](../../contracts/dh116_teacher_power_left_contact_refit_v1.json) for source and artifact SHA-256 values and the reproduction command.
