# Power re-seating on the measured palm contact frames

Antigravity and Codex independently traced the World-space solver contact
point, force, outward normal, and material-node ID to the same SDK 6504
exchange used by the previous continuous flexion audit. The force-weighted
palm witnesses at nodes 31 and 9 anchor separate left and right scans.
The source is the 10 mm specimen on two end supports.

The finite sweep used three rotations from −20° through +20° in 5° increments.
Five axial offsets ranged from −8 mm through +8 mm. Up to 2 mm of projected
correction kept the current material node near the registered palm SDF.
Every candidate used the original Reset wrist position and angle bounds, actual
curved wire, original joint travel, and matching archived palm and finger SDFs.
An open endpoint had to clear the original table, supports, and thumb geometry.
The independent per-finger check rejected a pose if another sampled wire node
penetrated that same pad beyond the original 0.15 mm limit. Normal tests included
the prescribed material half and opposition to the palm for the four fingers.

| Hand | Palm node | Grid poses | Open endpoints screened | Best independent near pads | Full combined endpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
| left | 31 | 3645 | 477 | 4/5 | 0 |
| right | 9 | 3645 | 386 | 4/5 | 0 |

- Left best grid pose had roll 0°, yaw -20°, pitch 5° and slide 0 mm. Remaining pads were `finger53_link` (7.008 mm).
- Right best grid pose had roll -15°, yaw -10°, pitch -10° and slide -4 mm. Remaining pads were `finger53_link` (8.593 mm).

The grid establishes an explicit negative result for this local search. The
next Power probe needs a different material station or a staged change in
wire pose before full closure. The omitted continuous wrist transition and
live contact qualification remain mandatory before applying a new candidate.
Any future lift still uses the actual wire displacement after the supports
are unloaded. Three-grasp Teacher qualification and wire-specification
generalization remain pending.

The first scan reached its 310 s CPU bound. Explicit world-axis separation now
handles distant supports and the original SAT remains active near the clearance
threshold. The revised scan completed. The related CPU suite passed 133 tests.
This replay added zero Isaac or PPO runs.
