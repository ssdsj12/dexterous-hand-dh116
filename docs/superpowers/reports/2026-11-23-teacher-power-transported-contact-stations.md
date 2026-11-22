# Power fore-aft transport to measured material stations

Antigravity and Codex reviewed the shared material-frame transport. The
replay starts at the same World-space palm witnesses on material nodes 31
and 9 in SDK 6504. The support layout, 10 mm wire centerline, measured wrist
readbacks and original SDF captures remain the source.

The new mode preserves the wrist and contact-point offsets relative to the
source rod frame, transports those offsets and the outward force direction
to the target station, and recomputes the local tangent. It searches the
measured stations and four nodes toward the midpoint per hand. Each station
uses three rotations from −20° to +20° in 10° steps and axial offsets of
−4 mm, 0 mm and +4 mm. Candidate wrists must stay inside the original Reset
workspace. The previous table, support, penetration and thumb gates apply.

The archived support loads are on nodes 1 and 39. The diagnostic support
zone includes those nodes, their immediate material neighbors, and nodes
whose rod surface lies within the original 2.1 mm clearance of a support box.
The resulting indices were 0, 1, 2, 38, 39 and 40. Pad contacts in this zone
are recorded separately from independent near-pad counts. These masks are
part of this CPU search and the original native Power force contract remains
the source for physical qualification.

| Hand | Measured node | Proposed nodes | Poses scanned | Open endpoints kept | Best scored pads | Little gap in best pose |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| left | 31 | 31, 30, 29, 28, 27 | 1875 | 199 | 3/5 at 30 | 5.169 mm |
| right | 9 | 9, 10, 11, 12, 13 | 1875 | 166 | 3/5 at 11 | 7.488 mm |

- Left retained endpoint counts were 31 (72), 30 (55), 29 (42), 28 (25), 27 (5).
- Right retained endpoint counts were 9 (57), 10 (39), 11 (42), 12 (24), 13 (4).

No pair qualified for the combined static endpoint check. A future full
candidate must pass whole-hand rod clearance and left/right separation at
the open and closed endpoints. Continuous wrist and finger paths and new
native load evidence must follow before Lift can be tested.

The comparison supports a staged contact attempt using an interior patch
and measured rod motion before the Power wrap. Thumb and finger contact
during that stage is intermediate evidence. Full Power still requires the
original five distal pads, palm, opposition and independent Lift, Hold and
Slip checks.

The related CPU suite passed 135 tests. The scan starts zero Isaac or PPO
runs. Pinch, Tripod and Power native qualification and held-out wire
generalization are pending.
