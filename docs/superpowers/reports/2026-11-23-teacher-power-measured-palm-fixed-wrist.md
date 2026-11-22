# Power wrist reach from the measured palm boundary

The archive-based CPU command binds SDK 6504, rod tick 78024, the backend joint readback
and the actual body transforms from the naturally closed bilateral palm witness.
It verifies source hashes, the same Reset and episode, the 41 world nodes,
the actual support layout, and the original 10 mm wire diameter.
Both palm meshes had positive measured load on their assigned wire halves at the
boundary. The archived case had zero support-free rows.

The independent sweeps hold each measured wrist and the rod shape fixed.
Each flexion sweep covers its full original coupled joint interval. The thumb
opposition angle stays at its measured value during the thumb flexion sweep.
The five distal cooked hulls per hand are aligned to their measured body frames,
and the source-to-PhysX position residual is checked before the sweep. The
continuous bound covers spacing between samples using the existing point-travel
model. Two borderline results were independently rechecked with an adaptive
disjoint interval cover.

| Hand | Cooked pad | Lower bound on gap | Excess above 0.5 mm geometric witness limit |
| --- | --- | --- | --- |
| left | `finger53_link` | 3.269 mm | 2.769 mm |
| right | `finger23_link` | 0.621 mm | 0.121 mm |
| right | `finger53_link` | 0.776 mm | 0.276 mm |

These gaps exclude a five-finger wrap from this fixed-wrist boundary. The bounds
describe pad travel relative to the archived rod. The wrist target and live rod
response require their own verification. The other independent finger sweeps
leave simultaneous closure and force unverified.

The next Power approach should reseat the wrists around the measured palm
witnesses before closing. Confirm the ordered, independent hand and support
geometry at the selected pose, then close using the same accumulated wrist
targets and monitor their absolute readback. The force check still requires both
palms, all five pads and opposition on the correct wire halves. Measure eventual
Lift using the lowest and median wire node height from Reset after support loads
clear, then verify the original Hold and Slip limits.

The CPU replay exits 2 for the demonstrated unavailable fixed-wrist approach.
The related suite passed 157 tests. No new Isaac or PPO run took place.
Native Power, Teacher and held-out wire success remain pending.
