# Tripod same-episode quiet wire and first left thumb reaction

The optional diagnostic starts the original straight free wire in the previously verified 9 mm by 450 mm parked scene. The simulation settles through SDK1426 with gravity and both supports active. Before any approach command, it checks the same live episode against the recorded quiet centerline, link poses, joint positions and backend targets that the CPU path screen used. It then keeps that solver and episode alive.

The left wrist issues (+5.600, −1.400, +11.400) mm of world translation over at most 96 control windows, with two reserved hold windows. The right wrist and both sets of finger targets stay fixed. The action and geometry checks preserve the original table and support clearances, the original 0.25 mm non-target clearance, and the original −0.15 mm selected gap limit for the left thumb terminal pad. Its allowed material stations are 34 to 36. The independent SDK budget is at most 2210 steps, with 26496 solver substeps and a 600 second soft wall bound.

A valid first contact requires positive normal force on an allowed rod-to-thumb pair and a nonzero hand reaction. The selector holds targets as soon as that contact appears. At the immediately following SDK step the monitor requires the actual delivered hand composer wrench to match exactly. No grasp, lift, learning or policy qualification is assigned by this diagnostic.

The CPU regression passed 95 tests, including rejection of a modified trajectory command, the wrong pad, a shifted centerline and a different joint pose. Antigravity independently checked the native settling start versus action end, the contact station, the response timing and the legacy path. Native execution is pending.
