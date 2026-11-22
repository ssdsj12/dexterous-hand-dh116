# Tripod same-episode quiet wire and first left thumb reaction

The optional diagnostic starts the original straight free wire in the previously verified 9 mm by 450 mm parked scene. The simulation settles through SDK1426 with gravity and both supports active. Before any approach command, it checks the same live episode against the recorded quiet centerline, link poses, joint positions and backend targets that the CPU path screen used. It then keeps that solver and episode alive.

The left wrist issues (+5.600, −1.400, +11.400) mm of world translation over at most 96 control windows, with two reserved hold windows. The right wrist and both sets of finger targets stay fixed. The action and geometry checks preserve the original table and support clearances, the original 0.25 mm non-target clearance, and the original −0.15 mm selected gap limit for the left thumb terminal pad. Its allowed material stations are 34 to 36. The independent SDK budget is at most 2210 steps, with 26496 solver substeps and a 600 second soft wall bound.

A valid first contact requires positive normal force on an allowed rod-to-thumb pair and a nonzero hand reaction. The selector holds targets as soon as that contact appears. At the immediately following SDK step the monitor requires the actual delivered hand composer wrench to match exactly. No grasp, lift, learning or policy qualification is assigned by this diagnostic.

The CPU regression passed 95 tests, including rejection of a modified trajectory command, the wrong pad, a shifted centerline and a different joint pose. Antigravity independently checked the native settling start versus action end, the contact station, the response timing and the legacy path. The first native execution is recorded below.


## Native contact and shutdown result

The frozen 9 mm by 450 mm run reproduced the quiet supported endpoint at SDK1426 with zero differences against the screened native poses, rod centerline and joint readback. The same episode executed 742 control physics samples. The left thumb terminal pad contacted node 35 at SDK2167, with peak normal force of 8.858 mN. At SDK2168 the held action applied the preceding solver's matching world reaction through the original left composer, with a force norm of 7.031 mN. The right palm position stayed unchanged and the wire remained supported. An independent readback verified 2939 archived artifact hashes and both recorded exchange clocks.

The outer wrapper treated the reserved SDK2210 limit as a required exact endpoint. It rejected the successful early endpoint at 2168 before the one-time cache release. The fault close returned, but the ownership guards failed with live articulation and asset views. GDB recorded SIGSEGV in the PhysX Tensor plugin during Python finalization. The outer exit was 255 and the run did not qualify natural termination. The simulator was never forced closed.

The correction only admits this contact recipe's verified early endpoint if the live SDK clock, solver clock, quiet start, reaction transfer and written completion record agree. A rejected boundary within this mode still attempts the original one-time cache cleanup and preserves the original exception. The archived successful reaction evidence exercises the endpoint rule independently. The correction passed 106 CPU tests, including rejection of missing completion evidence and mismatched clocks.

At the SDK2168 pose an independent left-finger CPU sweep found guarded residual SDF gaps of 0.205 mm for the index and 0.565 mm for the middle even when each finger was moved independently toward the wire. Their predicted normals oppose the thumb. The right index screen found two feasible endpoints in 380 checks, while continuous approaches from the swept-clear parking positions still failed. This evidence directs the next contact search. Neither stable opposing load nor off-support lift has been measured.

## Corrected native result

The single corrective run at `559bfff2137fe4096be7d7043cbe7a715b833170` reproduced the SDK1426 handoff, the 742 action samples and the SDK2167/2168 contact and reaction. The measured forces matched the first run. An independent audit verified 2942 artifact hashes. The one-time cache release ran at SDK2168 with no change to the SDK or app clocks. After `app.close()` all 15 tracked Python weak references were dead. The outer process and debugger inferior naturally exited 0 without a signal, forced exit, or timeout after 310.02 seconds. The weak references measure Python ownership and the external receipt measures process termination. This qualification covers this bounded recipe.

No Teacher queries or PPO updates were executed. The wire remained in the supported task and only the left thumb was qualified for reaction transfer. The opposing-load, wrist-fixed hold and off-support lift gates remain open.

## Pose adaptation across initial wire shapes

The same SDK2168 trace served as the starting state for CPU geometry checks. Nine sampled left thumb-to-index routes passed the existing checks at 41 samples per route. The example with a 0.12 rad world yaw about node 35 and a 0.05068 rad index increment predicts opposing normals with dot product −0.4674. Its lowest certified gap of −0.14946 mm leaves about 0.54 micrometres of margin to the existing −0.15 mm guard. These samples use the captured rod geometry and do not establish a clearance margin for live deformation or sustained load.

For two 10 mm by 500 mm training specimens, the CPU mapping shifts the selected contact by the measured centerline change and adjusts its orientation to the local material tangent and its offset to the wire radius. Each soft and stiff condition yielded 10 of 10 qualified thumb endpoints. The tangent rotation was 0.04955 rad for the soft specimen and 0.00957 rad for the stiff specimen. The node height changed by −2.744 mm and +0.303 mm respectively relative to the 9 mm source. These are endpoint checks. Full paths, reaction feedback and held-out generalization are pending.

The next physical gate is sustained opposing measured load. After that, qualification requires a fixed-wrist hold, complete release from supports, and the existing 30 mm lift, 5 second hold and 0.5 mm maximum slip. Report Pinch, Tripod and Power separately on held-out diameter, length and stiffness combinations, with pose and force adjustments attributable to each specimen.
