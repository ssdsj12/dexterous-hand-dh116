# Measured supported wire settlement

The existing DeformX CUDA solver ran with the same free rod, gravitational acceleration, table and two support SDFs used in the resized Tripod experiment. The isolated solver reproduces the previous 9 mm Isaac episode from SDK3 through SDK73 exactly in node positions and vertical support loads. Both support pairs were active. The isolated run has no hands and cannot qualify grasp.

| Training specimen | Time to quiet window | Centerline maximum drop | Maximum node speed in final 48 steps |
| --- | ---: | ---: | ---: |
| 9 mm × 450 mm | 2.967 s | 18.781 mm | 0.0943 mm/s |
| 10 mm × 500 mm, soft | 3.633 s | 28.645 mm | 0.0883 mm/s |
| 10 mm × 500 mm, stiff | 2.983 s | 19.352 mm | 0.0970 mm/s |

Each window retained both original support contact slots over all solver substeps. The initial rod stayed the authored straight rest shape. A later grasp must continue from the same physically settled solver instance to preserve strain and state. A fresh reset directly from curved points would change the solver rest shape.

A 9 mm CPU parking screen checked the whole measured sag sweep against all 44 source and cooked hulls. Selected left and right parking poses keep 3.60 and 2.00 mm rod clearance. The source hand joints, original straight rod and supports remain in the original frames. This gives a separate 1426 SDK, 17088 solver-substep plan with zero control or PPO. Independent end-pose screening against the final line shape considered 158 left and 13 right translation or small wrist rotation candidates after SDF and opposition screening. None passed all original geometry guards.

The opt-in native variant reuses the original Isaac ownership path. Its terminal gate checks 48 consecutive endpoints with the same episode identity, actual bilateral support pairs, closure of vertical support load to measured weight, gravity sag and the existing quiet motion limits. It records a post-settling lowest-node reference for the subsequent lift calculation. Its own success grants only a settled geometry measurement. The previous eight-step probe keeps its original scope.

Antigravity reviewed the original lifecycle and identified a missing terminal gate call in the first diff. Codex verified the omission, attached the gate and its frozen source, and repeated the 93 CPU regressions. The native quiet variant ran once and is recorded below. The next contact route must close on real pads before reaching the unchanged hull limits, then transfer opposing loads into an off-support Lift.


## Isaac result and contact refinement

The one-environment 9 mm × 450 mm parked scene reached SDK1426 with 1424 measured solver exchanges and 17088 internal substeps in 126.06 s. It exited naturally with code 0 and no signal. Independent replay verified 1454 raw artifacts. All 1424 frames matched the standalone centerline and vertical support loads exactly. No hand contact appeared. The two support loads at the boundary were 0.11631 N and 0.11635 N. The last 48 frames had maximum speed 0.09430 mm/s and displacement 0.000149 mm. The native lowest-node Lift reference is 345.7192 mm.

A finer offline endpoint search found 20 admissible left thumb candidates in 387 shape checks. On the right, 480 endpoint checks found no admissible contact, with the proximal thumb link the most frequent rejection. These screens use actual measured settled geometry to plan contact. They establish neither measured hand reaction nor a completed grasp.

Using the measured SDK1426 joints and pad transforms, six paths also passed a CPU swept-hull screen for the entire fixed-joint translation. The selected straight path moves the left palm (+5.600, −1.400, +11.400) mm. Its non-target hull bound is at least +2.524 mm through the path. The thumb hull lower bound is −0.101 mm at worst and the endpoint SDF at material node 35 is −0.029 mm. The original selected hull floor is −0.150 mm. The corresponding upward-first path failed. Execution must first reproduce the quiet window and then continue in the same live episode with real-time geometry and wrench checks.
