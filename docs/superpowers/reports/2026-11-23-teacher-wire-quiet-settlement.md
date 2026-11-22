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

Antigravity reviewed the original lifecycle and identified a missing terminal gate call in the first diff. Codex verified the omission, attached the gate and its frozen source, and repeated the 93 CPU regressions. The requested native quiet variant is pending. The next contact route must close on real pads before reaching the unchanged hull limits, then transfer opposing loads into an off-support Lift.
