# Native Power proximal pose-source result

The process completed naturally with both outer and GDB exit code 0 after 33.913 seconds. At SDK step 4 it read two poses for each original `finger52_link` without advancing the simulation between reads. The legacy PhysX rigid-interface pose differed from the exact single-body Tensor View by 0.09353 mm left and 0.09365 mm right, and by about 0.016211 radians on both hands. The matched rays still hit the same physical link identity. The Tensor View pose matches the scene-query surface on each observed ray.

| Side | Rigid-interface error P95 / max (mm) | Tensor View error P95 / max (mm) |
| --- | ---: | ---: |
| Left | 0.447923 / 0.516633 | 0.000051 / 0.016792 |
| Right | 0.447968 / 0.516725 | 0.000055 / 0.016796 |

The Tensor View comparison returned 36 owned hits per side out of 64. The minimum observed normal dot product improved from roughly 0.472 to 0.99957. Neighboring finger surfaces still occluded other rays. This is a surface witness for the observed rays and original cooked mesh; it does not qualify an SDF or task contact. No physics parameter or collision geometry changed.

The earlier capture specification counted the two explicit samples and omitted the two steps consumed by SDK initialization. Actual total SDK step count was four per capture. The code now budgets four and checks the exact setup count before the explicit steps. The remaining budgets for control, solver and PPO were zero throughout. The clock correction passed 66 related CPU tests.

Frozen recipe `15c385e935d1ad7401968118b2edeb292d54a78db49233673ff82926c67a0a7f`, outer receipt `687dcfeec7741aa3e48c822a4b8a692074ef110029281e1451723064aaf0ca12`. The corrected pose evidence can be used with exact topology and signed-distance audits for a separate candidate SDF bake.
