# Power high wrist bypass native result

The prior opening replay matched the sealed SDK 880 state. From that state, both Power wrists followed their own two-stage paths on the continuing rod scene. One GPU 2 instance completed 3584 SDK steps, 418 control windows and 42984 solver substeps in 586.3 seconds. The worker and GDB exited naturally with code 0.

| Boundary | Left position error | Right position error | Left rotation error | Right rotation error |
| --- | ---: | ---: | ---: | ---: |
| SDK 1408 | 0.090 mm | 0.090 mm | less than 0.001 rad | less than 0.001 rad |
| SDK 3584 | 0.098 mm | 0.076 mm | 0.00725 rad | 0.00479 rad |

Across 2704 new SDK rows, both supports registered all 32448 solver-substep contacts each. There were zero hand or table pairs and zero live geometry stops. The independent decoder verified that the backend finger targets remained the ones at the measured opened SDK 880 state.

This establishes a physically reachable high pre-grasp waypoint with the original hand assets and commands. A subsequent staged descent needs to establish measured palm contact, then opposing five-finger contact and support-free lift. No grasp, Lift, Hold or Teacher training passed here. The frozen source passed 219 CPU regression tests before the native run.
