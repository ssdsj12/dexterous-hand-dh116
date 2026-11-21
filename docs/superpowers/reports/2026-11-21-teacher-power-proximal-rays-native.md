# Native Power proximal ray witnesses

CPU and GPU PhysX each completed two SDK setup steps and two explicit capture steps in separate sequential Isaac instances, with natural outer and GDB exit code 0. Both original `finger52_link` convex hulls had 34 vertices and 61 polygons. The CPU/GPU vertex and directed topology comparisons were exact for each side. Each device and side returned 36 of 64 rays on the intended original rigid body and collision owner. The other rays include intervening original finger51 and finger53 shapes and do not attest the proximal surface.

| Physical device | Left ray depth error P95 / max (mm) | Right ray depth error P95 / max (mm) |
| --- | ---: | ---: |
| CPU | 0.000042 / 0.016849 | 0.000044 / 0.016836 |
| GPU | 0.447923 / 0.516633 | 0.447968 / 0.516725 |

The ray comparison uses the SDK body pose and raw cooked triangle fan in the same record. GPU normals on the weakest observed rays have a dot product of approximately 0.472 against the chosen fan normals. A rigid alignment fit to the GPU observations reduces but does not eliminate the face residual. The exact collision owner is witnessed on the reported rays, while sub-millimetre GPU contact geometry for the candidate SDF remains unqualified. The next diagnostic must compare the runtime Tensor View and rigid interface poses and the native query surface at the same SDK step. Neither force routing nor Power qualification was attempted.

The CPU run took 30.053 seconds and the GPU run 33.015 seconds. The frozen fingerprints, natural exits and hashed rays are recorded in the readiness JSON. The related CPU regressions passed 68 tests. DeformX never stepped in these captures.
