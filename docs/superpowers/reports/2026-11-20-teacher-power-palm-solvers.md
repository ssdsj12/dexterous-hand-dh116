# Power palm loads in the original CUDA solver

The verifier reads the two independently measured PhysX palm ray sets and both fixed, original-mesh SDFs. A tangent is selected from original named palm hits. The solver receives a 41-node free rod, original gravity and material, and one palm per isolated fixture. The installed DeformX extension was verified against the original native DH116 profile. Each fixture took one 480 Hz exchange with 12 solver substeps on GPU 2. The CUDA process exited naturally with code 0. No Isaac instance ran in this test.

| Palm | Contact pairs | Active solver substeps | Inward force on palm (N) | Norm of force balance residual (N) |
| --- | ---: | ---: | ---: | ---: |
| Left | 7 | 1 | 4.251497 | 2.98e-8 |
| Right | 7 | 1 | 4.013423 | 1.33e-7 |

Raw pair forces and torques reconcile with the reported rod force and the palm wrench. The force and torque axes are expressed in world coordinates. Torque is about the origin of the individual palm mesh. Recomputing each pair moment from its contact point and force gives maximum errors below 0.37 μN·m about the mesh origin. Using the world origin as the reference instead gives errors of 7.62 N·m on the left and 13.32 N·m on the right. The installed solver source and the two preserved raw pair files support this correction. Identity, directions, six force/torque components and raw pair SHAs are in the readiness contract. The result index binds the actual output directory and the parent-process receipt. The initial tangent placement intentionally produces measurable isolated contact and is unrelated to a trained Power posture.

The CPU regression passed 113 tests. To run the verifier, use `python -B -m scripts.verify_dh116_power_palm_reactions --rays <palm_rays.json> --left-bake <left directory> --right-bake <right directory> --output <new directory>` in the current Ubuntu environment. Omit the output argument to perform only source and geometry preflight. The next verification must route each measured palm force and its center-of-mass moment into its named live PhysX articulation link and measure reaction there. Enveloping contact, lift, hold, slip and Teacher readiness remain unverified.

The post-publication joint CPU regression passed 341 tests, including the fixture moment-origin assertion and the work-package registry parser. The moment-reference correction was checked by replay of the existing pair files.
