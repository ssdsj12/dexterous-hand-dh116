# Power pad clearance refit on the measured flexible rod

The refit starts from the existing sealed Power replay at SDK step 6504 and rod tick 78024. The rod centerline, collision sources, calibrated SDFs, and hand identities come from that replay and its bound audits. The optimization changes candidate wrist orientation and six coupled active joint targets. It keeps the original wrist workspace, material station ownership, opposition, hull clearance, table and support rules.

The optimizer refreshes conservative separating directions from the limiting rod segments between rounds. It includes both endpoints of each capsule segment in the support residual. The optional sliding fit lets the assigned pad move along its surface at the same material node. Nearby negative samples are penalized, and every candidate is rechecked with the original locked SDF and full raw and cooked geometry. The final check independently rejects invalid state, wrist overtravel, and infeasible coupled joint targets.

| Fit | Hand | Seeds | Rounds | Rejections | Accepted |
| --- | --- | ---: | ---: | --- | ---: |
| Fixed contact rays | left | 8 | 24 | rod_envelope 16, wrist_workspace 8 | 0 |
| Fixed contact rays | right | 2 | 6 | locked_sdf_or_station 5, wrist_workspace 1 | 0 |
| Sliding pad on the owned node | left | 8 | 24 | locked_sdf_or_station 6, not_opposed 2, rod_envelope 16 | 0 |
| Sliding pad on the owned node | right | 2 | 6 | locked_sdf_or_station 6 | 0 |
| Sliding pad with explicit final limits | left | 8 | 24 | locked_sdf_or_station 6, not_opposed 2, rod_envelope 16 | 0 |
| Sliding pad with explicit final limits | right | 2 | 6 | locked_sdf_or_station 6 | 0 |

The fixed-ray refit brought one right thumb cooked-hull feasible upper bound to −0.15228 mm against the existing −0.15 mm condition. That pose also lost its required ring-pad contact. With sliding enabled, all six examined right-side fits retained the thumb and palm stations while the ring pad remained separated. On the left, five of the six station rejections lost the index-pad contact, four lost palm contact, and one assigned an additional node to the palm. Left-side full-envelope failures also remain. These outcomes favor a wider hand-and-rod alignment search that preserves all three contact constraints.

Antigravity independently read the implementation and confirmed that the soft residuals cannot grant admission. Its recommendation to validate wrist and coupled limits directly at the final gate is covered by a failing-then-passing regression.

The recorded rod is one geometry and state. These tests do not evaluate load, approach, release from support, lift, hold, slip, PPO, or generalization. The next Power candidate needs a continuous route from the measured boundary and native contact evidence. An eventual multi-specification experiment must reuse the held-out catalog for diameter, length and stiffness. Use measured centerline and tangent for local pose correction, and qualified paired rod and hand reactions for bounded force adjustment at each control step. Evaluate each grasp and wire specification independently against lift ≥ 30 mm, unsupported hold ≥ 5 s, and slip ≤ 0.5 mm.

The Ubuntu CPU regression passed 183 tests. The search exit code was 1, which indicates that no bilateral candidate passed. The readiness record links the replay fingerprints and exact commands.
