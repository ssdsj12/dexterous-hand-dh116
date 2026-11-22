# Wire size and stiffness generalization

The opt-in v2 catalog keeps the nine existing size candidates and adds an independent stiffness pair to training. Validation and test each use a pair with a held-out size and held-out modulus scale. A pair changes axial and bending stiffness at fixed radius, length, density, gravity and contact coefficients. These are simulated elastic candidates. No manufacturer measurements were assumed.

Catalog preparation rejects duplicate physical specifications at solver precision, repeated geometry across splits and repeated modulus scales across v2 splits. It preserves the v1 catalog. Resized Isaac Reset resolves the v2 material through the same solver construction and SHA binding. The training-only pilot rejects held-out specimens. Catalog and spec identities remain in audit metadata.

## Standalone CUDA result

Three 10 mm × 500 mm wires were constructed on a GeForce RTX 4090 with the existing backend. The getter measured 32.50 g for each, actual matching radius and rest length, and the following distinct parameters.

| Spec | EA (N) | EI (N·m²) |
| --- | ---: | ---: |
| Soft | 640 | 0.0144 |
| Reference | 800 | 0.0180 |
| Stiff | 960 | 0.0216 |

The constructor check ran zero solver steps and zero Isaac steps. Physical response, grasp and trained generalization are pending.

## Training and evaluation

For each of Pinch, Tripod and Power, check approach and same-clock finger or palm loads, establish opposing contact, then verify actual cable Lift after complete support separation. The adaptation evidence must include palm-relative wire position and tangent, each expected pad's normal load and reaction vector, command and tracking response, and the validity of the fixed-anchor slip measurement. Use only the most recent completed physical interval when constructing the actor observation. Keep raw catalog identities and exact stiffness on the measurement and evaluation side to prevent the actor from relying on specimen identity.

Evaluate a frozen policy on wholly held-out wire specifications and independent physical initial conditions. Report per-grasp and per-spec success, corrective wrist and finger increments, opposing loads, support load, net Lift, Hold and Slip. Keep 30 mm, 5 s and 0.5 mm as the acceptance thresholds. A fixed-size material pair isolates whether the policy adapts its grasp to stiffness. Score the worst tested specification alongside the aggregate.

Antigravity confirmed the need for solver-precision material identity and previous-interval force. Codex verified the catalog and both input readers independently. The current feedback collector records palm geometry and reactions, with actor integration and end-to-end trials still pending.
