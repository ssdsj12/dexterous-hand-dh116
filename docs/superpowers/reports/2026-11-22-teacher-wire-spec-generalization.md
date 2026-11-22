# Wire specification and force-aware Teacher generalization

The opt-in catalog produces nine training candidates and two independent specifications for each of validation and test. Spec SHA includes the entire solver material and length contract. Replays of one spec belong to the same split. Validation/test lengths and diameters are both held out from training. These are deliberately small candidate cohorts derived from the existing baseline. Different manufactured materials and unseen physical initial conditions still require their own sampling and validation.

We keep 41 material stations, the original hands and the 24D action contract. The scene generator moves supports along the wire's reference material axis and reanchors each hand's initial wrist to its own declared material station. Uniform circular wire candidates use area scaling for linear density and axial stiffness and fourth-power scaling for bending stiffness. The original parameter and scene files are untouched. Geometry preparation preserves the table/support elevations and checks each support remains under a material station. It grants no native Reset or grasp qualification.

## Actual standalone CUDA and recorded Isaac source

RTX 4090 / GPU2 used the existing DeformX binary `88170101…`. Three independent free-rod constructions passed zero-step parameter getter checks:

| Diameter / rest length | Measured total mass | EA | EI |
|---|---:|---:|---:|
| 9 mm / 450 mm | 23.6925 g | 648 N | 0.01181 N·m² |
| 10 mm / 500 mm | 32.5000 g | 800 N | 0.01800 N·m² |
| 10.5 mm / 475 mm (test spec) | 34.0397 g | 882 N | 0.02188 N·m² |

Model-free measurements at a recorded Isaac SDK4 endpoint link the actual wire's material stations to each palm. The adapter publishes metric position and unit tangent in that hand's palm frame and complete per-pad mean/minimum reaction, including the solved force direction on the object. Force can also be expressed relative to verified wire weight, so mass is explicit and a fixed Newton target need not be reused across diameters. The existing three grasp-specific reductions remain separate. Power retains its own palm and five-finger records and rejects crossed material-side support. The old sample has zero hand reaction. It proves the readback path only.

## Native PPO integration required

Antigravity read the frozen builder, solver, native getter and model packer. Codex verified their source locks. Radius, length and mass are still fixed in the original Isaac Teacher path. New shapes must use their own solver construction and a recorded native Reset, then a verified state packer. Group rollout environments by physical specification because the radius and contact coefficients are set on RopeBatch. Keep the original camera and model contracts available.

Train the three grasps on real normal-force and pose feedback over the train specifications, varying the mounting and Centerline pose independently. Require stable opposite forces before lifting the wire clear of both supports. Evaluate 30 mm Lift, 5 s Hold, 0.5 mm Slip, success rate by grasp/spec, peak and minimum pad load, and pose corrections on entirely held-out settings. No multi-spec Isaac rollout, PPO training, or generalized grasp has been validated yet. The nominal left-wrist diagnostic did not qualify stable grasp.

## Review and regression

Antigravity matched both implementation file hashes in the follow-up. The final feedback adapter records equal and opposite force on the hand in the same palm frame. Wrong-side and unexpected contact preserve their raw force and mark reaction feedback invalid. The Power side check includes the actual registered proximal finger52 contacts. Geometry spec identity rejects exact duplicates at float32 solver precision. The standalone reader rejects duplicate catalog names or hashes. The final CPU/Registry regression passed 116 tests in 79.81 s. Three final CUDA getter checks passed again after these changes.

The previous nominal wrist dose was tracked to (+25.03, −124.89) µm by SDK2192, despite reaching only half its target after the first control. Tripod still lost reaction in the settled interval. The available evidence supports closing the loop on actual finger load and geometry rather than increasing a fixed positional dose.
