# Fixed-wrist held-out reset preparation

Each grasp uses the same nominal left and right wrist poses and initial finger joints over eight unseen combinations of diameter, length and modulus. Thirteen matched horizontal offsets and yaw values per combination yield 104 distinct physical setup identities per grasp. The replay plan moves the wire and the centres of its two axis-aligned supports together. Both supports remain seated on the original table, each under its own half of the wire. Only the evaluator receives the trial and solver identities.

The CPU command `python -B -m scripts.prepare_dh116_wire_holdout_resets --scene ... --catalog configs/dh116_training/supported_wire_generalization_factorial_holdout_v1.json --placements configs/dh116_training/supported_wire_holdout_placements_v1.json --left-material-id ... --right-material-id ... --output ... --audit-hulls` keeps the full cohort. The audited envelopes comprise 34 source USD convexes and ten locked cooked pads in predicted reset frames. It records unresolved continuous distance queries as failures.

| Grasp | Static clearance | Wire gap not proved | Support gap not proved | Unresolved wire queries, by trial |
| --- | ---: | ---: | ---: | ---: |
| Pinch | 24/104 | 80 | 0 | 2 |
| Tripod | 72/104 | 32 | 0 | 0 |
| Power | 0/104 | 104 | 48 | 8 |

The Power reset needs a common open-hand starting pose before its whole-hand closure can be tested across this cohort. Pinch and Tripod likewise need nominal pre-grasp poses and measured approaches that cover their rejected initial conditions. Select those starting poses with training specimens and keep the held-out matrix fixed. At native Reset, verify actual link poses, both support reactions and the solver specification. Compare frozen policy runs with full feedback and separate pose or force ablations on the same initial conditions.

The current package contains CPU reset candidates and clearance evidence. Native held-out Reset, Teacher optimization, independent Lift and any generalization pass remain pending. Formal success still requires actual off-support Lift of at least 30 mm, Hold of at least 5 s, and no more than 0.5 mm Slip.
