# Right Tripod reaches the near-rod approach endpoint

The same staged geometry and motion were used after inserting three zero-action windows before approach. Removing the inserted rows recovers the original 62 controls and all original target knots. The gate still uses eight consecutive, actual samples from this instance and the previous position and velocity limits. The new recipe binds the earlier natural exit 1 receipt and the saved speed-only failure measured at SDK 330.

| Measured gate | Original settle | Extended settle |
| --- | ---: | ---: |
| SDK step | 330 | 354 |
| Peak right wrist angle error (rad) | 0.000132229 | 0.000080083 |
| Peak right wrist angular speed (rad/s) | 0.001340168 | 0.000811673 |
| Gate with 0.0002 rad and 0.001 rad/s limits | Fail | Pass |

The right approach then completed. The measured ledger contains all 520 physical samples, 65 control windows and 6,240 DeformX substeps. Every sample preserved source/cooked clearance and zero non-table contact pairs. The minimum right table clearance was 2.475 mm and the minimum right non-target rod clearance bound was 29.935 µm at source finger13, SDK 516. The final conservative cooked gaps were 0.469 mm (thumb), 0.527 mm (index), and 0.459 mm (middle). These are clearances from original collider envelopes at the measured link poses.

The left hand stayed parked. No right pad touched the rod, and no closing, lift or PPO took place. The 29.935 µm non-target margin constrains the next step. Screen a separate three-pad closing path from the actual terminal geometry before commanding an additional motion.

App.close returned and outer exit was 0 without forced termination. The 298-test CPU regression included checks for inserting only stationary windows and rejecting an old-clock window at the new gate. The exact worker is `python -B -m scripts.run_dh116_tripod_near_settle_extension`. The [readiness record](../../contracts/dh116_teacher_tripod_near_settle_extension_v1.json) retains both run identities and all file hashes required for replay.
