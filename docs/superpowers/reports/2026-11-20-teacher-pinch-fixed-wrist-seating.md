# Pinch fixed-wrist finger seating

This trial extends the original 45 control windows. At SDK 362, both wrists and the right fingers stayed fixed. Left finger11 and finger21 received target increments of −0.003 and −0.0005 rad over two windows, followed by eight steady commands. The routine preserved the original joint limits and action rate. Contact gates passed at SDK 202, 282, and 362. The last gate re-certified the measured position, command and tracking interval for each side. Antigravity checked the frozen source SHA, gate order, action indices and inactive-axis tracking, which were independently verified before publication.

The native run completed 55 control windows, 442 SDK steps including two setup steps, and 5,280 DeformX substeps. No contact guard fired. App.close returned and the outer process exited naturally with code 0. The physical ledger SHA was independently recomputed and agreed with the outer receipt.

| Same-run interval | Both hands signed | Both hands above Pinch load and opposition gates | Longest qualifying run | Table pairs |
| --- | ---: | ---: | ---: | ---: |
| SDK 283–362, fixed-wrist baseline | 960/960 | 473/960 | 16 substeps | 36,585 |
| SDK 363–442, added finger seating | 585/960 | 390/960 | 21 substeps | 37,176 |
| SDK 299–362, baseline settled | 768/768 | 420/768 | 16 substeps | 29,382 |
| SDK 379–442, after new ramp | 470/768 | 292/768 | 15 substeps | 29,707 |

Four pads remained above 0.05 N in every compared substep. The left thumb's median normal load increased from 0.148 to 0.170 N. All 375 lost signs were left-sided. The left thumb net force projected outward in 369 of those samples, although its normal component pointed inward in all 375. The contacting node identities stayed at index 34 and thumb 35. The two increments ran together, so this replay does not identify their individual effects. The minimum source clearance from the rod increased at left finger13 from 1.45 to 166.36 µm and shifted at finger22 from 157.87 to 153.49 µm. Measured table clearance stayed above the 2.10 mm guard floor.

Keep the earlier fixed-wrist trajectory as the stronger opposition baseline. This seating candidate stays outside the grasp qualification path. A future candidate must preserve the thumb net force projection while improving continuous bilateral load; table contact was present throughout these measured intervals. No lift or PPO update was attempted.

CPU regression passed 292 tests. The exact native worker is `python -B -m scripts.run_dh116_pinch_fixed_wrist_seating`. Original bounded inputs, source stamps, raw ledger and native close markers are retained under the receipt in the [readiness record](../../contracts/dh116_teacher_pinch_fixed_wrist_seating_v1.json). The CPU reduction used `inspect_interval` to verify issued targets against the complete frozen action schedule and the existing `audit_dh116_pinch_thumb_projections.measure_step` routine for force decomposition.
