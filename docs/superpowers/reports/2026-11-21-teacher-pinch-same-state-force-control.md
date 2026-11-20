# Pinch same-initial-state force hold control

The zero-action run used 55 control windows. The first 45 replayed the exact previous action schedule. At SDK 362, the baseline and both earlier candidates matched exactly in measured q, qd, joint targets, all 41 rod node positions and velocities. The next ten baseline control windows were entirely zero. The original SDK 202, 282 and 362 gates passed, and the 2.10 mm table clearance and original joint protection stayed active. App.close returned and the native worker exited naturally with code 0.

| Same SDK 379–442 window | Both hands signed | Both hands clear all original load and opposition gates | Longest qualifying sequence (solver substeps) | Table pairs |
| --- | ---: | ---: | ---: | ---: |
| Zero-action hold | 733/768 | 197/768 | 14 | 29,613 |
| Left finger21 +0.0001 rad | 729/768 | 293/768 | 17 | 29,483 |
| Left thumb and index redistribution | 470/768 | 292/768 | 15 | 29,707 |

The shared SDK 299–362 prefix qualified 420/768 substeps. The unchanged hold fell to 197/768, establishing that deterioration also took place with no new command. The single-axis candidate qualified 96 more substeps than its matched control, a 12.5 percentage point increase on this trajectory. It retained substantially more signed opposition than the compound intervention. It remains useful for a force-feedback candidate. Its longest run was about 2.95 ms at 5,760 solver substeps per second, and table load remained present. All three trials use the same initial condition family and supply no estimate of generalization.

The zero-action worker completed 442 SDK steps and 5,280 solver substeps. No lift or PPO update was attempted. The 262-test CPU regression passed. Run with `python -B -m scripts.run_dh116_pinch_fixed_force_hold_control` and the exact previous contact prefixes. The [readiness record](../../contracts/dh116_teacher_pinch_same_state_force_control_v1.json) pins the comparison file, natural outer receipt and raw ledgers. Continue using same-time controls to evaluate load feedback before allowing a lift action.
