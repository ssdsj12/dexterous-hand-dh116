# Power reacquires a right-hand partial opposition endpoint

The 0/104 figure cited by the earlier Tripod report came from the original Power pose. The later [common pregrasp](../../contracts/dh116_teacher_wire_common_pregrasp_v1.json) already cleared 104/104 static Power starts across eight held-out wire specifications and 13 placements per specimen. The release-bound holdout matrix has 104 unique physical initial conditions. Its minimum margins are 7.772 mm above the original table threshold, 5.316 mm from the supports, and a conservative 3.656 mm wire gap. These figures cover initial geometry only.

The one-sided contact result below uses the original measured 10 mm wire at SDK 6504 and rod tick 78024. The new `--free-wrist-refit` option frees wrist translation within the original box and retains the original coupled joint limits. Both pads retain their assigned material nodes and can slide on their locked SDFs. The fit minimizes penetration at other nodes, including around the palm. The bounded numerical extension outside the SDF lattice helps the optimization return to the grid. The final gate reads the original locked data and checks material-node ownership, physical overlap limits, actual SDF normals, and original source and cooked collision envelopes.

Before the palm residual was added, six right-hand attempts cleared the original hull rules and both fingertip stations but also put the palm on node 8. Once the fitting residual included that neighbor, the six regained their three assigned stations and failed at the original cosine threshold of −0.25. Rejected fits then triggered a soft correction based on the current SDF gradients. The final candidate passed the unchanged hard check.

| Hand | Refit attempts | Locked SDF or station | Rod envelope | Opposition | Accepted static |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left | 30 | 20 | 10 | 0 | 0 |
| Right | 12 | 6 | 3 | 2 | 1 |

The right-hand candidate binds palm node 9, thumb node 4, and ring fingertip node 3. The measured-from-SDF gaps are −50.10, −49.55 and −43.80 µm, with one negative node per part. The true locked SDF normal dot product is −0.259561. The table and support margins are 33.698 mm and 14.690 mm. On the left, all 20 station rejections include the thumb. Refit states that passed the station checks still failed the original rod envelope.

The next step is to use the accepted right-hand waypoint in a continuous guarded approach and solve the left thumb geometry independently while retaining both sides' original constraints. The partial waypoint keeps required palm contact. Full Power still requires five-finger wrap and live opposing reactions, followed by unsupported Lift ≥30 mm, Hold ≥5 s and Slip ≤0.5 mm. The existing fixed RGB cameras are outside the scope of this CPU contact search.

The Ubuntu related CPU suite passed 196 tests. The archived search correctly returned exit code 1 because there was no bilateral candidate. There were zero new Isaac runs and zero PPO updates. The [readiness record](../../contracts/dh116_teacher_power_free_wrist_refit_v1.json) records the data and source SHA-256 values and the exact replay command.
