# 902 Pinch alignment from the measured approach

The last verified native 902 Pinch approach ended naturally at SDK 432. Both end supports remained loaded and the hand-contact ledger contained zero pairs. The new CPU fitter replays the sealed SDK 424–432 tail and searches each hand's measured wrist and thumb/index posture within the original workspace. Its search keeps the other three joint targets fixed. Each endpoint is checked against all 22 original source and locked cooked hulls per hand, the full nine-frame rod envelope, both support boxes, and the table.

The two-pad geometry check requires both original pad SDF gaps to be negative without exceeding 0.15 mm penetration, at one measured material station on the original hand's side. The opposing SDF normals must have a dot product below −0.2. The independent non-pad rod reserve is 0.25 mm, the support reserve is 1.00 mm, and the table reserve is 2.10 mm. This diagnostic matches the common physical clearance checks. Its shared-station test targets the observed Pinch material alignment.

| Side | Maximum thumb gap on captured SDK 424–432 | SDK 432 thumb / index | Refit thumb / index | Station | Refit normal dot |
|---|---:|---:|---:|---:|---:|
| Left | 1.550 mm | 1.418 / 1.504 mm | 0.192 / 0.144 mm | 36 | −0.870 |
| Right | 1.685 mm | 1.581 / 1.454 mm | 0.645 / 0.024 mm | 4 | −0.667 |

| Side | Refit limiting selected-pad rod hull and lower bound | Refit limiting non-pad rod hull and lower bound | Refit minimum support clearance |
|---|---|---|---:|
| Left | Source `finger14_link`, −0.122 mm | Source `finger13_link`, +0.422 mm | +11.209 mm at source `finger23_link`, support 1 |
| Right | Source `finger23_link`, −0.120 mm | Source `finger13_link`, +0.276 mm | +4.162 mm at source `finger22_link`, support 0 |

Both finalists pass the endpoint hull checks and fail the opposed pad SDF check. The right thumb still has the largest remaining gap. The right source thumb-proximal link `finger13_link` has 0.026 mm reserve above its wire clearance guard. On this archived tail, the left limiting support clearance increases by 9.754 mm and the right limiting support clearance decreases by 1.362 mm. The geometrical results are endpoint estimates. Command tracking, inter-hand clearance along a new route, and force transfer require their own verification.

The previous native right index seating run reached at most 0.144 N on the weakest substep of its best SDK row and produced zero loaded bilateral ticks. The older staged-left run stopped at the original wrist workspace floor at SDK 1792. These are separate historical attempts. This package generated no new Isaac steps, force or lift evidence, Teacher queries or PPO updates.

Reproduce from the repo root in the standalone CPU USD environment recorded in `docs/contracts/dh116_wire_joint_pregrasp_v1.json`, replacing its worktree paths with the current worktree.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.fit_dh116_902_pinch_alignment \
  --receipt "$PINCH_APPROACH_OUTER" --output "$FRESH_JSON" --starts 3 --max-evaluations 48
```

The sealed source outer SHA-256 is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7`. The terminal row SHA-256 is `ea6985622f9b45a280f005f56c5f48ba2cc2ab9482b62394632d31f7e0206888`. The CLI returned 1 with the full blocked result in `sync-pinch-902-alignment-v2.json` (SHA-256 `2328a0643f260c54517c76540d196e3f2d5d616162c3454cc1c4f40eeaf58949`). Readback of the canonical final window is in `sync-pinch-902-canonical-final-window-v1.json` (SHA-256 `78f4ea3b61c2ab465c0e83a13335adcbbac99b174381ec3e3ec964dcb32874cc`).

The CPU regression suite passed 164 tests including direct replay of the sealed archive. The suite and fit commands are recorded in `sync-pinch-902-alignment-v2-argv.json`. The log `sync-pinch-902-alignment-cpu-v2.log` has SHA-256 `6ae3d795a9c0b9d7353f43321638118df0664252e2821320eefea81b6eb04800`.

Next, search a different thumb approach direction or neighboring material station, then certify the full issued 24D command path and same-clock opposed load and reaction on each hand. Bilateral fixed-wrist hold remains a prerequisite to independent support-free lift acceptance at 30 mm, 5 s, and 0.5 mm slip. The fixed RGB configurations are bound by the existing Reset loader.
