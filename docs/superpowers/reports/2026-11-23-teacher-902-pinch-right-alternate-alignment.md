# Alternate right-hand 902 Pinch alignment

The earlier bilateral endpoint fit left the right thumb and index 0.645 mm and 0.024 mm from the rod at material station 4. The right source thumb-proximal link `finger13_link` had 0.276 mm rod clearance, 0.026 mm above the 0.250 mm guard. This CPU replay tests whether moving to adjacent material or changing the wrist/thumb approach provides enough room for both pads to reach opposing contact.

The search starts from the sealed, naturally exited 902 Pinch approach, SDK 424–432. It moves the station 3 and 5 seeds by their measured rod-node displacement from station 4 and compares nominal and signed tangent-roll seeds at all three stations. The alternate starts roll by ±0.22 rad and vary the two thumb joints. The fit keeps non-grasp finger targets fixed and intersects local search spans with the original wrist workspace and finger limits. Each endpoint rechecks all 17 source hulls, five locked cooked pad hulls, all nine rod frames, both support boxes, and the table. Both pads must find the proposed material station on the right side and have negative SDF gaps with the original penetration and normal opposition limits.

| Right material station | Best alternate seed | Thumb / index gap | Normal dot | Minimum support gap | Limiting non-pad rod clearance |
|---|---|---:|---:|---:|---:|
| 3 | Positive tangent roll | 0.766 / 0.253 mm | −0.524 | 1.053 mm | 0.275 mm at source `finger13_link` |
| 4 | Material translation | 0.679 / 0.127 mm | −0.511 | 3.603 mm | 0.275 mm at source `finger13_link` |
| 5 | Material translation | 0.785 / 0.331 mm | −0.486 | 14.305 mm | 0.274 mm at source `finger13_link` |

All nine alternates passed the endpoint hull checks and failed the opposed pad SDF check. The three listed finalists also placed the index source hull within 0.029 mm of the 0.150 mm maximum penetration. Station 3 left 0.053 mm above the 1.00 mm support reserve. Station 5 gained 10.143 mm of support clearance over the earlier station 4 fit, but increased both pad gaps. The scan neither established a qualified right endpoint nor opened a command-path proof. The earlier right-thumb gap remains the comparison baseline for these nine cold starts.

Reproduce in the standalone CPU USD environment recorded in `docs/contracts/dh116_wire_joint_pregrasp_v1.json`. The actual command and environment are recorded in `sync-pinch-round2-alternate-v2-cold-argv.json`.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.search_dh116_902_pinch_right_alignment \
  --receipt "$PINCH_APPROACH_OUTER" --radius 1 --max-evaluations 48 --output "$FRESH_JSON"
```

The bounded sweep returned exit 1 with its completed, blocked evidence in `sync-pinch-round2-alternate-v2-cold.json` (SHA-256 `8c7a411113059569bf408258eee09c35d581ac8c9efbb84256bd6f89622e6bf4`). Its source outer SHA-256 is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7`.

## Warm continuation of the sealed station 4 fit

A second screen imported the SHA-verified earlier fit from the same archive and varied its wrist about the measured rod tangent by ±0.06 rad. The corresponding thumb flexion seed changed by ±0.015 rad. The best candidate used the negative tangent roll, reduced the larger pad gap from 0.645 mm to **0.431 mm**, and increased the index gap from 0.024 mm to **0.215 mm**. The pad normal dot reached −0.989. The source thumb-proximal `finger13_link` remained limiting at 0.279 mm, with 0.029 mm above the rod guard. The source `finger14_link` limited selected-pad rod clearance at −0.123 mm. Source `finger22_link` limited support clearance at 4.263 mm. Both pad gaps remained positive, so the warm candidate also failed the opposed pad SDF check.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.search_dh116_902_pinch_right_alignment \
  --receipt "$PINCH_APPROACH_OUTER" --baseline-result "$FIRST_ROUND_FIT" \
  --radius 0 --max-evaluations 48 --output "$FRESH_WARM_JSON"
```

The warm continuation returned 1 with complete evidence in `sync-pinch-round2-alternate-v2-warm.json` (SHA-256 `14d431563ee931463952fb27773b84a85f749c387711d025fa1279d8afc405d3`). Its environment and command are in `sync-pinch-round2-alternate-v2-warm-argv.json`. The final CPU regression passed **168 tests**. The log SHA-256 is `0a083e933776a2947990e584bd277011a60c1597254a36f8fdb6f0ecadbce279`.

No new Isaac run, issued-action trajectory, force transfer, or lift was attempted. The next right-hand alignment needs to close both remaining SDF gaps while preserving the thumb-proximal and selected-pad hull margins. A later command-path screen and native force verification remain prerequisites for a bilateral fixed-wrist hold and support-free lift.
