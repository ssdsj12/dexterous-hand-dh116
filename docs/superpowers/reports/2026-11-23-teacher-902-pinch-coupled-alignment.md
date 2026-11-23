# 902 Pinch coupled wrist and digit alignment

The earlier independent frozen fits left the left thumb and index 0.192 / 0.144 mm from the supported rod at material station 36, and the right pads 0.431 / 0.215 mm away at station 4. This package fits around those two sealed poses using the same native SDK 424–432 approach and immutable side-specific geometry.

The fitter minimizes the larger of two pad gaps while jointly varying wrist translation, wrist rotation, thumb abduction and flexion, and index bend. The local basis follows the measured rod tangent and cross-section. Each proposed correction is checked again against the nonlinear pad and hull measurements, all nine rod frames, both supports, the table, and the original wrist workspace. A second phase supplements the sampled directions for selected-pad envelopes using certified convex-group or segment witnesses. Any added direction is evaluated against every vertex and every archived node of the group. Non-pad envelopes retain their original proof. The original 24D control is untouched.

| Side and station | Warm thumb / index gap | Coupled thumb / index gap | Pad normal dot | Minimum support gap | Source `finger13_link` non-pad rod gap | Source `finger14_link` selected-pad rod gap |
|---|---:|---:|---:|---:|---:|---:|
| Left, 36 | 0.192 / 0.144 mm | 0.082 / 0.077 mm | −0.863 | 1.102 mm | 0.255 mm | −0.149 mm |
| Right, 4 | 0.431 / 0.215 mm | 0.397 / 0.325 mm | −0.989 | 4.070 mm | 0.252 mm | −0.147 mm |

The maximum gap fell 57% on the left and 7.9% on the right. The right index gap increased as the coupled fit reduced the larger thumb gap. Each final fit passed the original whole-hull endpoint checks and failed the opposing-pad SDF gate because both gaps remained positive. The selected source thumb hull was the only guard crossed by rejected witness-phase trials on either side. At the final endpoints, proximal thumb reserve above 0.250 mm is 4.54 µm left and 2.35 µm right. Selected thumb hull reserve above −0.150 mm is 0.66 µm left and 3.45 µm right. These are measured-archive geometric bounds on the explored correction within the archived neighborhood.

A probe that also supplemented the non-pad witness reached 0.079 / 0.078 mm on the left but failed the unchanged original non-pad endpoint check at 0.243 mm. The final run supplements selected-pad directions exclusively and retains that rejected probe only as diagnostic evidence. No candidate entered command-path verification. A materially different follow-up is to screen another thumb pad facet and its neighboring wire segment for separation between the source hull and the pad SDF contact locus before choosing a guarded command route. Bilateral fixed-wrist force hold and support-free lift remain unverified.

Run in the existing standalone CPU USD environment recorded in `docs/contracts/dh116_wire_joint_pregrasp_v1.json`.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.fit_dh116_902_pinch_coupled \
  --receipt "$PINCH_APPROACH_OUTER" --warm-result "$SEALED_WARM_FIT" \
  --side "$SIDE" --iterations 32 --output "$FRESH_JSON"
```

Both commands returned 1 after recording the blocked endpoint results. The source outer SHA-256 is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7`. The terminal row SHA-256 is `ea6985622f9b45a280f005f56c5f48ba2cc2ab9482b62394632d31f7e0206888`. The fitter SHA-256 is `3a125e042e08742242813f77d961cfc3aa0d8ba636ffbe78f266e6a5234068fd`. The shared endpoint evaluator SHA-256 is `4526cd4834b7f74b8c8cfffd22f0a5564d853c4ddef141501471879baa3ab29f`.

| Archived output | SHA-256 |
|---|---|
| `sync-pinch-round3-coupled-left-v4.json` | `6c7f4c9f62ab1f936cf85218903cc5656eabba5c5fb06db1c6eb687bf3e99b73` |
| `sync-pinch-round3-coupled-right-v4.json` | `7fc275f62c076260f4c30a55790a76bbcb598bd1e9b5a05e4ed0537fd5ec38f1` |
| `sync-pinch-round3-coupled-left-v4-argv.json` | `4ebc98b0e10bb6f718440c2aaccb32bfff71ac5d67f1d2140b8f6d3f3c2f6925` |
| `sync-pinch-round3-coupled-right-v4-argv.json` | `d1aca3ec0810cb10f2094fbdc82cd17dbbf8c70d028e439602cec3972ea129e3` |
| `sync-pinch-round3-coupled-v4-cpu.log` | `c6c94d9988f76824727ca79987479361f40235bd8a06ab9966438c6438057684` |

The regression suite passed **171** tests, including the real warm replay and a check that supplemental directions account for earlier recorded rod frames. No Isaac run, force transfer, Teacher query, PPO update, or lift took place.
