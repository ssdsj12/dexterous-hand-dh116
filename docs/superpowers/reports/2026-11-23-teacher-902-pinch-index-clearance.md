# 902 Pinch staged index clearance

The R5 primary left-38 and right-4 endpoints remain the target pair. At the archived R5 direct route's first failure, SDK-style issued substep 210, the left source index `finger23_link` approached material segment 36. Rechecking the emitted float32 wrist and finger targets with all SDK 424–432 rod frames reproduced the finite-segment interval **−0.152013 mm**, beyond the original −0.150 mm selected-pad limit.

| Left perturbation at the original blocker | Source index rod gap | Whole-hand endpoint geometry |
|---|---:|---|
| Index bend −0.03 rad | +2.506 mm | Clear |
| Index bend −0.06 rad | +5.336 mm | Clear |
| Index bend −0.09 rad | +8.265 mm | Clear |
| Index bend +0.03 rad | −2.718 mm | Selected pad and non-pad rod guards fail |
| Wrist roll −0.08 rad about the measured rod tangent | −6.897 mm | Selected pad and non-pad rod guards fail |
| Wrist roll +0.08 rad about the measured rod tangent | +9.936 mm | Another selected pad and non-pad rod guards fail |

Opening the measured index before wrist alignment preserves reserve on the limiting link while protecting the rest of the hand. The staged command first withdraws the index at the archived issued wrist target, then aligns with that joint reserve, and finally restores bend at the qualified fixed wrist. The source opening and terminal seating were prescreened along five index-only fractions for each eligible offset.

The first two left-only replays passed the old left blocker and reached the same new right issue 230 SDK substeps into wrist alignment. The −0.03 rad mode stopped at global substep 294 and the −0.06 rad mode at 342 because their initial openings took different times. Both issued the same right state at the stop. Right `finger23_link` had original full-envelope lower bound −0.165630 mm and feasible finite-segment upper bound −0.146913 mm. The guard classified it as unresolved. The archived left-only proof is `sync-pinch-round6-index-v1.json` with SHA-256 `deb5e2638727c9c22099f5bece9ae7493a2b77fc28ec0e48c4446c4de856becb`.

## Paired index reserve

An independent segment check on the exact right state from the −0.03 rad left-only replay reached **−0.152213 mm** at archived SDK 424, segment 4. The original combined envelope returned an unresolved interval at that pose. The earlier-frame segment witness separately establishes a crossing of the −0.150 mm selected-pad guard. Opening the right index by −0.03 rad at the same issued wrist adds reserve and yields +2.516 mm clearance for source `finger23_link`, with all original hull checks passing. Right −0.06 and −0.09 rad also passed the static trials. The paired replay uses the smallest safe prescreened index reserve, −0.03 rad on each side, and preserves both original wrist goals.

| Phase | 60 Hz command windows | SDK substeps |
|---|---:|---:|
| Withdraw both indices at the archived issued wrists | 8 | 64 |
| Align with the index reserves | 142 | 1,136 |
| Seat both indices at fixed wrists | 8 | 64 |
| Total | 158 | 1,264 |

The paired command sequence completed all **1,264** original 24D issued substeps without an endpoint guard stop. First right thumb SDF contact appeared at node 4 on substep 1,185. The smallest cross-hand Y separating bound was **289.495 mm**, checking all source and cooked hulls on both sides. The left-38 and right-4 terminal contact nodes remained exclusive on every archived rod frame. Terminal gaps across the nine frames were −0.03036 to −0.02513 mm left and −0.03027 to −0.02358 mm right. Starting conditions use the archived measured wrist readback and previously issued backend and projected targets. Maximum terminal deviation from the prescribed targets was 22 nm position, 4.67e−8 rad wrist orientation, and 1.75e−6 rad active joints.

This proof uses the exact original command kernels and float32 submitted targets with ideal tracking and the fixed nine-frame archived rod envelope at each checked SDK boundary. Swept-volume validation, live wrist and finger tracking, opposed force, reaction delivery, fixed-wrist hold, and lift remain open. No Isaac, GPU, or native solver action was taken.

## Reproduction

Use the CPU USD environment recorded in the regression and replay argv files. The optional right blocker takes the previously sealed left-only evidence and runs just the paired mode.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.certify_dh116_902_pinch_index_clearance \
  --receipt "$PINCH_APPROACH_OUTER" --left "$LEFT_RELIEF_JSON" --right "$RIGHT_RELIEF_JSON" \
  --blocker "$R5_ROUTE_JSON" --right-blocker "$R6_LEFT_ONLY_JSON" --output "$FRESH_JSON"
```

The left-only command exited 1 after the two guard stops. The paired command exited 0. The source native receipt SHA-256 is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7`. The staged replay source SHA-256 is `f9faf3bb2b4c214c6700af5e82889b425bcfade458250d38df7d1abc55d9af0b`. The original command-step SHA-256 remains `f0ccc75e17bdc93894988ec65afd82562162002334b3329d985ddafa11a0d114`. SHA-256 of the three paired command phases is `7357136f4ef4b348848895080d138c8102064f591e6dd3d49c8170c01a9f36f2`.

| Evidence | SHA-256 |
|---|---|
| `sync-pinch-round5-relief-route-v2.json` | `593a9fc121306f04313c9e21ad5d440d5d4ca851896e62260b3c8cea7bff3d76` |
| `sync-pinch-round6-index-v1.json` | `deb5e2638727c9c22099f5bece9ae7493a2b77fc28ec0e48c4446c4de856becb` |
| `sync-pinch-round6-index-v2.json` | `4bb06240a9eef836b53503ef4782a947e4e93089c7bda7d44f1d6db208ddd1a1` |
| `sync-pinch-round6-index-v2-argv.json` | `a160e365bc12792b847a832e77114c12d00e74f4dea1c24dedf9dca36872366e` |
| `sync-pinch-round6-index-v2-cpu-argv.json` | `b06484f41c8e7cef9a9e03d270366fc89f892cb70e551e5eae89825ae5ce3486` |
| `sync-pinch-round6-index-v2-cpu.log` | `b9f0ca08b05f392043ee7e635cfb8d8c7788da08111571fb9853a16259f36199` |

The CPU regression suite passed **186 tests**, including the sealed first-block pose, stable fixed-wrist restoration, and independent left/right bend staging.
