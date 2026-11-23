# 902 Pinch facet and rod-envelope feasibility

The CPU screen replays both sealed round-three endpoints against the original SDK 424–432 archive. It pairs alternate face-centre and guarded pad rays from the locked SDF and cooked hulls at fixed side-specific finger postures. Each proposed wrist aims the two selected SDF loci along neighboring material, including exact node alignments. Axial spins of 0°, ±10°, ±20°, and ±40° preserve the intended two loci while moving the wrist. The original workspace filters proposals before complete endpoint checks of all 22 source and cooked hulls on the nine archived frames, both supports, and the table. The endpoint contact gate requires one common negative material node with opposing SDF normals.

## Source and cooked witnesses at the sealed endpoints

Each active facet was mapped from certified finite-segment distance over four neighboring rod segments across all nine frames. The column at the SDF locus gives signed capsule clearance at the rod centre obtained by moving along the local SDF normal to `phi = 5 mm`. Face normals localize geometry. Opposition is measured from the locked SDF gradients.

| Side and pad | Source facet and nearest rod segment start | Source facet dot SDF normal | Current source / cooked gap | Source / cooked gap at SDF locus |
|---|---|---:|---:|---:|
| Left thumb | 598, 35 | 0.999878 | −0.148 / +0.070 mm | −0.190 / −0.007 mm |
| Left index | 1020, 35 | 0.995089 | −0.136 / +0.006 mm | −0.037 / −0.036 mm |
| Right thumb | 339, 3 | 0.996645 | −0.147 / +0.348 mm | −0.538 / −0.036 mm |
| Right index | 532, 3 | 0.999069 | −0.038 / +0.323 mm | −0.307 / −0.0004 mm |

The right thumb source hull exceeds its original 0.150 mm penetration limit by 0.388 mm at the measured SDF onset. The source and cooked envelopes differ by 0.502 mm at that same point even though the active source facet normal and the SDF normal agree within about 4.7°. The left thumb exceeds the limit by 0.040 mm along its original witnessed SDF normal.

## Alternate facets

The fitter screened 64 of 112 compatible pairs on the left and all 53 compatible pairs on the right. Original-workspace filtering left 237 left and 101 right proposals. The 24 strongest shortlisted poses per side received full geometry checks. All 48 failed the original selected-pad rod envelope. Twelve left finalists had simultaneous negative pad SDF at a sampled continuous rod point within the 0.150 mm SDF limit. No right finalist did.

One left candidate attained negative gaps **−0.041 / −0.030 mm**, SDF normal dot −0.896, and exclusive node lists `[36]` for both pads. It used cooked faces 42 and 58, target segment 35 at fraction 0.75, and −10° axial spin for a 10.812° net wrist change. Its support clearance was 7.142 mm and non-pad rod clearance 1.701 mm. The original selected-pad gate rejected the source thumb hull. An independent finite-segment replay localized source facet 605 at SDK 424 on segment 35 with a signed capsule gap of **−0.384479 mm**, 0.234 mm past the guard. The left result isolates the geometry tradeoff at an exclusive, opposed SDF node.

The closest right common continuous sample among the finalists retained **+0.262 / +0.289 mm** pad gaps. It used cooked faces 61 and 14, target node 2, and +10° axial spin for a 23.219° net wrist change. At those isolated SDF loci, the source-hull gaps were −0.517 mm at the thumb and −0.328 mm at the index. The thumb ray remained within 0.72° of the original SDF normal, so this alternate wrist pose did not move contact to a distinct thumb surface region.

A geometric follow-up can screen a thumb region with lower source protrusion, then recheck the neighboring segment throughout approach. On the left it must remove the facet 605 collision without losing the exclusive node 36 SDF pair. On the right it needs a genuinely different thumb ray from the nearly collinear face 61 ray. The endpoint sweep leaves both command paths unqualified, and no Isaac run, native force transfer, or lift was executed.

## Reproduction

Use the standalone CPU USD environment from `docs/contracts/dh116_wire_joint_pregrasp_v1.json` with the current worktree paths.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.screen_dh116_902_pinch_facets \
  --receipt "$PINCH_APPROACH_OUTER" --baseline "$SIDE_ROUND3_FIT" \
  --side "$SIDE" --pair-budget 24 --output "$FRESH_JSON"
```

Both complete scans exited 1 because the endpoint checks blocked. The outer receipt SHA-256 is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7` and terminal row SHA-256 is `ea6985622f9b45a280f005f56c5f48ba2cc2ab9482b62394632d31f7e0206888`. The facet screen source SHA-256 is `f4090363182a75df0751d10b7664a12699bb274b094571f4e99ed416da33a563`.

| Evidence | SHA-256 |
|---|---|
| `sync-pinch-round4-facet-left-v4.json` | `4221aa7939e175b8b1b45d0e0d55cd577624b5fdfd3f68beffdfde4b4183a9b5` |
| `sync-pinch-round4-facet-right-v4.json` | `9a894bf3bf2f1808b9135f023ac9bb078f737233e367eb69d337f9b7298fc503` |
| `sync-pinch-round4-facet-left-v4-argv.json` | `032cce2869b7a7c469348cc932d19cd293ab3a4f4a1e78b8886d00b36e4c5e6c` |
| `sync-pinch-round4-facet-right-v4-argv.json` | `9fd98d82c59e48c5dcbb2cfc46c7aaeee81fb05d3fd615714d18c8a3cd962144` |
| `sync-pinch-round4-facet-left-negative-witness.json` | `febf04954c3f4347bd38c1a1852132d8d0eb0dd6ee0c29f658a94bb45c79084e` |
| `sync-pinch-round4-facet-v4-cpu.log` | `85012857d094e51a5c1f050e8c8fb351658d0a350843bba764ad2e03abe749c9` |

The CPU regression suite passed **175 tests**. It includes the genuine right facet replay and an axial-spin test that verifies that both paired loci stay fixed.
