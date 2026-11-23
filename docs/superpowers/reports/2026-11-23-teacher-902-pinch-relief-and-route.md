# 902 Pinch contact-locus relief and paired approach

## Registration

The authored scene `wire_radius_m`, sealed native Reset record, F4 SDF bake record, and CPU capsule radius all agree at **5 mm**. The `1 mm` SDF grids retain metre units, zero thickness and rounding, and the F4 link-origin basis. The F4 mesh-to-body transforms differ from identity by at most (2.23×10^{-16}). The largest F4 SDK-to-USD world matrix entry difference is (1.41×10^{-7}). At SDK 432, the uncorrected source FK reproduced the measured distal origins within **61 nm** across the four pads. This compares actual side-specific measured joints against the original USD link frames.

The four native archived mesh-binding rows agree field-for-field with the capture, cooked, and registered SDF hashes read from disk. The recorded binding provenance remains `caller_supplied_unverified`. The audit validates byte identity and frame agreement. Native force/contact qualification remains separate.

| Pad | Locked F4 capture SHA-256 | Registered SDF SHA-256 |
|---|---|---|
| Left thumb | `ff842335f14fe670736dd36e72494af085a2b6f019eea91914ad9fa292113dac` | `bfa26827982cd871f084c2d6cf3f276831f06b9d66732dd177000e1a46071af2` |
| Left index | `2e5d4ab0b55188c2499b7c835ec79c37bf6db4b9b0c32eb1981ba8e4ef124c81` | `cf96dbb5cfb36eeacab34c48bacae18eb9030d3fb916a3358da3b521568d38c3` |
| Right thumb | `62615e9350789c8f4ccd319c2c78cbd584830a56e0a2b9ed39cc61d57b8c9c29` | `a001ba259df9b9884f065d3f631b816ab6be3e35475d002286513876762807c0` |
| Right index | `ee95dde42a9cf6cadfd63a551856a4d3f0f20b1abe826b2cfd86f6ab0d978401` | `56f13247cd2866daa4df159a5701b33f566a207b245ad1ca3de7cdca05a9f17a` |

## Alternate SDF loci

The fitter intersects trilinear lattice edges at the original radius, ranks locations by source and cooked capsule guard violation, then uses exact hull-point checks to retain distinct regions at least 2.5 mm from the earlier witnessed SDF locus. It adjusts thumb abduction/flexion and index bend to co-locate two chosen loci, holds the other fingers, and turns the opposed SDF normal transverse to the measured rod tangent. The whole nine-frame rod envelope, source proximal hulls, both supports, and the table retain their original acceptance checks.

The prior left exclusive station-36 attempt still fails the source thumb hull with a certified finite-segment upper bound of **−0.381 mm**. On the right, translating along the old thumb SDF normal until first SDF contact makes the source thumb hull fail with a certified upper bound of **−0.538 mm**. This reproduces the previously diagnosed collision region in the original body frame.

| Side | Pair fits / co-located fits | Full endpoint checks / qualified | Primary station | Thumb / index SDF gap | New thumb locus departure | Minimum non-pad rod gap |
|---|---:|---:|---:|---:|---:|---:|
| Left | 32 / 30 | 24 / 7 | 38 | −0.02834 / −0.02833 mm | 3.063 mm | 1.426 mm |
| Right | 32 / 31 | 24 / 3 | 4 | −0.02865 / −0.02877 mm | 4.346 mm | 1.457 mm |

The qualified primary poses use −30° left and +15° right twist about the opposed pad axis. Left and right normals have dot products of −0.993 and −0.889. All 22 hulls per hand retain the existing bounds, including the right selected-pad full-envelope refinement. The two primary endpoints were combined and checked on each shared SDK 424–432 material frame. The four pads retained their exclusive side-specific contact nodes throughout. The complete hands have a conservative **289.495 mm** signed separation along world Y. Seven left and three right independent qualifying endpoints give 21 possible Cartesian pairs. One simultaneous pair was checked.

## Original 24D issued-target replay

The route starts from the archived backend finger targets, projected roots, palms and angles, SDK-measured roots, and physical mount anchors. It applies the existing 24D action kernels and float32 submitted finger and carriage targets every 480 Hz SDK step. After the archived first input, the CPU assumes ideal tracking of each submitted target. Geometry is checked against the static nine-frame archive at each sampled SDK command boundary. Swept-volume and native motion verification remain pending.

| Approach | Checked SDK substeps | Result |
|---|---:|---|
| Direct from the projected targets | 210 | Left source index `finger23_link` reached a finite-segment upper bound of −0.152013 mm, beyond the original −0.150 mm selected-pad guard. |
| Initial +X retraction, 8 mm | 53 | Right source index full-envelope interval remained unresolved, with lower −0.155105 mm and feasible segment upper −0.147442 mm. |
| Initial +X retraction, 16 mm | 52 | Right source index full-envelope interval remained unresolved, with lower −0.156198 mm and feasible segment upper −0.148534 mm. |
| Clearance-guided withdrawal, 8 and 16 mm | 0 each | Left departure +Z passed the sampled withdrawal check. No left terminal fixed-posture withdrawal passed all five samples for any of the 26 directions at either distance, so no staged commands were submitted to the CPU replay. |

The direct run kept at least 313.340 mm of cross-hand separation up to its first guard. The next approach needs a different joint and wrist orientation transition to clear the left source index hull while preserving the new endpoint contact. Both endpoints remain eligible for continued CPU route screening, with no qualified route yet. No Isaac run, native force hold, lift, Teacher query, or PPO update was executed.

## Reproduction and evidence

Run in the existing standalone CPU USD environment using the worktree-specific paths recorded in the argv receipts.

```bash
OPENBLAS_NUM_THREADS=1 python -B -m scripts.screen_dh116_902_pinch_relief \
  --receipt "$PINCH_APPROACH_OUTER" --facet-archive "$SIDE_FACET_RESULT" \
  --side "$SIDE" --pair-budget 24 --output "$FRESH_SIDE_JSON"
OPENBLAS_NUM_THREADS=1 python -B -m scripts.certify_dh116_902_pinch_relief_route \
  --receipt "$PINCH_APPROACH_OUTER" --left "$LEFT_RELIEF_JSON" \
  --right "$RIGHT_RELIEF_JSON" --output "$FRESH_ROUTE_JSON"
```

Both locus commands exited 0 with independent qualifying endpoints. The joint route command exited 1 with a verified paired endpoint and completed blocked route evidence. The sealed native outer receipt is `ac6345ae0b377b70a098155786571e14cd91855af3a252471529ed7218d360f7`. The relief code SHA-256 is `90aacd07ec0a581fd2b124d5f4287a07bb3bbe387d77f8d638b546b8f17f0650`, and the final route code SHA-256 is `eac1c77ac18f27ec26990ddec9b80b130c3011a7e20da8e7b194b02c0e3d93ab`.

| Evidence | SHA-256 |
|---|---|
| `sync-pinch-round5-relief-left-v3.json` | `c606f84d04beadb939039dc39148e14cf15b050ac3d2795222aad43682cac89c` |
| `sync-pinch-round5-relief-right-v3.json` | `553e446fe5c67da938f05be4d5a9dc87ef13dd548d6bee1be994322541da2f38` |
| `sync-pinch-round5-relief-left-v3-argv.json` | `b640e1ec4adc0bbd09010dbd3bf35092ca0d7e30d536467e84d9e54640d89585` |
| `sync-pinch-round5-relief-right-v3-argv.json` | `be6fd71c11473aa803605874a73147140d28f36207e23217ec59c9878fc68a91` |
| `sync-pinch-round5-relief-bilateral-endpoint-v1.json` | `a3621728c16bbc9a5599b4665f29a556b6241ff84bb8594f0b8594b479f7ba6f` |
| `sync-pinch-round5-relief-route-v1.json` | `2e6b77ef386a8bdfdcba1e078c1aa47e2021a5816a726662b3e74aacfcca3a52` |
| `sync-pinch-round5-relief-route-v2.json` | `593a9fc121306f04313c9e21ad5d440d5d4ca851896e62260b3c8cea7bff3d76` |
| `sync-pinch-round5-relief-route-blocker-proof-v1.json` | `27ffd0afdbc0f8fd1e2ed89111d014edbc940c639de79dde9313d1d8fb71d7fc` |
| `sync-pinch-round5-relief-route-v2-argv.json` | `31eb4df3c85c87568f3b1c3059994f8de567823a42ff2f13c89eb2537c299f81` |
| `sync-pinch-round5-relief-route-v2-cpu.log` | `abcf10c09ab1a6a9e8a8a86df738259572587263d6c02cb549a89e80635d0b4a` |

The CPU regression suite passed **183 tests**, including the registered real capture, the simultaneous nine-frame pair, and a withdrawal with an unsafe intermediate sample.
