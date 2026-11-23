# Loaded pair continuity for supported wire

The audit compares fixed solver node and named mesh pairs across consecutive physical clock records. Each source row is checksum-verified against its independent natural-exit receipt. The reader first reconciles the complete 12-substep force packet with its recorded solver totals. It then counts uninterrupted pair samples whose individual normal force exceeds 0.05 N. This count describes an individual pair. The existing grasp gate still uses the total pad and hand loads and the validated bilateral wrench.

| Archived phase | SDK ticks | Hand pairs | Loaded hand pairs | Longest relevant uninterrupted pair |
| --- | ---: | ---: | ---: | --- |
| Pinch approach | 424–432 | 0 | 0 | Absent |
| Tripod seating | 1328–1416 | 5,745 | 5,586 | Left index, node 35, 19 solver substeps or 3.30 ms. Left thumb and middle each lasted 1,068 substeps or 185.42 ms. |
| Power palm witness | 6441–6504 | 1,515 | 0 | No individual pair exceeded 0.05 N. |

The Tripod index is the intermittent left finger in this interval. A longer wrist hold at the same scripted target would still need that index to retain contact. Right thumb and index each retained a fixed loaded pair for all 1,068 substeps, while the right middle retained one for 672 substeps. Whole-grasp success still requires qualified opposing forces and support-free Lift.

The frozen CUDA source writes the geometric node-sphere point as the final node position minus contact radius along the solver's outward normal in `sim/contact.cuh` (SHA-256 `8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb`). Pair capture has solver substep, node, mesh, point, normal, force and torque. Reconstructing the last substep sphere center from these fields agrees with SDK endpoint positions to an observed maximum of 0.035 µm on Tripod and 0.040 µm on Power. This confirms the interpretation on the archived samples. Continuous rod material rotation, a fixed pad anchor and the solver's relative tangential displacement are still absent from this capture. In the same CUDA source, `surfaceContact` computes relative tangential motion, then applies a saturated friction law. Force alone does not determine that motion. The diagnostic keeps native material slip unresolved.

## Reproduce

Use the pinned CPU environment from `docs/contracts/dh116_wire_joint_pregrasp_v1.json`.

```bash
python -B -m scripts.audit_dh116_loaded_pair_continuity \
  --receipt "$SEALED_OUTER" --receipt-sha256 "$OUTER_SHA" \
  --grasp "$GRASP" --first "$FIRST" --last "$LAST" --output "$FRESH_REPORT"
```

The command set is `pair-continuity-release-r4-ko7m8bvt/argv.json` in the private evidence directory. Result digests are `d0bf1028dc16a1f2574d54645034c7a5ceca1b5e9a84e3b05af3dc4b4d066a4d` (Pinch), `f6665b1c9c1e0b4369fbe1c4710bb0f66e241038e331724353bb4b502bffa01c` (Tripod), and `eb227ceec786236764aa1e9ad958b3c768b37f83ffedf960a69c1c777d687f2b` (Power). The 113 CPU checks passed, including pair discontinuity and repeated tick rejection. Test log SHA-256 is `9e22e6375d630ea6a7687c0642541ea50442ef6a635d26a465064b16f3223364`. This audit acquired zero new physical states.
