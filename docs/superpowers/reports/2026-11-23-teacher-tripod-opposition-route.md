# Tripod continuous left thumb-to-index route

The CPU verifier consumes the independently sealed, naturally closed Tripod run at SDK2168. The bound receipt uses the 9 mm by 450 mm supported wire, material node 35 and the measured transfer of the preceding thumb reaction. The SDK frame and post-step rod share the same completed tick. The archived clearance check itself ran before that rod exchange and uses SDK2167 nodes. The final pulse report confirms a separate clearance read at SDK2168. The verifier checks the outer terminal state, both row hashes, exact next-step wrench delivery, epoch identity and retained support loads.

An inexpensive projection previously gave the left thumb less than 1 µm of margin. Finite-segment refinement gave the thumb source mesh a lower bound of −18.705 µm at the start. It gave the left index source mesh a lower bound of −116.791 µm at the previously sampled endpoint. The original selected gap limit stays at −150 µm. Unselected hulls still require more than +250 µm. The opt-in `WireClearance.certify(reserve_m=...)` refines bounds until it establishes the original threshold plus a requested displacement reserve. The endpoint result retains the original threshold and raw geometric lower bound separately.

The route certifier proves linear world-translation chords and seed-relative intrinsic XYZ interpolation between 128 original 24D wrist commands. The finger phase holds the wrist and varies only left finger21 with the original nominal mimic chain. A per-link travel bound covers the intervals between geometric evaluations. Every left and right source hull and the ten archived cooked pads are checked against the finite rod, original table clearance and both supports. Each hand has seven measured pose corrections from the sealed row. Remaining links use nominal FK from the archived active positions and the original mimic mapping. A required 20 µm reserve is taken against the rod, separately from the bound on moving hull points.

| Candidate | Wrist windows | Pivot samples | Close samples | Terminal thumb / index SDF |
| --- | ---: | ---: | ---: | ---: |
| World rotvec (0, 0, −0.12) rad | 128 | 2000 | 19 | −0.050 / −19.978 µm |
| World rotvec (+0.08, 0, −0.08) rad | 128 | 2794 | 19 | −0.050 / −19.988 µm |

Both continuous hull proofs succeeded within the frozen geometric model. Both pad SDF minima remain at node 35. The separate terminal comparison with imposed shifts and bends passed 4 of 9 cases for the fixed pose and 9 of 9 for centerline-station/tangent tracking. These shifts and bends were CPU inputs. The interval proof assumes the archived rod coordinates subject to its specified reserve and the nominal link model. It does not bound future elastic deformation, control lag or load-dependent mimic error.

Two source-reader attempts failed before route evaluation. The first conflated the archived pre-exchange clearance clock with the same-tick pose and rod. The second assumed that all collider poses had been archived. Both logs are retained and hashed in readiness. The corrected command checked the real receipt and both route outputs with exit 0. The focused suite and final combined suite ran on Ubuntu. The final suite passed 112 tests.

To replay the world yaw case with its private receipt and a new output path:

```bash
PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -B -m scripts.certify_dh116_tripod_opposition \
  --run "$NATIVE_RECEIPT_DIR" \
  --receipt-sha256 760fe69c15cdf6e624a3660ca0e81d4343ed33198cf5c1e71a64dbb1b3c72031 \
  --world-rotvec 0 0 -.12 --index-increment 0.05068016052246094 \
  --reserve-m 0.000020 --pivot-windows 128 --output "$NEW_PROOF_PATH"
```

The next Isaac run must complete the initial thumb contact control window with zero target increments and retain the same exchange instance. It then needs live pre-command clearance and recorded consecutive thumb/index reactions, with the right hand held. These two left pads only cover the first opposition check for Tripod. Bilateral three-finger grip, fixed-wrist load retention, off-support lift, PPO and held-out wire trials remain unverified. No new native steps or model updates were executed in this package.
