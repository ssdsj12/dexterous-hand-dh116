# Whole original rod-group convex-envelope certificate

Base HEAD 1b9655b32a9af31467050d81515b0157a6d1e280 / branch11-17.

## Runtime change
The opt-in rod_envelope_clearance already certified each entire 3-node x T-frame convex hull, but only proposed directions from two final-frame physical segments. On a bent group, both directions can miss a valid separating plane. Keep the original fast direction stage, both segment stages, every node/frame/vertex, radius5mm, selected-pad limit0.15mm, and total40-QP budget.

New _group_envelope_bounds minimizes distance between the complete source hull and the complete group convex hull using bounded SLSQP (200 iterations). It repairs and checks a strictly feasible source witness and nonnegative normalized group weights, computes a non-expansive full-set support lower bound and feasible upper bound, and requires their gap<=1nm. Only the support LOWER bound may pass. Fallback consumes the same QP budget; unknown/failed/below-floor results fail closed. Convex witness weights are geometric certificates, not new material nodes, reconstructed hidden wire, or contact measurements. No physical/controller/camera/action/reward parameter changes.

The standalone full-path certificate now explicitly includes this helper source SHA; native frozen dependencies already contained it. Old receipts/proofs remain immutable.

## TDD and saved actual frame
New analytic rotated bent-group fixture has exact hull-to-group distance4.853mm: old code rejects its -0.147mm capsule gap, new code certifies it. Separate tests cover shared budget2, all27 history nodes, a group collision at4.840mm despite safe adjacent segments, and unknown solver failure. Initial behavior red:2 failures. Broad related227 passed/0 failed/41.76s; final strengthened narrow29 passed/0 failed/2.80s.

SDK765 raw original arithmetic was independently reproduced with tabletop_batch_geometry.rotated_points (no quaternion normalization): original fast and two segment bounds match the archived native guard, including -0.150210972697mm. Fallback uses one additional QP and gives full lower/upper -0.146765289688/-0.146765182934mm (width1.06754e-10m), inside the original -0.150mm floor. The earlier exploratory SciPy normalized-quaternion reconstruction differs by about3nm and remains separate; it is not the exact native replay.

A fresh 80-window/640-substep CPU command certificate passes initial, whole path and geometric endpoint candidate checks and explicitly binds the new helper hash. It is not new native or force success.

## Independent missing-index cause
Recorded SDK764 geometry queried with the unchanged source SDF:
| Source pose / rod | Index gap |
|---|---:|
| ideal endpoint / settled SDK240 rod | -20um |
| ideal endpoint / actual SDK764 rod | +130.05um |
| measured SDK764 link pose / actual SDK764 rod | +183.68um |

Maximum rod-node displacement since SDK240 is4.598mm; it includes natural motion and contact response, and is not assigned wholly to contact by this audit. Actual joint tracking contributes to the gap in addition to rod motion. These are CPU queries on recorded real frames, not new native measurements or supervised-success labels.

With measured other joints/base and all9 real rod frames756..764, adding0.002rad to the index closes its SDF gap to -6.94um but fails the unchanged complete source-envelope floor at -240.78um. Therefore single-index extra closure is not a safe fix. No such action was issued.

## Review and current qualification
Antigravity 902-whole-envelope R1-R2 via codex-dh116-ssh-reviewer read the same helper source SHA8f511fd6... and found no substantive defect. Codex independently checked the derivation and exact native transform. Its advice to prioritize actual66+rod Tripod PPO smoke over another scripted-feedback controller is a proposed next step: current provider/source code still needs actual style-specific audit, not an assumption that a Pinch factory already supports Tripod. Final test strengthening and standalone helper dependency were added after its main patch read.

No new native run, policy query or PPO update. The previous native SDK765 guarded failure remains failed, index load remains0, three-pad loaded rows0, supports remain loaded. No Lift/Hold, Teacher, BC/DAgger or Student admission. Both fixed RGB configurations and all528 protected files/three original models remain unchanged.

## Reproducibility
CPU: run the16 related files listed by whole-envelope-regression.log under the original CPU environment; narrow python -B -m pytest -q tests/test_dh116_continuous_envelope_refinement.py tests/test_dh116_902_joint_contact_path.py.
Actual saved-frame replay: private ref902/replay-whole-envelope-exact.py; raw guard, both CPU replay versions, gap decomposition and invalid compensation screen remain outside Git. Fresh proof: ref902/execution-closure/cpu-proof-whole-envelope.json.
No raw models/assets/logs/data were uploaded.

Registry regression:60 passed,0 failed,18.50s. Both fixed640x480 camera JSON SHA values were independently checked unchanged before publication.
