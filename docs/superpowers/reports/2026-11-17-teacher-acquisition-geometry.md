# Actual-source Pinch/Tripod acquisition candidates — geometry only

## Actual result
No Isaac/CUDA/PPO was run. A bounded per-hand FK/SDF search used the existing native cohort's measured first-frame rod and base transforms and the original locked pad SDFs. Finger axes, passive coupling, joint limits and each hand's quaternion are retained. Root translation is bounded to +/-25mm per axis, active q changes to +/-0.35rad within the original limits. These are separate reset proposals, not executed motions or expanded recovery budgets.

| Style/hand | Required-pad gap before (mm) | Candidate gap after (mm) |
|---|---|---|
| Pinch left thumb/index | 8.718545 / 12.087942 | 0.400005 / 0.399999 |
| Pinch right thumb/index | 5.659561 / 6.862424 | 0.399950 / 0.400958 |
| Tripod left thumb/index/middle | 9.410377 / 2.687993 / 6.199727 | 3.554831 / 4.623876 / -0.688513 |
| Tripod right thumb/index/middle | 1.626240 / 9.974446 / 3.237929 | 0.399619 / 0.399834 / 0.399854 |

**Neither S0 is frozen or admitted.** Pinch meets only the <=1mm geometric gap prerequisite; this does not prove executable bounded acquisition, force closure or safe final reset. Tripod left has both remaining gaps and sampled penetration and is rejected.

Minimum table clearance source17/cooked5 pads: Pinch L55.650/56.704mm, R59.455/61.157mm; Tripod L43.749/44.099mm, R51.645/51.688mm. Table and end-support hull checks did not find intersections, but some nonpad-to-rod comparisons remain unknown. Missing nonpad cooked shapes, inter-hand/self clearance, fresh settling and measured reaction-force contact remain blocking.

## Implementation
- tabletop_teacher_acquisition_geometry samples only the 41 actual material-node positions in the original finite SDF. Out-of-grid remains unknown; no curve interpolation or predicted force is introduced. Trilinear output remains float64.
- Input validation rejects nonfinite radius/grid/nodes, invalid spacing/origin and non-rigid transforms.
- Source convex checks reuse the existing original geometry module. A failed continuous convex QP remains recorded. A supporting-plane lower bound may prove clear only when an entire segment is strictly separated; otherwise it stays unknown.
- Search copies FK transforms before translation; it cannot accumulate changes into HandPointFK's shared root matrix.
- The CLI binds actual 5mm radius and original task SHA 5843f88dc38aac8ed55706a9cfa03ff09302089be46fd1627b96e1c1ab7a1ca7.

Constructor seeds.nodes are **exactly the original straight supported_seed_plan nodes**, retaining original 0.5m rest length and construction masses. The observed curved reference has length0.499860479m/0.500013249m and is stored separately; using it as a new constructor would change the physical rest state. No such replacement is made.

## Evidence and verification
Authoritative proposal-v6.json SHA ac21168383b091f6f233b10057aa075141f53e00eb0eeaf1f2a2608ebdb30282, outside Git at /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-acquisition-20261117/.
Earlier invalid proposals remain preserved. Current boundary validation changed the implementation SHA after v6 search; boundary-verification.json explicitly distinguishes those versions. Re-evaluation with the new boundary checks matches all old counts/unknown states and gaps to <=2.984e-16m, without repeating search.

TDD27 targeted tests passed; full relevant regression **304 passed,0 failed,0 skipped**. Command, log hash and all source hashes are in readiness. The primary gap table was independently read back from the preserved proposal. All528 protected hashes and old models remain unchanged.

## Next gate
Acquire actual nonpad cooked geometry and verify the corrected native initializer before freezing any contact proposal. Then perform bounded force-driven acquisition with original per-pad episode budgets. Full Teacher success still requires bilateral opposed force, support-cleared >=30mm Lift, >=5s Hold and <=0.5mm fixed-anchor Slip; geometry is never substituted for reaction force.
