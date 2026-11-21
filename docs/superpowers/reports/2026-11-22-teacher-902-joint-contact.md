# Small joint alignment restores a complete CPU Tripod closing path

Base: f0df5ae63e2b8f679e3336b1bc2dd785c629c53b, branch 11-17.

The previous middle-only closure failed because the continuous source hull and node-based cooked SDF constrain different geometry. The independent search changes only a bounded right wrist alignment and thumb/index/middle closing joints near the original 902 endpoint. Original source SHA 871e37907badbd4a7b534ac4e458fe42a43e80dbf15216e87e1746c7873827b0 is independently pinned. Thumb spread, unused fingers, left hand, assets, cameras, material, joint/action contracts and all original safety thresholds are retained.

## Actual implementation
- scripts/search_dh116_902_joint_contact.py: decode_candidate bounds translation norm <=0.5mm, rotation norm <=0.015rad, three grasp-joint deltas <=0.02rad; candidate_passed requires the original 22-shape continuous guard plus three negative SDF gaps, opposition and right-side node identity. SDF and source supporting-plane costs guide a deterministic finite optimization only.
- scripts/certify_dh116_902_joint_contact_path.py: select_target binds original request, search SHA, receipt SHA and selected ordinal, then independently recomputes bounded decode(x) and rejects changed target fields. certify reloads actual rod evidence and original side-specific USD, checks all 239 initial/settled frames, compiles 64 original-rate windows with exact float32 state evolution, and measures every one of 512 substeps with the original geometry checker.
- configs/dh116_tasks/flexible_902_tripod_joint_contact_candidate_v1.json is a separate opt-in candidate. It grants neither native execution nor Teacher admission.

## Saved-data measurements
Search: 8 starts, 2691 actual residual calls, 4.028s, no budget exhaustion. Five endpoint candidates pass. Two full paths were compared, candidate 0 (smallest motion) and candidate 4 (wider guard reserve). Both pass; candidate 4 is retained.

Official fresh proof: initial_passed=true, closure_geometry_passed=true, endpoint_contact_candidate_passed=true.
Full-path minima: noncontact 2.545798mm; support 1.008004mm; table 46.545430mm; contact source/cooked envelope -0.139654mm (unchanged floor -0.150mm, reserve 10.346um).
Terminal three gaps: -0.0200008 / -0.0200180 / -0.0200006mm, node IDs [3,2,4], opposed normals pass the original threshold.
Action list SHA 1b3b6f2ff0b2ecd9642a10796ca16a1bbcf92e1b24258c82677d853107f82862.
All are CPU measurements against archived actual rod geometry; they are not new native contact forces, grasp or Lift success.

Test-first collection failures for missing new modules were retained. Five new behavior tests protect coordinate/side/finger bounds, source-guard rejection despite negative SDF and source/receipt/search/target identity binding. Final related CPU suite: 228 passed, 0 failed, 69.71s. The first private full-path prototype rejected an unresolved /home directory alias; resolving the evidence directory fixed it, with the first error log retained. Production CLI then completed a fresh full proof with exit 0.

Antigravity R1-R3 independently read exact source versions and found no blocking implementation defect. It could not read configs through the source-only MCP allowlist; Codex checked the actual file bytes and recorded SHA. Review correctly identified the smallest-motion candidate's narrow 3.25um source reserve; selected candidate 4 improves it to 10.35um. Support lower-bound reserve is only 8um, so it must not be advertised as robust live tracking margin.

## Required next native gates
The new wrist initial pose needs an independent bounded Reset-only receipt and actual all-frame clearance audit before any closure. Then regenerate the closure proof against the new actual rod, bind the exact plan/source SHA and run one serial guarded closure with real pair/wrench evidence. Intermediate substep material-node identity and REFERENCE-vs-frozen-plan SHA need explicit native-plan checks. Static negative SDF is not reaction-force proof. No old consumed fingerprint is reused.

No new Isaac/solver/Teacher/PPO steps were run in this package. Both fixed RGB configurations and 528 protected files / three original checkpoints are checked before release. New output models or raw native logs are not uploaded. Pinch, Tripod and Power retain independent final validation; this is a right-side Tripod path candidate only.

Evidence remains private under ref902/joint-contact-*.json/.log. Runtime replay and strict original force/position safeguards remain the next qualification boundary.

Registry regression: 60 passed, 0 failed.
