# 902 tracking reserve: distinguish coarse bounds from real clearance

Baseline: 2841d972504174f215e26d0a0e184d00a6023c67, branch 11-17. This package adds a read-only geometry refinement and executes it on actual archived command witnesses. It does not change native guards, action contracts, terminal 902 grip, physics, or camera configuration.

## Implemented
scripts/audit_dh116_902_tracking_reserve.py:
- rod_reserve_certificate retains every one of the nine 41-node snapshots. Finite-segment witnesses only provide feasible upper bounds and directions; every improved lower bound uses the full original nine-frame three-node group.
- box_reserve_certificate combines AABB and SAT lower bounds with a feasible vertex/box upper bound. SAT is never described as exact distance.
- _summary distinguishes certified reserve, proved insufficient reserve and unknown. Original contact -0.15 mm and support +1 mm guards remain unchanged. Requested extra reserve is a diagnostic target, not a new success threshold.
- main checks witness SHA and bounded distinct sample ordinals, requires all 22 shapes and both supports, and records the implementation SHA of this script and three distance helpers.
- No QP failure or budget exhaustion silently certifies an unresolved interval. No code path grants native or Teacher admission.

## Actual archive replay
The same 1032-command-step ideal staged path was reconstructed, with exact sample equality to the prior screen. Seven geometry witnesses were exported (1, 72, 477, 600, 721, 853, 1032). The new CLI executed successfully using the server CPU environment. Witness SHA 217b030e3e9e909607d3c735b3955bd1792b735523c8b2fc4119aac901e42ebf.
Audit JSON SHA 3300b8c3ac0f44c992edc626064ccc7cc70b653a1e8c9587183c4cbe82f69d14.

| Ideal substep | Rod signed-gap lower / upper (mm) | Support lower (mm) |
|---|---:|---:|
|1|0.347759 / 0.982545|1.383913|
|72|0.555664 / 1.312579|1.383913|
|477|0.191542 / 0.397961|3.112721|
|600|−0.081740 / −0.081740|3.632167|
|721|−0.081740 / −0.081740|3.811040|
|853|−0.131727 / −0.124301|3.760575|
|1032|−0.116606 / −0.115136|3.743624|

The prior 0.122/0.245 micrometre figures were reserves of coarse lower certificates, not measured exact clearances. In particular, the support lower bound at substep853 improves from 1.000245 to 3.760575 mm. However, the contact reserve above -0.15 mm is genuinely narrow at selected states: about68.26 micrometres at600, bounded18.27–25.70 at853, and33.39–34.86 at1032. It is not sufficient to assume tolerance to the prior measured approximately1.14 mm wrist/link tracking discrepancy. Initial support clearance is only about1.384 mm, so later support improvement does not qualify the whole path.

The seven witnesses do not certify all1032 steps or native tracking. Every output explicitly keeps full_path_certified, robust_route_qualified, native_qualification and teacher_admission false.

Three additional static waypoint candidates open the right middle finger an extra0.02/0.04/0.06 rad while moving the wrist. Each endpoint satisfies the existing geometric guards; the0.02 candidate middle SDF gap becomes+2.015147 mm. These are endpoint checks only, not executable native recipes or contact evidence. The final902 target is retained for subsequent closure.

## Review and validation
Antigravity used repo_info/read_source on the same Ubuntu11-17 worktree. R1–R3 reviewed bounded refinement and actual source SHA. Codex corrected the assumptions that SAT gives exact distance, that only segment witnesses can supply upper bounds, and that a later failed optimizer must invalidate an already proved insufficient reserve. R3 found no blocker in the mathematical implementation. Its source SHA was37a44d73…; a subsequent metadata-only addition binds three helper source SHAs, with a new CLI regression. The final full SHAs are in readiness.

Failure-first evidence includes four missing-module failures, a missing-CLI failure and missing-helper-identity failure. Final related regression:218 passed,0 failed,41.48 s (16 new cases). Analytic tests cover negative-gap certificates, genuinely insufficient reserves, prior-frame collisions, translated geometry, unresolved optimization, invalid inputs and SHA/scope binding. Raw logs and arrays remain private.

Run command:
PYTHONPATH=<existing CPU environment> /mnt/mydisk/xk/envs/dh116/bin/python -B scripts/audit_dh116_902_tracking_reserve.py --witnesses <private>/ref902/command-v3-staged-path-witnesses.json --sha256 217b030e3e9e909607d3c735b3955bd1792b735523c8b2fc4119aac901e42ebf --output <new private output>

## Native failure and next implementation
The earlier Tripod attempt remains failed atSDK476, natural outer exit1, zero hand wrench, no successful grasp. It is not rerun. The source guard failure and cooked positive gap remain separately recorded. No new Isaac, solver, Teacher query or PPO update occurred here.

Next implementation: separately compile a staged middle-opening/wrist-transit/fixed-wrist-closure candidate, certify every intermediate command, require same-clock measured wrist pose and velocity convergence, and use real opposed contact plus delivered reaction to terminate closure. Reaching a requested joint pose alone never proves grip. Pinch/Tripod/Power and all Lift/Hold standards remain separately unqualified.

528 protected files, three historical checkpoints and both fixed RGB camera SHA values are checked unchanged before publication. Windows imports the exact Ubuntu commit; PR#2 remains Draft. No private raw evidence or assets are uploaded.

Registry regression:60 passed,0 failed; combined CPU checks278. Registry log SHA 4e2f2412d80b5bb79b709793ebf7620b5fb3d36f7762d6591f33b02fa0857fdc.
