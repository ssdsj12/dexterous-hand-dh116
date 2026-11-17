# Teacher reaction-force closure repair — CPU package

## Outcome and scope
The new opt-in diagnostic execution path blocks wrist ascent without measured bilateral opposed normal loads. Closure is not inferred from finger angles, appearance, CAD gaps or policy commands. This package ran **zero Isaac experiments and zero PPO updates**. It does not establish a successful grasp or qualify the existing three candidate models.

Base: 56e7f16303e13e7e49eae75492e3ffbaef78d53b on 11-17. Current original Teacher source remains byte-identical (f76d42c53bd7adb904f5d7e28741f83657fe688db7b5f64d66a436e60cecf9d6). Original default paths are unchanged.

## Actual prior failure evidence (not a new native collection)
Cohort: /mnt/mydisk/xk/dh116/runs/teacher_supported_video_20261117T052535Z; native_frames.jsonl SHA256 29616b6d732ede3d5a8ae2cb6159d0be97550de214b741d9e9ed3c8aa6a8bb5b.

| Mode | Left/right palm Z change | Rod outcome | Interpretation |
|---|---|---|---|
| Pinch | +79.976 / +79.937 mm | max minimum-height change 6.450 mm; no support-cleared frames | Hands ascended without sustained bilateral grasp |
| Tripod | +67.393 / +67.469 mm | max minimum-height change 0.0107 mm; no support-cleared frames | Hands left rod behind |
| Power candidate | -9.033 / -9.083 mm final | max support-cleared net lift 21.009 mm | No palm/proximal full-wrap qualification or full success |

The old record saves only the last physical substep of each control, so sampled force episodes do not prove continuous 480Hz holding.

Locked-SDF checks against actual SDK poses find initial Pinch thumb/index gaps: left **8.719 / 12.088 mm**, right **5.660 / 6.862 mm**. Tripod thumb/index/middle: left **9.410 / 2.688 / 6.200 mm**, right **1.626 / 9.974 / 3.238 mm**. These are geometric diagnostics, **not contact measurements**. Existing FK matches all 17 visible link positions at the measured seed to <=0.000480 mm and rotation <=0.00000101 rad; independent per-hand mappings are retained.

## Actual code changes
- tabletop_teacher_force_closure.reduce_force_closure validates the original atomic pair packet with existing validate_pair_capture, reconciles independent node force and mesh wrench totals, and checks each solver substep independently. Only normal compression counts; tangential load, table/support load, one-sided contact and transient averages cannot qualify.
- Required normal-loaded pads: Pinch thumb/index; Tripod thumb/index/middle. Every pad >0.05 N, required-pad sum per hand >0.3 N, normal cosine <=-0.25. Tripod requires every pad on opposing faces. These diagnostic thresholds are not a full wrench-space force-closure proof or final success criteria.
- Power always reports power_palm_proximal_source_missing under the existing 13-mesh layout. Five tips alone cannot certify palm enclosure.
- compile_force_runtime freezes the old source SHA; limits execution to N=1, evaluation-only and zero PPO; enables complete raw pair capture; hooks **every** physical substep. Both wrists remain fixed in Close/PreliftHold while original finger action integration and limits stay intact. The source candidate disabled PhysX table collision; only this independent task re-enables it. Old source, USD and material values are unchanged.
- ForceContactHook reuses PhaseOwnerV2 endpoint mathematics. Actual-pose hold targets are latched, not moved along with the wrist command. Contact loss clears lift permission before the next physical action and discards command lead. Invalid/overflowing contact data aborts and saves a terminal failure.
- Each row separates original policy action, clipped source action and issued projected action; saves actual active/passive q/qd, limits, targets, measured link poses, raw pair packet, solver/SDK times and run/environment/episode/policy/physics identity. Failed attempts never claim the previous action was issued.
- ForceDiagnosticLedger uses exclusive output creation, flush/fsync for each JSONL row, no overwrite. Nonfinite values use explicit non-measurement tags rather than zero.
- Candidate prelift gate: 0.25 s plus actual node/palm finite-difference speed <=2/1 mm/s. This is not the required five-second final Hold. Final contact slip is NA and final success always false. This projected-action diagnostic cannot perform PPO optimization or enter an expert pool.

## Tests and review
TDD evidence includes missing APIs, lowercase integration, invalid geometry, terminal source/JSONL, raw action, pre-action clock/identity and stale-issued failures. Logs remain outside Git under /home/xk/.local/share/dh116-autonomous-3c80166/evidence/force-*20261117*.

**269 CPU tests passed, 0 failed, 0 skipped** across new modules, phase math, packet reconciliation and supports/recording/video/control regressions. A test executes the extracted source's eight-substep loop on CPU test doubles, checking all eight projections and SDK-target calls. This is not Isaac simulation. Spec and independent quality review found no remaining blocker.

Exact command/environment/log hashes: docs/contracts/dh116_teacher_force_closure_readiness_v1.json. Full regression log: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/force-contact-full-regression-20261117.log.

## Preservation and open problems
All **528 protected source hashes match**. Three candidate model hashes are in readiness. Two previously inventoried /tmp/dh116-wp3-teacher128-v02 checkpoint paths remain missing; no replacements are synthesized. Prior native data is read only. No weights, raw data or run logs are uploaded.

Remaining work:
1. Resolve actual initial acquisition/reachability; a gate does not close the 5.7–12.1 mm gap.
2. Native bounded test of measured force, clocks, joint safety, Reset and shutdown, using a new frozen worker/budget. Prior budgets are not reused.
3. Actual Power palm/proximal collision/force sources and five-finger wrap.
4. Qualified fixed-anchor slip and support-cleared full Hold.
5. Independent new Teacher contract/training after the physics chain passes; current candidate reward returns are not formal R0 qualification.

Success remains **30 mm / 5 s / 0.5 mm**, bilateral opposed contact, no external support/upward force/attachment. RGB, Teacher/MPC and Student qualifications stay separate.
