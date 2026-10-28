# Command execution consistency repair

User scope: make grasp/lift commands execute coherently, especially continued force feedback when contact load drops. Real force recovery, not renamed telemetry, is the acceptance target. Continue the teacher-success → actor imitation → PPO route afterward.

## Evidence

Grid36 first-episode env22: held load targets 10/10 N; right load falls from 3.97 to 0.206 N, right mimic error approximately -0.007 rad leaves 0.0011 rad permitted closing lead. Existing feedback integrates to +0.08 with blocked flag false. No successful lift.

## Authorized corrections

1. Propagate actual task clearance to reposition instead of separate 2.1 mm constant; feed accepted safety goal back to arrival logic. Retain hard physical guards.
2. Preserve captured grasp but use bounded incremental internal force corrections from applied command; close integrator saturation must not disable lift correction. Report and prevent windup on projected constraints. Resolve pressure/coupling directional conflict with bounded neutral target, not mutually overriding demands.
3. Record request, governed target, submitted command, measured response separately; align execution metadata (0.20 rad/s, normal-load targets, effective Z floor). No physics-success claims from labels.
4. Focused regressions, review, then singleton real Isaac search/replay. Compare measured force recovery, actual height, contact geometry and physical faults. No BC/PPO until qualifying successful teacher/student validation.

## Acceptance limits

Mechanical bounds, per-link force guards, actual velocity guards, mimic hard limits and existing task success are unchanged. A physically blocked request is not guaranteed executable. Any remaining inability to recover real force stays explicitly unresolved rather than being relabelled fixed.

## User clarification: position pause is independent of force

Normal position-lead/coupling-budget pause must not disable contact pressure feedback. Contact effort remains permitted below the existing severe coupling threshold (0.012 rad), while ordinary position closure remains governed by 0.008 rad. Existing overload, severe-conflict, thumb-limit and actual hard physics guards remain. The total active motor torque budget stays 0.2 Nm. This is an intentional separation of position and force authority, not a guarantee of measured force or lift success.

## Verified so far

- Offline controller/world regressions: 226 passed before effort-permission addition; effort/world/training/servo subset 106 passed after intentional position/force separation.
- Real Isaac candidate `runs/success_teacher_20261028_163600_contact_effort8`: 8 first episodes, zero lift success. It demonstrated force feedback during a held position target: env2 left thumb command held at 0.420545383 rad during 3.464583–3.920833 s, explicit effort rose 0.089036→0.103894 Nm, measured load 0.666→2.947 N, actual angle moved only 0.000619 rad. Force was intermittent, not a sustained 3 N.
- Actual FF+implicit-drive budget remained at most 0.2 Nm. No motor-budget increase. Right contact loss repeatedly cleared FF; right had one above-axis contact rather than stable opposing support.
- The .008-rad FF comparison (`...contact_effort8_budget8`) was not adopted: user explicitly requires force generation during ordinary position pause. Final FF cutoff is severe coupling .012 rad; the .008 position governor remains.
- Geometry12 candidate completed 12 first episodes, zero lift success; reposition chatter/timeouts persist.
- Further verified root causes: release/recontact lacked hysteresis, one hand could reclose before partner arrived, and a hand's motion timeout counted time blocked by the other's release. These state-machine fixes are under implementation.

No qualified teacher dataset, BC fit, or student PPO has been launched. Success remains the existing real height/stability predicate; no failure trajectories are relabelled as successes.
