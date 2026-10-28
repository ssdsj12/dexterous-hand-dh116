# Paired thumb loading diagnostic

User authorized the next step after the v136 rejection: isolate unloaded motion from loaded thumb motion before changing rewards. Use the existing fixed-plate fixture; no PPO update or training parameter deployment.

Four isolated hands: left/right thumb × plate present/plate moved 0.1 m outward. Identical targets within each pair: 0.5 s settle, +0.15 rad at 0.05 rad/s, 2 s hold, return, 0.5 s release. Only selected thumb pad collision enabled. 480 Hz, PGS, 64/16 iterations, active speed 0.3 rad/s, passive speed 2.97 rad/s, mimic frequency 400, stiffness 20, damping 0.2, effort 0.2 N·m. Fixed frictionless plate isolates normal loading; compliant contact K=10000 N/m, C=30 Ns/m matches training pad estimates. These fixture contacts differ from the grasp geometry and friction, so success does not imply successful lift.

Record signed coupling error, commanded/actual joints, force/contact location, all joint velocities, and explicitly estimated PD torque. Stop the entire trial on the first physical fault; save the failing frame. No-load is invalid if contact exceeds 0.05 N. Full paired result is valid only when all four finish without faults; a truncated trial remains useful for locating the first violation. Repeat twice only if first trial is physically valid.

- [x] Add and fail tests for paired configuration (same posture and gains, only plate distance changes).
- [x] Add opt-in paired-thumb mode to existing fixture script; defaults unchanged.
- [x] Verify CPU helper tests, compile, and launch one GPU Isaac instance.
- [x] Analyze measured normal force, tracking and coupling; preserve evidence and state limitations.

Alternatives rejected for this first isolation: changing rewards cannot isolate physics; immediately increasing effort confounds the current failure. A later bounded effort comparison requires evidence from this unchanged-effort test.

## Evidence-driven follow-up

v137 completed both repeats without faults. No-load tracking is accurate; loaded hold force is only 0.0534/0.0700 N and thumb position error is about 0.14 rad. The model-estimated PD torque clips to 0.2 N·m, but this is not proof of actual drive output. Run v138 with exactly the same fixture and one repeat each at 0.2/0.4 N·m, plus total projected joint force and incoming wrench readback. The latter includes constraints and must not be called isolated motor torque. Reject any hardware-calibration or production deployment claim.

v138 completed all eight cases: doubling effort increases steady normal force only from 0.0534/0.0700 to 0.0985/0.1172 N. Add a one-variable PGS(0) vs TGS(1) diagnostic at original 0.2 N·m, no other setting change. This probes numerical load transfer rather than increasing physical limits. Stop on first original fixture guard; no production solver migration authorized by a fixture result alone.

## Bounded position-lead candidate

v139 TGS stopped at 0.7917 s on left loaded-thumb mimic_error; rejected. v138 PGS traces show large target error coinciding with declining sustained reaction. Test a single control change in v140: limit the requested thumb target to measured q ±0.008 rad, then apply the existing 0.05 rad/s slew bound. Retain PGS, 0.2 N·m, all contact/coupling/guard settings. This is a diagnostic anti-saturation hypothesis, not a hard tracking-error clamp, since slew has priority if external motion is fast. No production patch unless subsequent grasp-loop validation supports it.

## Original grasp loop check

v140 repeated 1.606/1.831 N steady force at original 0.2 N·m without faults, but mimic error remains 0.0053/0.0074 rad. v141 compares only thumb position lead 0/0.008 rad with original 0.004 lift gate, 4/8 N fixed controls, two spatial repetitions. Bounds and slew retain priority; no optimizer updates. CPU tests verify other joints/default behavior and lift gates are unchanged (51 passed, 1 skipped).

Final v141: 8/8 episodes completed without hard faults, all phase timeout; 0 early lifts, 0 successes. Candidate remains diagnostic-only. Dominant remaining blocker is loaded left finger22 mimic error and downward friction. Results documented in `../specs/2026-10-26-thumb-load-isolation-results.md`. No production deployment or reward edit.
