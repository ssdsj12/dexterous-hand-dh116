# DH116 V2 Load Identification Implementation Plan

> For agentic workers: execute inline with systematic-debugging and test-driven-development. Request independent review before final implementation acceptance.

**Goal:** Produce a documented V2-aligned simulation configuration with demonstrated useful grip-force authority and restart PPO after validation.
**Architecture:** Separate factual manufacturer metadata, explicit candidate simulation parameters, physical readback and identification evidence. Preserve legacy contracts and runs.
**Tech Stack:** Python, NumPy, USD/PhysX, Isaac Lab, SB3 PPO, pytest.

## Global Constraints

One Isaac process; CPU0/numeric single thread; no fabricated force or success; preserve .02rad hard protection; no claim of complete hardware calibration without hardware data; no commits or push.

- [x] Stop and preserve v47 (zero learning updates).
- [x] Run four-slot coupling sweep using identical .55N requested load, 400/800/1600/rigid coupling and unchanged drives. Record live load, mimic warning/stop, pose and lift. Reject any hard fault. v48 and contact-last comparison v49 both exited normally; no actual object lift. Rigid coupling increases load but does not solve grasp loss.
- [x] From measured evidence choose the next single-variable force-authority probe. Compare actual contact, not implicit PD estimated torque alone. Check contact order/gains/effort only if required by evidence.
- [x] Implement a versioned parameter profile with regression tests first: reject nonfinite/out-of-range parameters; preserve legacy defaults; serialize every changed controller/physics quantity into checkpoint contract.
- [x] Add physical parameter readback and explicit approximate torque labeling. Validate mass/range assumptions against V2 manual and current USD before adopting changes.
- [x] Run related offline tests, controlled-load trials and random-action admission. Review implementation independently.
- [ ] Start the verified 512-env PPO configuration, confirm actual learning progress and checkpoint output; document remaining identification uncertainty honestly.

Experimental diagnostics are saved separately from production source and never treated as policy learning. Candidate values are chosen by measured probes, not assumed in advance.

## Manufacturer website follow-up

The supplied Feishu library links to the official model repository and SDK releases. All four local V2 URDF/MJCF files are semantically equal to official commit 14275cfe32cce1fe13a16df712e8a0e610a0227e. Thus re-downloading does not repair the thumb-range conflict or the finger32 mass/COM anomaly. Do not silently fill the base mass to 490 g and describe the entire inertial model as calibrated. Full details and source-access limitations: `docs/dh116_v2_web_search_20261024.txt`; machine-readable evidence: `docs/dh116_v2_web_evidence_20261024.json`.

## Correction evidence (2026-10-24, this execution)

v50 compared gain/effort (20/.2, 50/.2, 100/.2, 100/.5) with rigid linear coupling at .55 N requested hand load. All four reached lift but none lifted the object; increasing motor limits alone is not a validated remedy.

Exact source-USD forward kinematics identified 26–43 mm/rad upward drift in the old pressure direction. The corrected seed solves the relative pad Jacobian along the diameter and compensates midpoint displacement with bounded palm motion. At +.08 rad, offline nonlinear height error is at most .342 mm. This is local feedforward: actual movement remains subject to tracking, clearance and command limits, with clipping/target-error diagnostics.

v51 compared compensated/uncompensated geometry at .55/1 N. v52 compared 16/64 position iterations using corrected geometry at .55/1 N. No hard fault and no successful lift in either four-slot probe. v52 reached close/hold/lift in all four slots; maximum object rise .143 mm. Final contact normals include downward force on the object after pads slide upward. Do not describe phase entry or commanded force as successful grasp.

New explicit simulation profile: requested hand load .1–4 N, .5 N/s slew, existing 10–20 Nm/rad gains and .2 Nm effort. The request ceiling is not measured grip capacity and is not the manufacturer fingertip-force rating. Rigid linear coupling is opt-in at 480 Hz only; old defaults/checkpoints remain separate.

Offline verification: 253 passed (one expected overflow-warning protection test); mimic/CLI tests 56 passed, 1 skipped. Independent review found a legacy .15 Nm readback mismatch, now fixed and tested. A reviewer's standalone USD import failed because its conda shared-library path was absent; project-wrapper test passed.

v53 GPU8 physical preflight read back gains/damping/effort for all 16 hands and verified object material, reset isolation and measured bilateral contact. Full random-action admission and 512-env restart are recorded separately after completion.

GPU8 admission finished:24/24 safe repeatable episodes,0 grasp successes,exit0. v54 launched512-env fresh-policy pipeline with1048576 budget. Its own admission and actual updates/checkpoints remain pending; the restart checklist must not be marked complete until those are observed.
