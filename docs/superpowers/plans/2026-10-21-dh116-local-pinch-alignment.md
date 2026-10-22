# DH116 Local Pinch Alignment Execution Plan

> **For agentic workers:** Use superpowers:executing-plans for the user's selected Inline Execution. No subagents, commit or push.

**Goal:** Validate the approved ≤3 mm local pinch adjustment in real CUDA-rod/Isaac close, hold and lift, without changing acceptance thresholds.

**Architecture:** Use the actual V20 step3103 physical initialization and unchanged USD/mesh/reference artifacts. Optimize hand coordinates only with Panda frozen, then use the existing `reframe_close_initialization` and original verifier. New geometry must earn all phase evidence again; a static pose is not an expert trajectory.

**Tech Stack:** Existing USD FK, Open3D mesh queries, SciPy least-squares, Isaac and CUDA Cosserat rod.

## Global Constraints

- Actual four surface-point shifts ≤0.003 m; bilateral axial mismatch must improve.
- Preserve Panda joints, palms and Panda-only symmetric lift allocation; hand MPC remains12D.
- Original6 pad partitions, source clearance≥1 mm, loaded-gap preservation≤10 µm for candidate screening.
- Original contact load, normal separation, slip≤2 mm, qualification/dwell and lift/whole-wire-clearance gates unchanged.
- Original USD/rod/material/force caps/100 µs physics/16 contact subsamples unchanged.
- No existing live Isaac; verify this immediately before a single new launch. Do not restart for a poll timeout.
- Initialization is a warm, already-contacting physical continuation, not a general from-open skill or imported phase success.

## Task 1: Actual surface/joint screening

Files: generated diagnostic `runs/dh116_contact_mpc_close_gpu_20261021_v20/local_alignment_pose_probe.py` and report `local_alignment_pose_probe_v1.json`; no production source changes.

- [x] Compare original real meshes and actual rod at pinned3103 state, not display markers.
- [x] Fit Panda-fixed thumb2/index1 coordinates with the same parked fingers and original joint bounds.
- [x] Check complete source geometry and all6 original pad regions with the existing physical interval envelope.
- [x] Select `hand_only`, fraction1, where bilateral axial mismatch and all3 mm/10 µm bounds pass. Do not select offline palm proposals or enable `opposed_pinch`.

Reproduce:
```bash
DH116_NUM_THREADS=1 bash scripts/python.sh runs/dh116_contact_mpc_close_gpu_20261021_v20/local_alignment_pose_probe.py --output runs/dh116_contact_mpc_close_gpu_20261021_v20/local_alignment_pose_probe_recheck.json
```
Expected: hand_only/fraction1 `static_candidate_passed=true`; this is not physics success.

## Task 2: Export and validate a fresh warm-close state

Files: generated `runs/dh116_local_pinch_alignment_20261021_v2/{pose.json,initialization.json,artifact_check.json}`. Existing implementation: `src/go2_pvcnn/control/dh116/close_initialization.py::reframe_close_initialization(payload, states, alignment=report)`.

- [x] Reject the export unless the selected report passed and candidate Panda coordinates exactly equal the pinned source.
- [x] Export using the existing reframe function. Preserve full rod state, robot velocities and initial drive deflections; set all qualification flags false. Original point references and manifest remain unchanged.
- [x] Load through `load_active_pose` and `load_close_initialization`, verify real joint/target limits and original physical dynamic state. Keep the provenance of inherited load-reference/bias values; these are command references, not measured qualification after reframing.
- [x] Run existing affected state/IK tests (18 passed in1.01s):
```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_close_initialization.py tests/test_dh116_arm_ik.py -q
```
Expected: exit0. No new production behavior is implemented, so no new production RED/GREEN test is required for this generated experiment.

## Task 3: Real contact verification and complete recording

Files: original verifier, new output `runs/dh116_contact_mpc_close_gpu_20261021_v21`, log `/tmp/dh116-contact-mpc-cuda-v21.log`; no runtime source edits.

- [x] Check no project verifier/Isaac process exists and new output does not already exist.
- [x] Launch the original V20 CLI with only `--output`, `--active-pose`, `--active-pose-sha256`, `--close-initial-state`, and its SHA changed to Task2 artifacts. Keep CUDA rod and every original acceptance criterion. PID157969/session21797; recheck actual process/summary on resume.
- [x] Check actual first-frame source/pad gates, contact forces and12D hand/Panda allocation. Actual ongoing prefix3 completev2 intervals validates; formal source rejects incomplete run.
- [x] Follow this same PID through original1s hold: hold14700/time1.4701s→lift24700/time2.4701s, no restart or phase credit. Actual249 completev2 intervals(148close/100hold/1lift), including true.1ms boundary, pass time/token/state-chain validation but remain diagnostic. Panda owns transport; hand_mpc_hold keeps12D targets.
- [x] Follow V21 to actual terminal27063/lift.2363s: rightthumb separation1.2154mm/s invalid under original1mm/s gate despite.9544N load; slip.1852mm, wire not lifted. OldPID/session ended; retain diagnostic source, no qualification or formal training. Local alignment alone did not solve transport. Actual14699 close backup and all prior files preserved.
- [x] Follow separate numerical experiment V23 to terminal16922, no stop/restart for polls. close/holdPASS,liftFAIL: right non-pad finger13 source guard .999886mm<1mm, all6pad-region checks pass. Last fullvalid16921/slip1.413684mm retains4contacts, but wire not raised. Keep failed data diagnostic, no full-expert qualification.
- [x] Screen hand-only candidates from V23 validclose2024 with cumulative3mm surface-point budget measured from original V20 points. Four alignment/lowering objective trials yield3static passes, actualpoint shifts<1.6mm and original10µm gaps preserved, but normals remain downward. Corrected loaded-gap manifold scan tests82thumb states per side; aligned selected candidates pass all original source/pad checks but still no upward normal support. Diagnostic run: `runs/dh116_local_pinch_alignment_20261022_v3`; no exported state or runtime change. Initial scan's endpoint-only root bracket missed local branches and is superseded by `support_scan_v2.json`; it is not an infeasibility proof.
- [x] Offline-only palm-down2mm comparison stays<1.34mm actualpoint movement but fails leftfinger13/table .966722mm under unchanged1mm. It is outside Panda-fixed deployment and was not selected or executed in physics. No source algorithm, friction/caps or success gates changed.
- [ ] Obtain next design choice if palm/arm posture adjustment is needed. Do not quietly deploy an offline proposal or infer a global impossibility from a finite static scan. Actual new full expert grasp/lift and formal distillation remain incomplete.
- [ ] Only if actual terminal grasp/lift AND whole-wire clearance succeed, use the formal complete replay loader for successful expert training. Otherwise retain diagnostic-only data and report the exact failed stage; no formal teacher/student fit.
- [ ] Update README, CHANGELOG and root CODEX_RESUME with actual results, live PID if applicable, improvements, limits and next action. Do not mark the overall successful-expert/distillation goal complete from geometry/tests or launch alone.
