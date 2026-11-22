# Archived Teacher Common Source Semantic Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Reject archived Teacher common tensors that disagree with their original named mounted-hand source packets, even when internal hashes were recomputed.

**Architecture:** Apply the already approved execution-chain repair design sections4.3 and8 to the existing archive input owner. Reuse validate_mounted_observation before constructing common371; retain all existing raw-rod, parameter, clock, owner and SHA joins. Historical Pinch identity replay is a separate work package.

**Tech Stack:** Existing Python3.10.12 / PyTorch2.7 CPU environment; pytest. Ubuntu existing isolated11-17 worktree; Windows publication relay.

## Task1: Reproduce at the owning archive boundary

**Files:** Modify tests/test_native_teacher_actual_inputs.py, existing test_rejects_bad_durable_input_binding parameter table.

- [x] Add common_values case with expected mounted observation mask/values differ rejection.
- [x] In that case mutate bootstrap pre_action.common.values[0] by0.01 and observation.common371[0,0] consistently. Recompute pre_action/observation pack hashes and matching bootstrap reference hashes; leave original source_packets unchanged.
- [x] Run project Python -m pytest -q tests/test_native_teacher_actual_inputs.py -k common_values. Require failure DID NOT RAISE ValueError before changing production.

## Task2: Minimal source-owner fix

**File:** src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_inputs.py:_rebuild

- [x] Import validate_mounted_observation from tabletop_teacher_mounted_wrist_source.
- [x] Immediately after common=buffers["pre_action"]["common"], call validate_mounted_observation(common) before tensor conversion.
- [x] Run the new case and whole source-owner test file; then related model/contact-source files. Expected all pass. Do not change existing failure tests to accommodate unrelated errors.
- [x] Re-run existing real Tripod8 archive source reconstruction and8-action seeded numerical replay as offline analysis, with original raw SHA fixed. No new native data.

## Task3: Report and publish

- [x] Update report, readiness, Registry, CHANGELOG and README, explicitly retaining historical Pinch regression and native/grasp gates.
- [x] Verify528 protected files and3 historical Checkpoints, inspect exact diff; only stage the relevant implementation/test/plan/report files.
- [ ] Ubuntu commit; Windows imports identical commit, normal push11-17; read back remote SHA/tree/file list and Draft PR2.
- [ ] Publish manifest attestation in a separate commit; save private recovery. No Isaac/PPO launch or new physical success claim.
