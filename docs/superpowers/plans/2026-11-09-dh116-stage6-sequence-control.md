# DH116 Stage 6 Sequence Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the user's Stage 6 sequence-supervision, exact conditional command shadow, closed-loop branching evaluation and bilateral recovery code, then run finite native experiments without changing old paths or physical success criteria.

**Architecture:** Retain the 3844D four-frame Actor, 371D Critic, canonical24 actions, Tanh and hemisphere preprocessing. An opt-in native sidecar records pre-control state and all eight measured feedback packets. A shared pure-Tensor command kernel supports faithful recorded-feedback unfolding; privileged sidecar data never enters the Actor. Native source/branch/heldout and hybrid control results are independent.

**Tech Stack:** Existing PyTorch/IsaacLab, Python3.10 server environment, pytest, YAML/JSON/Torch provenance, one Isaac process at a time.

**Base and workspace:** `26f1bbcc5a0e596da55c5d89896befcb5622cda7`; existing linked worktree `C:/Users/xk/Documents/project/dh116-publish-11-5`, branch `codex/dh116-lift-repair-20261106`. Preserve untracked prior artifacts. Use new remote outputs and content-only source snapshots; never mutate Stage 5 freezes/checkpoints.

## Task 1 — Identity-bound sequence data (worker A)

**Files:** new `src/go2_pvcnn/control/dh116/tabletop_vision_sequence_dataset.py`; add opt-in helpers to `tabletop_vision_supervision_dataset.py`; tests `tests/test_vision_sequence_dataset.py`.

- [x] Write and run failing tests for duplicate/missing `(run_uuid,env_id,episode_uid,tick)`, wrong issued/previous actions, nonmonotonic policy/physical clocks, reset/terminal/gap crossing and masked padding. Existing `validate_dataset()` must retain v1 behavior.
- [x] Implement `validate_sequence_context`, `join_sequence_context` and `SequenceDataset(data, context, dataset_sha256=..., sequence_length=16)`. Join by unique identity and verified data/source/action-contract SHA, not array positions. Preserve original observations/history.
- [x] Implement actual CONTACT false→true, START ticks0–3, actual qualified HOLD, and reliable reference event or opposed-contact plus three policy samples of measured mean palm vertical velocity >0.0005m/s for LIFT_ONSET. RECOVERY requires separate native qualification.
- [x] Implement source→episode→window sampling, 0.4 uniform/0.6 event mixture and supplied event weights. Every legal window has positive probability; missing classes renormalize. Same-source replay copies stay together; one group is fit_only.
- [x] Test exact probability normalization, deterministic private-generator sampling and untouched inputs. For present long successful episodes, use full16 windows so B0/B1/B2 each expose exactly64 valid samples/update; explicitly report any short padded-window exposure separately.

## Task 2 — Exact shared command shadow and native sidecar (worker B)

**Files:** new `tabletop_vision_command_shadow.py`, sidecar capture helper; targeted extraction in `tabletop_tensor_world.py` and controller helpers; opt-in `sequence_context_sink` in `tabletop_vision_teacher_collection.py`; tests for the shared kernel, sidecar and old-path parity.

- [x] First reproduce and test the actual native command ordering: velocity integration, active profile corrections, measured-q lead clamp, joint limits, original slew/redistribution/pressure/final clamp; translation/rotation norm, target/transport/clearance/safe_z, allowed/live and seed intrinsic XYZ. Refuse unsupported callbacks rather than omit them.
- [x] Expose `shadow_step(command_state, action24, measured_context, controller_contract)` and `shadow_sequence(initial_command_state, actions, contexts, controller_contract)` using float64 shared pure functions and ACTION_CONTRACT scales. Return finger/palm/wrist_angle targets at every physical substep and branch masks.
- [x] Add tests for finite limits, norm bounds, original guard order, inactive masks, accumulated commands, gradients, unsupported profiles and identical old-path output. Shadow never calls SDK or advances physics.
- [x] Capture pre-action command state, pre-control measured packets and post-control original targets for all8 substeps, explicit clocks/UID and terminal truncation. Backend targets, cache and reconstruction remain separate. Context missingness is explicit.
- [x] Recollect native successful Teacher data into a new directory and compare Teacher shadow against recorded float64 targets and separately float32 native casts before enabling B2. Preserve old pools and labels byte-for-byte. Actual02:4698 rows,37584 live substeps,4638 full16 windows; strict current source SHA passed.

## Task 3 — Losses, BC training and model contracts (root with worker A)

**Files:** new `tabletop_vision_sequence_losses.py`; add independent `fit_actor_sequences` to `tabletop_vision_supervised_bc.py`; opt-in activation in `tabletop_vision_student.py`/`tabletop_vision_squashed_policy.py`; versioned contract/loading support; new tests.

- [x] Test masked action MSE, adjacent valid delta-error MSE and SmoothL1 normalized finger(.01rad)/palm(.0005m)/rotation(.01rad) target losses. Verify masked values contribute no gradient and expert targets are detached; invalid/missing contexts reject command training.
- [x] Keep old `fit_actor()` unchanged. Flatten true[B,L,3844], one tanh only for unbounded Actor, independent Adam, clip norm.5, private seed116601. Teacher/Student unfold their own commands against the same recorded feedback, not fake future dynamics or rewritten previous actions.
- [x] Add activation modes relu(default) and leaky_relu_001(new explicit contract only), preserving3→32→64→64 and legacy outputs. Main E0/B0/B1/B2 all inherit current relu and hemisphere, no new encoder reset.
- [x] Train from verified c58 candidate,3000 updates,4×16 batches,1e−4 LR; B0 uniform action-only, B1 event action-only, B2 event action+command ramp0→.1 between250/500. Actual three finals each192000 valid/0padding; snapshot/optimizer/RNG every250. B3 remains gated on native improvement, not training loss.
- [x] Every250 probe START/CONTACT/LIFT_ONSET/HOLD activations, gradients, coordinate-zero/permutation/small-geometry interventions. Actual final three models remain coordinate-sensitive; repeated-dead save/stop covered by tests. No hidden reset or train-MSE selection.

## Task 4 — Finite native branching/source/heldout entrypoints (root)

**Files:** new `tabletop_vision_branching_eval.py`, `scripts/run_dh116_stage6.py`, `configs/dh116/stage6_pinch_sequence_bc.yaml`; opt-in native training-script dispatch; tests.

- [ ] Provide runnable `prepare-data`, `train-bc`, `diagnose-branch`, `evaluate-source`, `evaluate-heldout`. Manifests bind actual model/data paths and SHA, assets/camera/preprocessing/source version; no guessed filenames or overwrite.
- [ ] Source evaluation uses the4 actual successful pinch initial states and pure Student from reset to first terminal. No Teacher/phase/GT coordination after start. Report source reproduction, not generalization.
- [ ] For CONTACT/LIFT_ONSET/HOLD ×1/4/16/60 steps, cold-reset and replay actual successful Teacher prefix before each branch, then Student. Separate first numeric divergence from task-related contact/tracking/slip/force/limit divergence against repeated Teacher range.
- [ ] Freeze a new final manifest before training; previous200 remains regression only. Run fixed-source and independent native comparisons for E0/B0/B1/B2, with each model's results/first-terminal traces separate.
- [ ] Expand expert data only through actual successful new resets: first±.5mm XY/±1°yaw, then conditional±1.5mm/±3°. Save failures separately. Corrective segments require independent schema and successful recovery; do not call an unqualified time-index oracle DAgger.

## Task 5 — Independent full-episode tracking and bilateral recovery (worker C)

**Files:** `tabletop_bimanual_tracking_guard.py`, `tabletop_tripod_recovery_runtime.py`, `tabletop_tripod_execution_guard.py`, associated tests and explicit candidate runtime.

- [x] Preserve current default and right-only Stage A. Introduce explicit hand×digit indices and per-hand acquisition origins/cancellation/pressure masks, enabling right/left/both only by opt-in.
- [x] Test every env/episode/hand/digit's cumulative1mm/1s budget, .5s no-progress stop, no reset on flicker, native post-guard billing, mandatory release precedence and preserving bearing digits.
- [x] Missing-pad geometry uses that hand's actual FK/Jacobian/pad region and original nonlinear collision checks; numerical convergence never implies real contact. Keep actual cooked/live qualification gate closed when missing.
- [ ] Add separately qualified共同双腕pause state machine: per-joint Teacher-calibrated trigger=max(.01rad,1.5×P99.9),20ms q extrapolation warning, original terminal highest priority, measured-state reconnection, at least.05s recovery. Do not change safety thresholds or feed privileged guard into pure Student silently.
- [ ] Run full original/unilateral-failure/共同pause first episodes and Teacher regression; report new overforce/left failure/deadlock. Fixed-wrist bilateral acquisition preserves stable pressure contacts. Contact-stable.2s, conditional load-transfer.5mm/s≤2mm then lift1→3mm/s and original30mm/5s success remain Teacher-only gates; moving-wrist recovery cannot bypass geometry.

## Task 6 — Review, verification and publication

- [ ] Run focused CPU red/green and legacy parity, then broaden only for new coupled changes; compare inherited failures rather than claim whole-suite green. Freeze sources and input SHA for each native job, use the existing single-instance lock and never stop other training.
- [ ] Independent spec/quality reviews for modules and integration. Test source M0/oldBC/oldPPO unchanged and no privileged Actor expansion.
- [ ] Update README/CHANGELOG and evidence reports at every commit/push, with actual command, outcome, limits and remaining gates. Preserve raw data/models; do not commit artifacts/runs/secrets. Verify remote SHA and PR staysDraft, never merge.
- [ ] Short PPO only after pure-source Student success plus independent validation without regression; retain rewards/Tanh, actor3e−5/critic1e−4,2epochs/rollout32 and expert auxiliary. Keep Actor-Adam reset/preserve as separate candidates. If gate fails, report it and do not launch PPO.
