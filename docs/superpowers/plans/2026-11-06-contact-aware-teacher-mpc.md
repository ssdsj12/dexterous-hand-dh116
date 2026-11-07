# Contact-aware Teacher MPC and both-style native validation

> **For agentic workers:** Use subagent-driven-development for independent evidence review, then test-driven implementation. Native execution is owned by the root worker and serialized by `/tmp/dh116-native-single-instance-xk.lock`.

**Goal:** Improve the failing Teacher-reference MPC using actual contact evidence and fully evaluate pinch and tripod separately, without changing PPO or physical success criteria.

**Architecture:** Keep the published local-pad MPC and legacy PPO paths intact. Add an explicitly selected experimental contact-aware controller only after diagnosing its failure. Keep native Teacher/MPC collection separate from visual Student inference and PPO storage. Every executed candidate uses a frozen source snapshot and a new output directory.

**Tech stack:** PyTorch CPU regression tests, Isaac native 480 Hz physics / 60 Hz control, actual RGB-D, the existing first-terminal evaluation and supervision collectors.

## Frozen baseline and invariants

- Branch starts at `b63501e8619f777afedc995a7cf88b6524ef0adc`; PR remains Draft.
- Preserve 3844D visual Actor, 371D privileged Critic, canonical 24D actions, Tanh Gaussian PPO and the original v2 route.
- Preserve original actuator bounds, materials, reset contracts, native 30 mm / continuous 5 s / 0.5 mm success criterion, and historical Checkpoints.
- Simulated contact/force/slip may guide a training-only expert; they are not hardware feedback or Student observations.
- Do not launch long PPO, inject external object forces, attach objects, change supports, or train from failed/unqualified MPC demonstrations.
- Full Student baseline uses the existing frozen 200-case manifest (100 unique initial states per grasp), deterministic actor-only inference and a 2400-control-step budget. An unfinished 180-step probe is not complete grasp validation.

## Tasks and acceptance

- [x] Diagnose pinch MPC failure from paired native command, pad-load, opposition and joint-prediction traces. Distinguish a measured correlation from a proven cause. Check whether incremental reference actions permit accumulated command drift.
- [x] Diagnose tripod's pre-lift `excess_force` from the actual first-terminal substeps and original threshold, including the responsible hand/link and whether a valid pad contact exists. Do not relax the guard.
- [x] Execute the current BCv2 Student on the frozen 100-per-style manifest while code work remains CPU-only. Record source SHA, actual reset identities, complete first terminals, and separately report invalid/inapplicable telemetry.
- [x] Specify the smallest evidence-supported MPC change; write regression tests that fail on the existing behavior before implementation. Cover bounded 24D commands, episode reset, reference indexing, nonfinite rows, per-environment isolation, contact-loss handling, and actual-issued-command prediction accounting.
- [x] Keep the old MPC selectable and export the new version, parameters, measured inputs, decision reason, nominal/issued actions and source identity in diagnostics. Default PPO behavior stays unchanged.
- [x] Run CPU tests, freeze the candidate, then run a bounded both-style native Teacher/MPC experiment with the same initial state/seed/camera as the reference. Execute only the first planned command before observing the next actual state.
- [x] Compare both styles using physical Lift P50/P90/P95/max, 30 mm rate, opposed contact, contact loss, torque saturation (simulation estimate), hold, slip applicability and original native success. A lower model cost is insufficient to declare improvement.
- [x] If a candidate fails, retain it and its raw evidence. Make any follow-up a separately identified single-factor experiment. Do not quietly tune physics or count the best observation as a new policy's success.
- [x] Independently review source changes and native evidence. Update README, CHANGELOG and the validation report with executed commands, results, missing evidence and next steps before committing/publishing the Draft branch.

## Evidence locations

Historical reference/MPC raw evidence: `/mnt/mydisk/xk/dh116/runs/teacher_mpc_supervision_20261106T1145Z`.

Current independent experiments: `/mnt/mydisk/xk/dh116/runs/mpc_contact_v2_20261106T1250Z`.

Editable isolated staging code: `/mnt/mydisk/xk/projects/dh116-mpc-contact-v2-20261106`. Each native job copies a frozen snapshot before execution; later edits do not alter a running job.

Raw runs and diagnostic artifacts remain outside Git. Report native execution and successful grasping separately even when execution exits with code zero.

## Evidence-driven extension: Student representation and startup fitting

The completed 100-per-style BC100 evaluation showed no opposed contact or successes. Independent input replay identified a quaternion double-cover mismatch (same pose, opposite signs) and a separate startup fitting error; inference itself reproduced the recorded actions correctly. This extension addresses those measured issues while preserving input dimensions and the original raw models.

- [x] Add a default-off Actor preprocessing version that canonicalizes only wrist quaternion signs inside the four-frame 3844D input. Preserve all raw sensor artifacts, remaining features and twelve Actor parameter tensors.
- [x] Bind the known adapter field through explicit BC v3 artifacts, strict warm-start derivation, inference, PPO construction, pre-Kit/static/exact resume, and the common front launcher. Historical absent fields remain raw.
- [x] Compare raw1000 and canonical1000 on identical source/initialization/sampling; independently verify all protected source/PPO state. Diagnose startup separately from whole-source MSE.
- [x] Add explicit canonical-BC startup sampling with default uniform behavior unchanged, private RNG and immutable split/distribution provenance. Test rejection of heldout mixing and forged distributions.
- [x] Fit and independently audit canonical1000_startup25 as an offline diagnostic. Record its improved startup error and slightly worse whole-source error; do not count it as a native grasp.
- [x] After the pressure run is audited, select only qualified successful MPC labels, fit an independent canonical/startup-aware Student candidate and evaluate both grasp styles with the same complete 100-case-per-style regression protocol. If a style has no qualified training data, report that limitation explicitly.
- [x] Finalize the native result tables and unresolved contact/phase/model-selection limitations before publishing the Draft branch.

The final candidate has 0/100 native successes in each style. Pinch acquires intermittent contact but does not retain it; tripod has no qualified demonstrations and no opposed contact. The completed experiment/verification tasks do not mark reliable grasping, model selection, broader generalization or Lift repair as achieved. See the validation report for the complete independent audit and deferred experiments.
