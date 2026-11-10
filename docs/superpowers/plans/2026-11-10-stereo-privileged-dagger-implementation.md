# DH116 Stereo Privileged Teacher–Student Implementation Plan

> **For agentic workers:** Use subagent-driven development. Follow Phase 0–7 in order; verify and commit each isolated phase. The user's follow-up authorizes continued implementation. Native qualification gates remain mandatory.

**Goal:** Implement an independent trainable privileged PPO Teacher, synchronized stereo RGB perception, temporal state estimation, BC, genuinely online DAgger, asymmetric Student PPO, and evidence-based evaluation.

**Architecture:** Keep every original controller, checkpoint, RGB-D observation, reward and physical success implementation unchanged. New modules use a versioned shared canonical 24D action digest and the audited 371D layout. Native scene construction is delegated to the existing explicit launcher in an isolated process; only the training callback is substituted, never the controller. Student input objects reject privileged fields. Native evidence gates control training admission and promotion separately from CPU implementation readiness.

**Tech Stack:** Existing Python/PyTorch; NumPy geometry; optional existing OpenCV StereoSGBM; pure PyTorch ResNet18 and Transformer; existing Tanh Gaussian arithmetic. No dependency installation or environment upgrade.

## Phase 0 — source contracts (read-only production files)

- [ ] Complete 371 rows, canonical 24D mapping and reward arithmetic audit.
- [ ] Test real CPU observation and controller kernels, valid/invalid feedback, action grouping, dt scaling and duplicate reward risks.
- [ ] Freeze source SHA manifest; distinguish legacy 261D/26D expert qualification from current 371D/24D.
- [ ] Record native support/contact attribution as unqualified; do not turn NA into zero labels.
- [ ] Run tests and commit docs/tests only. The original 11-10 ref remains unchanged.

## Phase 1 — privileged Teacher

- [ ] Add `tabletop_privileged_{teacher,ppo,reward,dataset,checkpoint}.py` with 371→256→256 actor/value networks and saved-latent Tanh Gaussian PPO.
- [ ] Test GAE terminal boundaries, stored action/log-prob identity, minibatch coverage, independent BC optimizer, resume contracts and exclusive outputs.
- [ ] R0 consumes the existing task reward exactly once. R1 geometry and R2 support shaping require explicit source-qualified evidence and remain disabled by default.
- [ ] Add collection/training/evaluation CLIs and baseline configs. Reject unqualified expert success pools and fake PPO transitions.
- [ ] Native Teacher full-episode success is **pending**, not implied by optimizer tests.

## Phase 2 — synchronized stereo RGB and geometry

- [ ] Add calibration, camera, depth, geometry and validation modules plus opt-in probe/evaluation CLIs.
- [ ] Two independent RGB cameras, one physical timestamp, render-only acquisition, strict calibration readback and reset/frame identity.
- [ ] Implement rectification, SGBM matching with left/right checks, confidence and explicit invalid masks; no simulator depth fallback.
- [ ] Backproject optical +X right/+Y down/+Z forward column vectors in metres; transform through world to current measured palm FK.
- [ ] CPU analytic fixtures verify geometry, masks and stale rejection. Isaac acquisition/accuracy remains pending until actually run.

## Phase 3 — stereo temporal estimator

- [ ] Add shared ResNet18, stereo cross-attention, geometry/proprio fusion, explicit spatial/time encoding and four-frame temporal Transformer.
- [ ] Output geometry/velocity/deformation/contact/force-proxy/slip and compatible 371D estimates. Apply per-field supervision masks from Phase 0.
- [ ] Quaternion supervision uses SO(3) geodesic loss; unavailable/invalid force and internal task-state supervision is excluded.
- [ ] Test shape/device/finite checks, missing views/points, history/reset isolation, prohibited Actor fields and differentiable masked losses.

## Phase 4 — synchronized Teacher collection and BC

- [ ] Add versioned pre-action RGB/calibration/proprio/state/action dataset and collector; raw/sample/mean/issued/applied actions remain distinct.
- [ ] Require exact frame/episode/time alignment and qualified full native Teacher success for successful demonstration admission.
- [ ] Train Student with detached Teacher mean, group-normalized action loss, masked state/depth/temporal losses and separate gradient diagnostics.
- [ ] Provide independent BC CLI/checkpoint. CPU fixture training is explicitly not qualified demonstration data.

## Phase 5 — online DAgger

- [ ] Query Teacher on the exact Student-visited pre-action privileged state, then execute a complete Student action (or declared whole-action owner schedule).
- [ ] Store Teacher labels without Teacher takeover; reject time/reset/history mismatches and stale samples. D0 is reference only.
- [ ] Keep failure-state labels separate from successful BC pools and PPO storage.
- [ ] Gate formal native DAgger on qualified Teacher evidence; isolated CPU collector tests do not satisfy that gate.

## Phase 6 — asymmetric Student PPO and deployment

- [ ] Stereo-only Actor with separate privileged value network; on-policy storage retains saved latent, exact issued action and original log probability.
- [ ] Reuse Phase 1 PPO math through typed actor/critic interfaces. Auxiliary BC uses separate optimizer and provenance.
- [ ] Add independently runnable inference/export and train CLIs. Deployment excludes Teacher/MPC/Critic and rendered RGB-D depth.
- [ ] Test privileged-input isolation, action-contract/version rejection and complete resume without changing old models.

## Phase 7 — evaluation and promotion

- [ ] Implement per-grasp metrics, Wilson 95% intervals, unique physical initial-state identity and failure breakdowns.
- [ ] Report Teacher, stereo, estimator, DAgger and Student results separately; require ≥100 different native initial states per grasp for final admission.
- [ ] Verify 30 mm/5 s/0.5 mm and unchanged task gates from actual terminal evidence. No fixture, clone or truncated episode qualifies.
- [ ] Save a machine-readable readiness report distinguishing implemented/CPU-tested/native-validated/task-qualified.

## Verification and preservation

Run all new tests on the already available CPU interpreter, with CUDA disabled and no Isaac imports at module import time. Preserve failed logs in new artifact files. Check all originally tracked `src/`, `scripts/` and `configs/` file SHA values after each phase; only new independent files may differ. Update README, CHANGELOG and phase reports with actual command/exit/test counts and outstanding native gates. Keep the PR Draft. Do not push, merge, launch long training or declare physical success without the corresponding authorization and evidence.
