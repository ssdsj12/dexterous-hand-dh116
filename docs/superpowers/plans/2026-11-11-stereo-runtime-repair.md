# Stereo runtime repair implementation plan

**Goal:** Repair the actual sensor-to-policy, identity and PPO data paths requested after9bfee42, without repeating J0/J1/J2-A/J2-B or changing physical success standards.

**Authority/baseline:** User supplied detailed P0/P1 design and explicitly requested implementation. Live ls-remote confirms9bfee426363527190011dbca164c104a00cbc999 unchanged. Existing local origin tracking ref was stale; live remote and local HEAD agree. No push/merge, no long training. Legacy models/checkpoints and all528 protected original files remain unchanged.

**Architecture:** Keep explicit legacy topology and latent/world policy paths. New tracker separates image topology from depth validity; deployable TaskStateAdapter selects versioned state fields and feeds the action network; paired palm-local points use a shared encoder with separate hand identity. Teacher native ledger is bound to actual checkpoint and camera identity before execution. Unique-frame CPU storage reconstructs only requested visual minibatches; accelerated pyramid is an independent opt-in mode. Software, bounded native checks, performance measurements and actual grasp qualification remain distinct.

- [x] P0-1: semantic staircase RED, normalized topology/visible fragments/temporal identity implementation, missing-depth and reset regressions; keep legacy comparator.
- [x] P0-2: field-by-field371D adapter, direct verified proprio/config, masked estimates and uncertainty; latent/task-state/hybrid action paths, separate configs/contracts, gradient and leakage regressions.
- [x] P0-3: measured FK palm-local typed sensor/history fields, shared point encoder/hand embeddings, translation-invariant local policy branch with explicit world auxiliary/legacy path; runtime and independent inference wiring.
- [x] P0-4: nonzero actual camera/checkpoint identity, execution owner before first action, sampled/mean/issued/applied distinctions and raw-step time/identity tests.
- [x] P1-1: unique exact RGB/depth/mask frame pool, indexed histories, scalar PPO ledger and minibatch reconstruction; preserve trainable visual encoder; exact input and frozen logprob tests; record real resource benchmarks separately from native collection throughput.
- [x] P1-2: explicit audit/accelerated pyramid with credible priors and global fallback, search counters/latency/fallback and fixed-reference comparison.
- [x] Integration: old/new CPU regression on local and existing server environments, source freeze, README/CHANGELOG/readiness/report and failure evidence. P1-3 rod interface/source readiness and P1-4 native prerequisites audited without fake solver or success. P2 formal training remains gated on actual qualified sources/Teacher success.

Ownership: tracker agent perception/tracking/tests; adapter agent new TaskStateAdapter/tests; identity agent privileged_cli/native/tests. Root owns observation/runtime/inference/encoder/state_estimator/Student actor integration; later isolated ownership for storage/pyramid. Logs remain under private artifacts/stereo-runtime-repair-20261111. Each item records test names, actual results and unresolved native gates before proceeding.

Final evidence: Windows 1030/1skip; server 1030/2skip. P0/P1 software and bounded runtime checks complete. Deformable coupling, native accuracy, checkpoint epochs across multi-update episodes, successful Teacher, and formal BC/DAgger/Student PPO remain explicit qualification gates; none is claimed complete.
