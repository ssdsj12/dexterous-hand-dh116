# DH116 P0 bounded-action implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Preserve the current training process and use red/green tests before production changes.

**Goal:** implement an opt-in, probability-consistent squashed visual PPO while preserving legacy3844D actor/371D critic behavior and preparing deterministic native validation.

**Architecture:** stable latent-based tanh probability helpers; a separately versioned visual policy and CPU storage/PPO; then explicit training/checkpoint integration. Current v2 is retained, v3 is opt-in. Hardware20D is a metadata-only future interface, not an actor extension.

**Tech Stack:** Python3.10, PyTorch2.7, RSL3.1.3, TensorDict, pytest; native Isaac4.5 only after existing run completes.

## Completed prerequisite evidence

- [x] Current training identity/GPU3 confirmed without control actions.
- [x] Branch codex/dh116-lift-repair-20261106 created from b42f1aa; original11-5 ref retained.
- [x] Separate server stage /mnt/mydisk/xk/projects/dh116-lift-repair-20261106 copied from current source.
- [x] T0 checkpoint1439, source, original request and complete chain logs preserved; CPU80 tensors finite and SHA verified. Source SHA d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9 refers to the checkpoint, not source archive.
- [x] Baseline command passed81 tests: tests/test_vision_student.py tests/test_vision_ppo.py tests/test_vision_training.py tests/test_vision_resume.py tests/test_vision_palm_lift.py tests/test_palm_lift_summary.py tests/test_palm_lift_telemetry.py.
- [x] Hardware audit delivered before production changes; no confirmed real sensor streams, retain3844/371.

## Task1: stable squashed distribution and isolated policy

**Create:** src/go2_pvcnn/control/dh116/tabletop_squashed_gaussian.py, tabletop_vision_squashed_policy.py.
**Test:** tests/test_squashed_gaussian.py, tests/test_vision_squashed_policy.py.

- [ ] Write red tests for exact bounded sample/latent relation,24D transformed logp, deterministic tanh(mean), no inverse-tanh reconstruction, finite extreme latents and gradients, same-policy ratio1/KL0, invalid input rejection and critic isolation.

```python
z = torch.tensor([[-20., 0., 20.]], dtype=torch.float64)
logdet = 2 * (math.log(2.) - z - torch.nn.functional.softplus(-2*z))
expected = (torch.distributions.Normal(mu, std).log_prob(z) - logdet).sum(-1)
torch.testing.assert_close(squashed_log_prob(z, mu, std), expected)
```

- [ ] Run new tests on the isolated server stage with CUDA_VISIBLE_DEVICES='', PYTHONPATH=src, OMP_NUM_THREADS=2; observe missing new API failures.
- [ ] Implement finite/shape/positive-scale validation and shared stable logdet/logp helpers, analytic latent KL and reparameterized current bounded entropy with supplied epsilon. Normal-range oracle is independent PyTorch TransformedDistribution; extreme cases use stable latent oracle.
- [ ] New visual actor reuses the current PointNet/MLP weights but removes its final output Tanh. Distribution applies the only output squash. Override deterministic inference. Preserve hidden Tanh layers and correct final weight initialization. Keep legacy source unchanged.
- [ ] Re-run red tests to green, then legacy vision-student tests. Review spec compliance before code-quality review.

## Task2: explicit latent storage, microbatch PPO and diagnostics

**Create:** src/go2_pvcnn/control/dh116/tabletop_vision_squashed_ppo.py.
**Test:** tests/test_vision_squashed_ppo.py.

- [ ] Write red storage tests: float32 CPU actions=tanh(z), exact latent_actions, old transformed logp, latent mu/sigma; transition clear/reuse; complete remainder samples; pin after all buffer allocations.
- [ ] Write red numerical optimizer oracle tests sharing rollout/permutation/entropy epsilon across full and micro=1/3/large logical batches. Compare GAE, losses, gradients, parameters, Adam, LR and metrics.
- [ ] Implement RSL Transition/storage subclasses that actually allocate/copy latent_actions. A dynamically attached transition field alone is insufficient.
- [ ] Implement sample_actions/evaluate_latents flow; do not sample PPO actions again during update. Generate entropy epsilon once per logical minibatch, index the same rows for each microbatch. Weight means by sample count, maxima by true global maximum.

```python
log_ratio = new_logp - old_logp
ratio = log_ratio.exp()
approx_kl = (torch.expm1(log_ratio) - log_ratio).mean()
clip_fraction = ((ratio - 1).abs() > clip_param).float().mean()
```

- [ ] Reject nonfinite logp/ratio/loss/gradient before Adam. Keep one clip and Adam step per logical batch. Capture pre-clip norm.
- [ ] Expose last_update_diagnostics separately from the existing value_function/surrogate/entropy loss return shape. Include raw_abs_max,|a|>=.99 element and env-any saturation, per-dim empirical bounded mean/std, latent parameter summaries, old/new logp, frozen-policy consistency, analytic/empirical KL,clipfraction,bounded/latent entropy and gradient norm. State aggregation semantics.
- [ ] Run new tests green and affected baseline suites; independent spec and code review.

## Task3: opt-in training, versioned contracts and explicit derivation

**Modify:** tabletop_vision_student.py, tabletop_vision_training.py, tabletop_vision_resume.py and the actual CLI option/forwarding paths after reviewing them.
**Create:** a narrowly scoped v2-to-v3 derived-checkpoint helper and tests; no implicit migration.

- [ ] Red tests: new sampling action is sent unchanged to env and previous-action history; JSON/TensorBoard diagnostics are present; oldv2 checkpoint rejected for exactv3resume before metadata writes; existing output directory and input checkpoint hashes unchanged on rejection.
- [ ] Keep default legacy paths/tests byte-equivalent; opt-in mode selects v3 maker/version and adds distribution/storage/entropy/KL/inference/latent-std contract. Shared physical ACTION_CONTRACT does not change.
- [ ] New runs require independent outputs; exactv3resume requires identical contracts. Explicit warm-start derivation preserves source bytes, records source SHA, actor/critic/Adam/RNG/LR/counters and marks distribution change. Never call it exact resume.
- [ ] Log diagnostics to iterations/monitor/TensorBoard. Preserve rod/palm definitions and native success accounting.
- [ ] Run affected suites and complete CPU-eligible regression with native/GPU tests explicitly excluded; preserve pre-existing failures separately.

## Task4: deterministic visual evaluation and best checkpoints

**Create:** visual-only evaluation adapter, fixed-trial manifest/aggregator, validation-based checkpoint selector and focused tests. Reuse sensor camera/history/observation packing; never feed371D observations to visual actor.

- [ ] Red tests for deterministic inference, first-terminal-only counts, terminal-before-reset metrics, invalid slip masks,100 cases per style, unchanged policy/RNG/optimizer in inference, paired manifest hashes, metrics quantiles and lexicographic best selection.
- [ ] Implement inference-only projection contract rather than training-downsize logic. Freeze completed-trial statistics, independent seed/state registry and validation/training split.
- [ ] Periodic full snapshots and independent best_validation/best_success_rate directories; no best_success label when success_count0. Immutable source and complete checkpoint provenance.
- [ ] Prepare bounded native runner that only starts after the tracked run reaches2048 naturally, own Isaac is absent and GPU3 is idle; exclusive lock and fail-closed identity checks. Do not kill any process.
- [ ] Run at least100 paired cases per grasp when the native slot becomes available, report actual results. Missing269/482/1328 weights remain explicitly unavailable.

## Later phases remain gated by user order

RewardR0–R5 implementation/experiments follow fixed-model regression, one factor at a time. Optional feedback/control follow proven sensor-source contracts; no20D append now. Teacher data require synchronized genuine RGB-D and same-state corrections. No long repaired PPO starts before action/native evaluation and diagnostic gates. Preserve original physics,0.2Nm and success30mm/5s/.5mm throughout. Acceptance is evidence-based; CPU tests are never a native grasp-success claim.
