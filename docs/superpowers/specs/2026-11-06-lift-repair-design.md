# DH116 isolated lift-repair design — 2026-11-06

## Authority and boundaries

Implement the user's T0–T8 document in its stated order, with the subsequent hardware clarification taking priority. Current v2 training must remain untouched and be allowed to finish its absolute2048-update budget. Use branch `codex/dh116-lift-repair-20261106` from `b42f1aa`, isolated server stage `/mnt/mydisk/xk/projects/dh116-lift-repair-20261106`, independent outputs and checkpoints. Keep3844D actor/371D critic. No new feedback features are enabled until actual hardware and driver provenance are verified; optional20D metadata is separate from actor inputs.

The existing raw Gaussian plus deterministic clamp admits a self-consistent latent-action PPO interpretation. The requested squashed policy is a versioned distribution/regularization upgrade, not evidence that the old importance ratios are automatically invalid. A height peak without qualified contact is not necessarily reinforced by the qualified lift reward.

## T0 evidence

Archived original Git source and user request as read-only local copies. Frozen complete checkpoint1439: `d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9`, 47644672 transitions, 37680 episodes, successes0; all80 checkpoint tensors finite on CPU. Complete policy-chain logs contain5+124+1310 rows. The original run and source remain untouched. Baseline targeted CPU suite:81 passed. The hardware audit report precedes production implementation.

Actual visual weights269/482/1328 were not found and cannot be reconstructed from logs. Available comparable fixed-camera snapshots are129 and1439; snapshot5 has a different camera/count contract and is secondary context. Never label a surrogate or renamed current checkpoint as a missing historical model.

## T1 distribution, storage and probability

Selected architecture: an opt-in v3 squashed Gaussian alongside the unchanged v2 implementation. The new actor outputs unbounded latent mean; it has only one output squash in the distribution. Deterministic inference is tanh(mean), not E[tanh(Z)]. Removing the old parameter-free final Tanh preserves parameter keys but requires an explicit new model/distribution version.

Sampling returns exact latent z, bounded action tanh(z), transformed old logp, and latent mean/std. CPU float32 rollout storage retains both latent_actions and actions, including pin-memory and complete indexing through flatten/permutation/logical minibatches/microbatches. Environment commands and previous-action history use the exact stored bounded action; physical safety projections remain unchanged environment dynamics and are separately diagnosed.

Use stable logdet `2*(log(2)-z-softplus(-2*z))`; never recover z through atanh(action). Reject non-finite latent/parameters/logp before physics. IEEE tanh may round to exact±1 for large finite latent; bounded actions remain in[-1,1], and latent-based logp stays finite.

Use current-distribution reparameterized bounded-action entropy: Gaussian entropy plus expected logdet. Generate one fixed entropy epsilon array per logical minibatch so microbatch/full-batch tests share sample-indexed noise. KL uses the exact latent Normal KL, which equals the KL under the common invertible tanh transformation. Record both analytic KL and empirical approx-KL, with finite guards.

Retain five epochs/eight logical minibatches, weighted microbatch gradient accumulation, one Adam step and one gradient clip per logical minibatch. Keep existing GAE, advantage normalization, physical action scales and actuator limits. Do not silently change v2 behavior or noise semantics.

Log rollout raw max, saturation threshold/rates, per-dimension empirical bounded action mean/std and latent mean/std. Log frozen-policy old/new logp consistency, update old/new logp summaries, approx-KL, analytic KL, clip fraction, bounded and latent entropy, and pre-clip gradient norm. JSON and TensorBoard identify units, phase, weighting and action groups.

New version/contract must reject v2 exact resume before writing output metadata. Any v2→v3 warm start is an explicit derived checkpoint in a new directory, retaining provenance and stating that the distribution changed. State_dict strict loading alone is insufficient. Current outputs/checkpoints are never overwritten.

## T2 native evaluation and checkpoint management

Add a visual-only deterministic evaluation adapter using real camera.sample, SensorHistory and sensor-only observation packing. No optimizer, no noisy policy sampling, no privileged actor path. Count only each trial's first terminal and freeze its statistics thereafter. Use a hashed fixed manifest with at least100 trials per grasp and identical initial states/physics/camera across paired models. Evaluation projection from1024 training environments is explicit inference-only metadata, not a training-resume downsize.

Report per-trial lift P50/P75/P90/P95/max,30mm achievement,5s qualified hold, fixed-anchor drift with validity, bilateral contact retention, actuator saturation and native successes. Preserve terminal-before-reset diagnostics. Best selection uses an independent validation set and lexicographic native success/hold/height achievement/slip/distribution/action stability; no-success results cannot be named best_success.

Native execution is gated on the existing run's natural completion, no active own Isaac, an idle GPU3 and an exclusive experiment lock. Never kill a process to gain a slot, never run a second Isaac. A queued validator must fail closed if the tracked run terminates unexpectedly or its identity changes. Historical missing checkpoints remain an explicit limitation.

## Later stages and acceptance gates

T3: retain R0 support_deficit byte-for-byte; add separate R1 height-hold, R2 upward progress, R3 decline penalty, R4 validated hold+up, R5 only combinations supported by single-factor results. Same seeds, initial states, starting weights and PPO budgets. Positive terms require valid stable contact/support/slip/velocity; use control-window deadbands and actual episode-reference heights. Native success thresholds and physical parameters remain unchanged.

T4: optional20D source/unit/rate/validity/timestamp/epoch interface only. Hardware report documents unknown channels; no native20D tensor is appended to3844D actor. A future measured channel requires a new observation contract and proven real driver, not a simulator tag renamed as a sensor.

T5: audit existing teacher-only SupportServo before an independent deployable-feedback bimanual controller. Do not install GT-based servo into the visual student. Preserve0.2Nm and all original limits; no external force on the object. Any optional controller has a separate actuator/dynamics contract and requires native fixed-pose validation.

T6: new synchronized RGB-D/calibration/encoders/FK/action/physical-time expert records. Privileged labels remain segregated from student features. DAgger correction must start from the same actual student state and pass native correction checks; old demonstrations cannot silently migrate into v3 actions.

T7/T8: start new training only after preceding diagnostics/native gates. Preserve all original models. Validate on at least100 independent cases per grasp, separately report deterministic/stochastic and unseen-state results. Do not claim success from CPU tests or a single height peak.
