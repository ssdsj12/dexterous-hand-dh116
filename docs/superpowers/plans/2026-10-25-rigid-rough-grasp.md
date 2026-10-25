# Rigid rough-rod grasp training

User requested switching back to a rigid object and checking hand friction. Existing authorization covers material configuration and starting PPO.

Use the existing solid cylinder (radius 8 mm, length 0.5 m, mass 0.12 kg) at the original pose. This is a rough-rod proxy, not a scanned/calibrated branch. Use explicit hand static/dynamic friction 1.0/0.8 and object 1.5/1.2, zero restitution, max friction/min restitution combination. Coefficients are training assumptions, not manufacturer or measured values. Preserve existing default contracts. Keep actuator, reward, action and success rules unchanged for this comparison.

- [x] Cancel flexible GUI auto-resume; save existing checkpoint and stop GUI.
- [x] Add profile/binding/runtime audit regression tests, observe failures.
- [x] Add explicit optional hand material and rough rod material; verify bindings, combine modes and cooked shape properties.
- [x] Run material/regression tests and single Isaac startup for new physics.
- [x] Start fresh 1024-env PPO, each env 1024 episodes, original CPU0/thread constraint; confirm live rollout and runtime safety. Retain prior checkpoints.
- [x] Install read-only observer and report material values, training status, limitations.

Validation: 196 tests passed (material, startup preflight, initialization, playback, expansion, vector training). Real r2 PhysX startup verified all 1024 objects at static/dynamic 1.5/1.2, both hands at 1.0/0.8, 17 collision shapes per hand, max/min combination modes. Hand meshes are instance proxies; bind physics purpose at their writable instance roots and audit the resolved collider binding.

Follow-up authorized correction: direct rigid PPO previously required bilateral zero-action contact before training. Direct mode now runs two finite/safe steps and permits contact failure as learnable experience, consistent with prior user approval. Explicit contact diagnostic mode retains its 120-step contact requirement. New tests reject physical and NaN contact failures. No success, reward, drive or action-limit change.

Fresh launch r3 replaces the deliberately stopped r2 preflight. First rollout/optimization status must be reported separately from startup validation.

Launch verified in r3: live PPO rollout with nonzero samples, updates=0 (first optimization pending), no safety stop. Observer installed; no claim of successful grasp or first optimizer update.
