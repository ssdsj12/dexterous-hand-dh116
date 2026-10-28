# Bounded policy noise implementation and validation plan

Goal: stop unbounded exploration growth observed in v121 without changing task rewards, physics, or learned actor/critic weights.

Evidence: iteration399 Gaussian std9.00–16.98 on normalized [-1,1] controls; minimum expected clipped fraction93.61%, entropy78.2 at398. Actor means are already tanh-bounded, but Gaussian entropy keeps increasing with log_std despite actuator clipping. v121 stopped at completed iteration400; no physical fatal stop.

Design: retain Gaussian raw samples and their likelihoods in RSL PPO. Project finite log_std to log([.05,1.0]) after every Adam minibatch and checkpoint load. Reset only projected noise-coordinate Adam moments; retain other state and step. This is an explicitly recorded exploration-policy migration, not an equivalent continuation. Keep clipping at the actuator boundary and log its observed fraction. Reject nonfinite noise. Alternatives: disabling entropy loses exploration support; changing to a squashed distribution requires new likelihood/entropy/KL integration and is out of this targeted fix.

Files: tabletop_policy_noise.py isolates projection; tabletop_tensor_training.py wires optimizer hook, exact migration, resume audit and telemetry; watch_dh116_tensor_ppo.py validates new telemetry; tests cover isolated entropy growth, resume state, rejection and real RSL updates.

- [x] Reproduce entropy-only growth with a failing test (std1.0941 after one update).
- [x] Implement projection and exact one-way contract migration.
- [x] Test lower/upper bounds, nonfinite rejection, real RSL optimizer and resume.
- [x] Independently review the current diff, resolve findings.
- [x] Resume real v121 iteration400 into a new run, no diagnostic overrides.
- [x] Confirm finite updated parameters, bounded noise, lower observed clipping, unchanged physics. Monitor ongoing lift performance without claiming success.

No commit or upload requested. Work occurs in the existing training checkout; one Isaac process only. User authorized correction of discovered issues; no further permission gate is added.

Validation: 90 related tests passed; independent reviewer23 passed. v122 resumed400→405, noise std.9316–1, observed action clipping.51399. 3 bounded mimic-error resets, no global fatal stop, success0. Checkpoint403 finite and12 network tensors updated. Background observer running; no guarantee of autonomous repair outside an active agent turn.
