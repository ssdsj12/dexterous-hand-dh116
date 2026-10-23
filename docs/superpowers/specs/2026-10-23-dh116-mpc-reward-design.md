# DH116 MPC-Aligned Reward Design

## Goal

Add dense, auditable rewards to the existing hand-only rigid-wire probe while retaining MPC joint targets as the teacher reference and keeping physics qualification independent.

## Design

Implement a pure NumPy reward function with validated finite inputs and per-term diagnostics. Normalize MPC action error by 0.15 rad and action delta by 0.10 rad; use exponential tracking reward (weight 1.0) and action-change penalty (weight 0.05). For the existing grasp/lift phases, add bounded moderate-contact shaping (weight 0.5, 2-8 N target band), soft force penalty above 15 N, lift-reference tracking with 1 cm scale (weight 1.0), and a one-time-equivalent lift completion bonus of +1 at 2 cm with valid contact. The collector uses the existing MPC progress and measured proxy state to calculate these terms and stores total reward plus reward-term diagnostics in the run summary.

Approach pose, material-point slip, and deformable-wire strain terms remain disabled because the current collector does not measure those signals. No new replay schema or phases are introduced. Existing hard abort and qualification rules (`physics_valid`, `qualify_proxy_lift`, 30 N abort cap and 2 cm proxy-lift threshold) are unchanged and cannot be overridden by reward.

## Validation

Unit tests cover MPC-perfect tracking, worsening tracking, smoothing, bounded contact shaping, force penalty, lift progress/completion, invalid inputs, and unchanged qualification gates. Collector tests verify reward terms are logged while `mpc_action` remains the reference. No reward result is presented as deformable-wire or real-grasp success.
