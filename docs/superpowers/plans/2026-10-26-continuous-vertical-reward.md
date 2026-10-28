# Continuous vertical force reward

User approved continuous upward reward and downward penalty after reviewing the formula. Preserve physical grasp/lift/hold success and protection limits. Use writing-plans and test-driven-development workflow; approval is already given.

1. Add failing tests in tests/test_dh116_continuous_vertical_reward.py for positive reward beyond one second and five points, negative linear cost, inactive/terminal/NaN handling, selected reset and CPU/CUDA frequency independence.
2. Add explicit window/continuous mode to TensorTask and CLI. Keep historical window default for compatibility; update launch script to continuous. Continuous reward is 0.5*clamp(Fz,0,10)*dt on valid non-failing live substeps, downward remains -0.5*max(-Fz,0)*dt on finite live substeps including failure terminal. Never reward invalid failure states.
3. Version the continuous contract: no time window or episode cap, unchanged observation layout, existing bounded legacy clock retained for shape compatibility. Monitor labels must not describe continuous reward as expired. Old-window checkpoints rejected unless exactly compatible; do not silently migrate.
4. Run regression tests, independent review, then stop previous Isaac after preserving checkpoint. Start a new 4096-env run without resume/initialize-from as previously requested. Preserve existing run; no additional deletion requested this turn.
5. Verify actual first PPO updates, continuous contract, zero initial counters/Adam state and finite rewards/checkpoint. Keep read-only observer active; record actual findings in README/CHANGELOG. Do not claim lift success before evidence.
