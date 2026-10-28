# PPO force target without slew

User confirmed on 2026-10-25: PPO chooses each hand's target in 0.1–12 N and the controller adopts it immediately. This replaces the 2 N/s ramp; it does not prescribe a fixed 12 N or overwrite measured contact forces.

Use an explicit null force-rate field, preserving finite-rate historical profiles. Scalar, NumPy and CUDA controllers must agree. Preserve original closure-offset and joint-command rates, geometry, effort limits, real-contact readiness, force failure handling and one-second close-start reward. The unloading threshold follows the new target immediately; actual force response remains a physics result.

Resume only via an exact documented 12 N / 2 N/s to 12 N / immediate contract migration. Preserve policy, optimizer and counters; keep one Isaac instance. Verify target rises and falls in one step, inactive rows and resets, no fictitious grip readiness, CPU/CUDA parity, finite PPO updates and actual simulator limits. Do not equate target stepping with instantaneous measured 12 N or manufacturer calibration.

Approved design question: “PPO selects the target and executes immediately”; user answered yes. Implemented and verified according to the companion plan. No hardware actuation or Git publication requested.
