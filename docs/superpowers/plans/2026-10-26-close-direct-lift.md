# Close directly to lift

User explicitly requests removing hold. Keep close readiness at each hand >=1 N and <=15.02 N, object speed <0.02 m/s, slip <=3 mm, continuous 0.2 seconds. Transition directly from phase 1 to phase 3; retain phase indices and 151 observation dimensions for policy compatibility. Legacy mode remains available without --skip-hold.

Keep upward reward in close and later stages. Activate downward penalty on first substep after close-to-lift, preserve latch until reset. Keep all lift motion guards, real height/stability success criterion and failure conditions. Do not interpret phase admission as physical lift success.

1. Add CPU/CUDA regression tests for timing, instability reset, direct transition, force latch, once-only grasp bonus and reset; exact checkpoint migration must reject unrelated changes and reversal.
2. Implement explicit --skip-hold task option, revised readiness/reward metadata and precise one-way checkpoint migration. Enable in default launcher.
3. Run relevant regression tests and read-only review. Preserve current completed PPO checkpoint and stop old Isaac before starting new run.
4. Resume policy/Adam/counters with new option, verify actual training updates, contracts and telemetry, update README/CHANGELOG. No new zero-start or deletion requested.
