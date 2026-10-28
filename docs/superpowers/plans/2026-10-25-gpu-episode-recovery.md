# Guarded GPU episode recovery

Goal: let PPO receive finite failed transitions and continue other environments after a verified isolated reset. The user authorized fixes on observed failures.

Evidence: v81 stopped at 241664 transitions, one PPO update, env1056 excess_force=30.7225N. Callback stops the whole rollout, so its terminal penalty is not consumed by PPO. This plan fixes failure handling, not the underlying contact-force cause.

- [ ] Reproduce handling in CPU integration tests: preserve negative reward, terminal observation, failure; reset one slot; continue PPO updates; evaluate with strict failure semantics.
- [ ] Add opt-in GPU substep recovery for 30<force<45N with all joint guards safe, or .02<mimic<.03rad with force<=30N. These are conservative recovery eligibility bounds, NOT changed episode thresholds or hardware ratings. Simultaneous guard violations remain fatal.
- [ ] Independently inspect all measured joints/velocities and geometry; reset physical state immediately on terminal substep; hold reset targets for remaining shared substeps and check each. Zero remaining substeps requires first-next-step validation. Never call sim.step during reset.
- [ ] Version training contract; retain strict checkpoint compatibility, all physical thresholds and evaluation. Record recovered failures separately from successes. Use fresh policy for new handling contract, preserve v81 checkpoint.
- [ ] Run regression tests and replay the recorded v80 physical failure prefix with one Isaac instance; verify terminal failure and post-reset state before starting long PPO.
- [ ] Start 4096-env close/hold/lift PPO and observe learning across recovered failures; retain monitoring and truthful status.
