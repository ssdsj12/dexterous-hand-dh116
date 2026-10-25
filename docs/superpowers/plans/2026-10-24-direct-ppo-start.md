# Direct PPO start

Goal: honor the user's request to begin learning without three full diagnostic
episode batches. Existing simulation startup checks and runtime guards remain.
No changes to physics, actions, rewards, or success criteria.

Implementation:
- Add opt-in `direct_training=False` to `run_vector_training` and expose
  `--direct-training` in the GPU entry point. Reject combination with probe-only.
- In direct mode, check seeded reset observations and reset safety information,
  but collect no admission episodes. Record `mode=direct_training`,
  `probes_skipped=true`, and empty batches; never claim full probes passed.
- Keep default admission, per-step safety callback, checkpoint and evaluation.
- Test actual SB3 updates with failed grasp episodes, reset safety rejection,
  training fault stop, and conflicting flags. Run the existing vector tests.
- Stop v62 with SIGINT (zero learned updates), preserve its artifacts, launch
  v63 with the same 512-env arguments plus direct-training. One Isaac only.
- Verify live PPO transitions and updates; update active pointer and docs.

Validation: five new tests first failed for the absent direct_training option;
after implementation all 128 vector lifecycle tests passed. Compile/diff checks
passed. v62 exited on SIGINT without learned updates. v63 PID27908 completed
contact detection in8 vector steps, with all1024 hands detected, then entered
training and collected3072 transitions. No optimizer update at that observation;
the first64-step rollout contains32768 aggregate transitions. Active pointer and
README/CHANGELOG updated. No claim of successful physical grasp or optimized policy.
