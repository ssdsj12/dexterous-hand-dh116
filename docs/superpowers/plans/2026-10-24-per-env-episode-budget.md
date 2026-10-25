# Per-environment episode budget

User confirmed512 environments,1024 completed training episodes per environment.
Keep64-step PPO rollouts, current physics/reward/actions, direct training and all
runtime fault guards. Failure episodes count; diagnostic/evaluation episodes do not.
Each environment must reach the goal; faster environments may exceed it while
waiting for slower ones. Complete the final rollout/update rather than discard
the terminal training data. Actual successes remain a separate metric.

Tasks:
1. Add integration tests to tests/test_dh116_tabletop_vec_training.py: variable
   episode lengths reach every-env quotas, rollout updates complete, runtime fault
   still stops, resume preserves completed counts, old-checkpoint migration rejects
   mismatched progress, invalid goals rejected.
2. Add episodes_per_env and optional resume_episode_progress to vector training.
   With a goal, call model.learn one full rollout at a time until all counters reach
   the target. Default timesteps mode remains unchanged. Expose goal/min/max/count
   in monitor/report. Save per-env statistics in checkpoint metadata; migrate old
   metadata using an explicit source monitor with identical timestep/update counts.
3. Let checkpoint loading accept the requested device so GPU resume remains CUDA.
   GPU CLI adds --episodes-per-env mutually exclusive with explicit --timesteps,
   and --resume-episode-progress for the existing v63 checkpoint.
4. Run vector and checkpoint lifecycle tests, compile/check diffs. Save v63 via
   SIGINT, confirm exit and saved model/statistics, then launch v64 with512 envs,
   episodes-per-env1024, resume checkpoint+progress, same64-step rollout and params.
5. Verify restored counters, CUDA, live training, one Isaac process; update active
   pointer and docs. Preserve all source runs; no commits/uploads.

Executed: eight initial budget tests failed for the absent interface, then all161
vector/scalar training and checkpoint tests passed (including two added resume
checks). Compilation and diff checks passed. v63 stopped after an update at229888
samples/31 optimization epochs, with one completed episode per environment. v64
PID31226 restored CUDA, n_steps64, all512 episode counters, and target1024; fresh
live sampling reached230912 cumulative transitions with no hard fault. Startup
contact checks passed. Active pointer and docs now reference v64. No claim of
grasp success; normal training-failure episodes count toward the requested budget.
