# Flexible PPO Environment Capacity Implementation Plan

> **For agentic workers:** Use isolated file ownership for checkpoint migration and capacity runner; root alone runs Isaac.

**Goal:** Increase flexible PPO parallel environments to the largest useful measured batch with VRAM headroom while preserving learned policy and optimizer.
**Architecture:** Explicit strict environment-expansion resume accepts only num_envs growth; original per-env statistics retained as a prefix, new slots start with zero history. Existing single-Isaac entry provides startup-only capacity measurements; test 1024/2048/4096 serially and choose based on memory and sample throughput, not allocation alone.
**Tech Stack:** IsaacLab/PhysX, batched Warp Cosserat, SB3 PPO, Torch, Python.

## Global Constraints
- One Isaac process. CPU0, one numeric thread.
- Keep 20D actions,136D observations, rewards, force control, original wire/hand poses, physics480Hz, policy60Hz, rod dt1e-5 and PPO n_steps64 unchanged.
- User authorizes expanded environments. Preserve old runs/checkpoints; no commits/push.
- Reserve approximately1.5GiB GPU memory; never use an OOM as a capacity target.
- Keep1024 completed episodes per environment. New slots have zero historical episodes; partial current episodes restart after migration.

### Task1: Strict checkpoint expansion (isolated agent)
Files: tabletop_training.py, tabletop_vec_training.py, dedicated checkpoint-expansion tests.
- [x] Reproduce tuple/list contract mismatch and default rejection of num_envs change.
- [x] Add canonical_contract(value), load_checkpoint(...,allow_env_expansion=False), run_vector_training(...,allow_env_expansion=False).
- [x] Explicit expansion permits only increasing num_envs with all physical/observation/action/reward fields equal; combine with reward-change flag rejected.
- [x] SB3 restores policy,optimizer,step/update counters with new VecEnv/rollout allocation. Preserve original statistics prefix; initialize added rows zero. Report expansion provenance and truthful per-env historical step counts.
- [x] Test2→4 environments, identical policy/optimizer before first new update, preserved old episode counts/new zeros; reject changed rewards/physics/shrink/malformed metadata. Run relevant existing CPU tests.

### Task2: Capacity entry and reporting (root)
Files: scripts/train_dh116_tabletop_gpu_ppo.py and new pure capacity helper/tests if needed.
- [x] Allow1..4096 environments and explicit --allow-env-expansion with --resume only.
- [x] Canonicalize contract before comparisons; forward expansion flag.
- [x] Startup-only run records steady bounded action throughput, per-step timing, GPU memory (global andTorch), reset/readback correctness, optional policy/optimizer migration verification.
- [x] Validate original local position precision at expanded grid scales; do not move seed geometry.

### Task3: Execute and verify (root)
- [x] Wait for current second rollout optimizer update, SIGINT/save; verify ZIP and metadata.
- [x] Serial actualIsaac1024/2048 startup-only capacity runs completed and exited0.4096 deliberately skipped after2048 showed no throughput gain; preserve learning feedback latency.
- [x] Compare measured throughput and memory headroom; launch expanded PPO from saved checkpoint using explicit migration, no weights-only restart.
- [x] Verify preserved counters/optimizer/policy, correct newnumenv, advancing samples/no protection. Update active pointers,README,CHANGELOG with actual evidence and remaining limits.

## Actual selection (2026-10-24)
- Source v66 safely saved at67,072 transitions/10 optimization epochs. Checkpoint ZIP and progress agree.
-1024:53.7038 transitions/s,3.606GiB peak sampled total-device use,3.510s full reset.
-2048:53.6477 transitions/s,4.079GiB peak sampled total-device use,7.252s full reset.
- Both preserve original geometry and reset isolation, load actual source policy/optimizer, and finish with no safety/runtime-buffer error. These are short3-step policy-inference benchmarks after2 startup steps; not sustained training/optimization benchmarks.
- Choose1024:2048 offers no measured throughput benefit and doubles64-step rollout latency. Do not fill remaining memory merely for allocation.4096 not tested; do not claim a physical memory ceiling.
- Fixed quadratic reset scheduling with batched selected resets and one-use backend reset consumption; no physics tick during reset.192 migration/lifecycle/reward tests plus33 reset/vector/task tests passed; one expected existing overflow protection warning.
- Actual source checkpoint separately verified on CPU shape-only1024 VecEnv: exact policy/optimizer equality, finite weights, [64,1024,136] rollout buffer; counters67,072/10 preserved.
- Actual run and evidence: runs/tabletop_flexible_20261024_v67_env1024_eps1024 and runs/flexible_env_capacity_20261024/selection.json. Each environment still targets1024 completed episodes. New environment histories begin atzero, unfinished source episodes restart.
