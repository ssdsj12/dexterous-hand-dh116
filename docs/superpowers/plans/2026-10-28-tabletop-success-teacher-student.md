# Tabletop successful teacher and PPO student implementation plan

> Execute in this session with subagent-driven-development for isolated units,
> then integrate and review. No commits, pushes or historical artifact deletion.

**Goal:** Obtain repeatable real grasp/lift success, imitate its executable
actions into the current actor, then optimize with the existing PPO task.

**Architecture:** A teacher runner branches after normal tensor setup and
before PPO construction. A success-only trajectory recorder feeds offline
actor imitation. A distinct actor-initialization option starts fresh PPO;
physical validation precedes optimization.

**Tech stack:** Python, torch, IsaacLab/PhysX, RSL-RL, existing pytest fixtures.

## Global constraints

- Same20D actions/156D observations, current task/physics/guards/rewards.
- Shared Z, six contact depths, per-hand force2–12N, existing small XY bounds.
- Actual lift success and three fresh-reset confirmations before qualification.
- One Isaac process; no unsafe checkpoint continuation or fake success labels.

## Task1: Qualified trajectory recording

Files: create `src/go2_pvcnn/control/dh116/tabletop_demonstrations.py` and
`tests/test_dh116_tabletop_demonstrations.py`.

Interface: `SuccessEpisodeRecorder(num_envs, observation_dim=156, action_dim=20)`;
`append(obs,actions,rewards,dones,extras)` uses `extras['success']` and
`extras['terminal_observation']`; `save(path,contract,qualification)` writes
only complete successful episode tensors with distinct episode IDs.

- [ ] RED: failures never enter successful output; success boundary correctly
  pairs pre-action observation/action and terminal observation; resets split
  episodes; nonfinite/incorrect dimensions rejected; empty success output rejected.
- [ ] Implement bounded per-environment buffers (maximum1000 control steps)
  and success-only publication with full contract/provenance.
- [ ] Run `bash scripts/python.sh -m pytest -q tests/test_dh116_tabletop_demonstrations.py`.

## Task2: Teacher search through the production environment

Files: create `src/go2_pvcnn/control/dh116/tabletop_success_teacher.py`,
`tests/test_dh116_success_teacher.py`; modify production trainer CLI and
`tabletop_tensor_training.py` at the existing setup/algorithm boundary.

Interface: `run_success_teacher(env,args,contract,obs)` and validated JSON cases
containing name,12 joint target offsets, contact depth,2 force targets,2x2XY.
Build normalized joint requests with existing `joint_actions`; force requests
with existing `force_action`; shared bounded Z correction uses valid measured
contact centers, never writes task targets or phases directly.

- [ ] RED: normalized bounds, shared Z, force range, missing-contact behavior,
  no success on lift entry, per-environment terminal handling and safe serialization.
- [ ] Integrate `--teacher-search-cases` / episode and control-step budgets,
  rejecting simultaneous resume/play/probe/actor initialization.
- [ ] Run focused tests, then actual small-batch coarse depth/force search.
- [ ] Refine failed-but-promising candidates using measured contact/load/gates;
  diagnose hard faults rather than weakening guards.
- [ ] Repeat any selected success for three fresh-reset trials and save proof.

## Task3: Imitation artifact and PPO actor initialization

Files: create `src/go2_pvcnn/control/dh116/tabletop_behavior_cloning.py`,
`scripts/train_dh116_tabletop_imitation.py`,
`tests/test_dh116_tabletop_behavior_cloning.py`; integrate initialization in
production trainer without using checkpoint resume.

Interface: fit the existing bounded actor against qualified successful20D
labels; artifact stores actor state, observation/action dimensions, dataset hash,
full contract and qualification. `load_actor_initialization(algorithm,path,contract)`
loads actor only and verifies the exact meaningful task contract.

- [ ] RED: reject empty/unqualified/incorrect-dimension/nonfinite datasets;
  reject physical/reward/controller contract drift; keep critic/optimizer/counters fresh.
- [ ] Train with whole-episode holdout and report action errors by channel.
- [ ] Verify actor-only loading and existing noise-bound compatibility.
- [ ] Run closed-loop student validation in real Isaac; require actual success.

## Task4: Controlled optimization and reporting

- [ ] Start4096-environment/1024-episode-per-environment PPO from validated actor.
- [ ] Confirm fresh counters/optimizer, unchanged rewards/physics and correct
  TensorBoard route. Record initial and post-update actual grasp/lift outcomes.
- [ ] Review changes and document exact tests, runs, successes and limitations.
  Do not mark complete if no qualifying teacher trajectory exists.
