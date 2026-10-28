# Force progress and lift milestones implementation plan

**Goal:** Add approved dense direction shaping and actual lift milestones while preserving PPO progress.
**Architecture:** New tabletop_grasp_shaping.py owns force filter/potential/dwell/paid state. TensorTask integrates per-physics milestone updates, once-per-control shaping, reset and observations. TensorEnv supplies runtime mg and counts existing lift gate masks. Trainer contracts, migration, CLI, telemetry and default launcher opt in.
**Tech Stack:** PyTorch CUDA, existing Isaac tensor environment and RSL PPO.

## Constraints
Only one Isaac instance; preserve logs/checkpoints and current weights; no changed force/physics/lift safety thresholds. gamma=0.995/control step, tau=.05 s, Phi=2*tanh(Fz/mg filtered), milestones3mm/+2 and10mm/+3 each.1s. All terminal next potentials0; no reset force in prior terminal reward. New observations5, new weights/moments0.

- [x] Write failing CPU/CUDA shaping tests: less-negative force improvement, negative regression, discounted telescoping with terminal and rollout boundaries, no inactive/invalid credit, 60/480Hz filter equivalence, exactly-once milestones, dwell interruption and selected reset.
- [x] Implement GraspShaping(begin/update/end/reset/observe), integrate TensorTask consume and end_step before snapshots, runtime weight validation, five observation fields.
- [x] Add exact one-way checkpoint contract and151->156 zero-column/Adam migration; CLI --grasp-shaping requires current approved reward flags. Assert algorithm gamma matches contract.
- [x] Add existing lift gate telemetry with conditional fractions; test masks and no control effect, runtime force event curves retained.
- [x] Run regression tests including existing force/phase/task/env/terminal/migration tests; read-only independent review.
- [x] Stop verified active trainer after preserving latest finite atomic checkpoint; resume new single v133 run, verify counts/Adam/observation migration, new reward terms/gate telemetry and finite updates.
- [x] Update README/CHANGELOG with actual evidence, report limits including whether any physical lift success occurred.
