# Tabletop PPO readiness implementation plan

> Execute the user-approved scope continuously; use isolated task ownership and
> an independent final code review. Keep changes in the existing 10-23 checkout
> so local simulation assets and user run paths remain available.

**Goal:** Make the three-stage residual PPO lifecycle testable and truthfully gated.

**Architecture:** Pure task/control helpers, a simulator-independent training
lifecycle, and an Isaac adapter in the existing tabletop script.

**Global constraints:** single core/thread; one Isaac instance; rigid proxy;
12D residuals; original physical success/safety thresholds; no push.

## Task 1 — task and control contracts

- [x] Add regression tests to `tests/test_dh116_tabletop_readiness.py` for the
  real 10.08 mm contact stop, clearance-only stop, nominal-reference drift,
  12D finite actions, negative idle reward and stage-vs-grasp completion.
- [x] Run these tests red using `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_tabletop_readiness.py -q`.
- [x] Implement helpers in `tabletop_task.py` and `tabletop_control.py`; retain
  old reward API and baseline gate for existing consumers.
- [x] Run new tests and `tests/test_dh116_tabletop_task.py` green.

## Task 2 — training lifecycle

- [x] Add behavioral tests in `tests/test_dh116_tabletop_training.py`: reject
  unsafe/missing probe evidence; permit safe task failures; block unreached
  curriculum phases; validate checkpoint contract; save/load real SB3 PPO on
  a small explicitly synthetic Gym environment.
- [x] Implement `tabletop_training.py`: admission, episode evaluation,
  curriculum progression, CSV/progress diagnostics, periodic/resumable
  checkpoints, independent full-sequence assessment.
- [x] Verify tests before connecting Isaac.

## Task 3 — Isaac adapter

- [x] Connect measured approach transition, independent nominal references,
  expanded observations, per-substep phase/control updates and finite failure
  handling in `scripts/train_dh116_tabletop_ppo.py`.
- [x] Add CLI stage/curriculum, checkpoint/resume, evaluation and probe options.
- [x] Record stage success separately from grasp success; propagate failures
  to a nonzero process exit. Preserve existing preview options.
- [x] Run focused tests and compile checks; perform serial real Isaac probe,
  then short PPO training only if current-stage admission passes.

## Task 4 — review and evidence

- [x] Independent review of complete diff, fix substantive findings.
- [x] Record actual commands/results/remaining physics limitations in README,
  CHANGELOG and training-readiness documentation. No historical test numbers
  may be presented as current verification.

## Completion evidence

Focused suite: 101 passed, 1 skipped. Independent review has no remaining material findings.
Real filtered v5/v6: 640 cumulative samples, 25 optimizer epochs, approach 5/5, close 0/5;
no lift promotion. Unsafe intermediate runs retained. Work remains local/uncommitted.
The software plan is implemented; physical grasp mastery remains incomplete.
