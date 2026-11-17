# Flexible R0 applicability v2 — 2026-11-13

## Outcome and limits

Base: `36e83f7e2bd663acf18d22ae109b1cc1f008a1a6`. This package removes a mathematical pre-grasp reward-domain dead end in an **independent opt-in diagnostic version**. It does not qualify a reward for PPO or change the frozen v1 task, coefficients, control/physical parameters, 371D/24D contracts or success criteria.

Original R0 requires material-anchor slip in every physical row, while its own slip routine returns unknown before bilateral grasp. Thus even otherwise complete pre-grasp observations cannot yield a finite diagnostic reward. The original characterization tests are retained unchanged.

## Implementation

- `src/go2_pvcnn/control/dh116/tabletop_deformable_r0_applicability.py`
  - `ApplicableR0Episode.reset`: optional measured reset approach distances must match environment, episode, reset, rod, physics/solver tick and timestamp exactly. A known distance needs an explicit source. No configured distance or first-step zero is substituted.
  - `step`: adds applicability separately from raw validity. Known pre-bilateral contact can make fixed-anchor slip **inapplicable**; its raw value stays null, valid=false, and only its contribution is zero if a coefficient was explicitly supplied. Any bilateral contact latches the requirement until reset, even if anchor data was never available. Unknown contact history also latches ambiguity; subsequent absence does not prove that contact never occurred.
  - `ApplicableR0ControlStep`: original clock, action, budget and terminal guards are inherited; independent aggregation separates measured samples, defined contributions, inapplicable samples and unknown applicability. Partial terminal windows and exact-once settlement are preserved. Recipe/source/config hashes differ from v1.
- `configs/dh116_tasks/flexible_r0_applicability_diagnostic_v2.json`: independent disabled-by-default diagnostic config. No production coefficients, training reward or native admission.
- `scripts/replay_dh116_native_parameter_r0.py`: consumes pinned existing native receipt and raw ledger, verifies all 17 radius/raw-derived parameter packets, and compares original/v2 R0 with identical observations. No native import, physics, reset owner execution or optimization.
- Three new test files cover all grasp pad profiles, source/identity refusal, missing history, contact loss/recovery, preserved NA, threshold boundaries, terminal/no-op, exact-once aggregation and actual saved ledger replay.
- Implementation plan: `docs/superpowers/plans/2026-11-13-flexible-r0-applicability.md`.

## Verification

**CPU: 131 distinct tests passed, 0 final failures.**
- 122 new/legacy reward, control, grasp and native-parameter protocol regressions, 166.66s.
  Log: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-applicability-regression-xkXYCO/pytest.log`.
- 9 read-only replay/provenance tests, 5.35s.
  Log: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-replay-tests-AxkXiS/pytest.log`.

Reproducible CPU commands (Ubuntu worktree, existing environment):
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_r0_applicability.py tests/test_dh116_r0_applicability_history.py tests/test_dh116_r0_reward_domain.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_control_step.py tests/test_dh116_native_parameter_endpoints.py tests/test_dh116_native_parameter_task.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_native_parameter_r0_replay.py
```

Red evidence retained:
- Missing-module red collection: `evidence/r0-applicability-red-4iu785/pytest.log`.
- Contact-history ambiguity regression initially failed (1 failed, 6 passed): `evidence/r0-history-red-7NT2Bw/pytest.log`. The first implementation incorrectly re-entered the proven pre-grasp domain after a missing-contact interval. Fixed by sticky ambiguity until reset; included in the 122 passing regressions.

## Read-only native evidence comparison

Historical receipt SHA: `f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`.
Historical ledger SHA: `1981d68067f6554e63a27b190b9eb55f8663501ebdc9daca4700c73929e4ba2a`.
Original run: 2 setup + 16 measured physical steps, 17 parameter snapshots, natural exit 0. **Not rerun in this package.**
Backend SHA: `2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec`.
Radius: 0.004999999888241291 m, actual solver host metadata, not real hardware measurement.
Actual pair samples: hand–rod 0; rod–table 7872. No successful grasp is inferred.

Both original and v2 replay exactly two 8-substep controls. For both controls:

| Term | Original | v2 |
|---|---|---|
| required_contact, contact_persistence, grasp_geometry, lift_progress, joint_limit | defined measured/computed zero | unchanged |
| external_support_violation | table pair establishes violation | unchanged |
| material_anchor_slip raw | NA, invalid | NA, invalid |
| material_anchor_slip contribution | unknown | 0, explicitly not applicable in known pre-grasp |
| approach_progress, hold_progress, instability | unknown | still unknown |

The blocked term count drops from four to three, **but diagnostic and training totals remain unavailable on these real samples**. No stable=true, approach distance, anchor or hand-table absence was fabricated.
Recorded comparison: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-actual-parameter-replay-kfwV9L/comparison.json`; `run.log` holds the command outcome. Re-run the script only with a new output path.
The 1..10 coefficient vector is the existing CPU fixture, not qualified training weights. Native R0 reset order remains unqualified.

## Remaining work / next package

1. Bind actual same-clock pad/rod approach geometry at reset and post-step; do not infer real contact from SDF geometry.
2. Define and validate the stability source against the original physical safeguards; absence of a guard error alone is not proof of stable grasp.
3. Continue actual dual-hand contact/anchor and hand-table source work. Slip stays required after any observed bilateral grasp.
4. Complete the remaining 371D source mapping and qualify production R0 coefficients before a new bounded PPO recipe.

Teacher Pinch/Tripod/Power admission remains **0/3**, optimizer updates **0**, new native steps **0**. R2 remains disabled. No historical checkpoint or success-pool changes. PR #2 remains Draft.
