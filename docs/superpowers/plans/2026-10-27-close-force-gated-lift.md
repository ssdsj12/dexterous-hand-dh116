# Close Force Gated Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make close continuously reject sustained downward hand-to-object force, reposition the grasp downward before reset, and admit lift only when the aggregate signed vertical force is nonnegative.

**Architecture:** Keep force measurement in `TensorTask.consume()` and use the existing `ForceDirectionReposition` state machine during phase 1. Add task-local downward impulse accounting and a close failure code path; keep physics reset centralized in `TensorEnv.step()`. Tighten lift admission in the task state transition, and raise only the coordinated close command slew while preserving coupling, pressure, joint, and transport governors.

**Tech Stack:** Python, PyTorch tensors, pytest, IsaacLab tensor task contracts.

## Global Constraints

- Use aggregate signed hand-to-object vertical force; positive means world-up.
- Preserve the existing `-0.15 N` reposition trigger, 1 mm correction, 2 mm maximum correction, and bounded release/reclose flow.
- Reset when the close phase exceeds `4 s`, when accumulated close downward impulse reaches `2.0 N·s`, or when reposition attempts/timeout are exhausted.
- Require `vertical_force_n >= 0` before close-to-lift admission.
- Increase coordinated close command slew from `0.05 rad/s` to `0.20 rad/s`.
- Keep the existing physical guards and do not alter the current checkpoint in place.

---

### Task 1: Add failing task-state tests

**Files:**
- Modify: `tests/test_dh116_close_downward_penalty.py`
- Modify: `tests/test_dh116_skip_hold.py`

**Interfaces:**
- Consumes: `TensorTask.consume`, `TensorTask.reset`, and the existing `loads()` fixture.
- Produces: failing regression tests for impulse reset and nonnegative lift admission.

- [x] **Step 1: Write the failing tests**

Add tests that construct the existing measured-load direct-lift task and assert:

```python
def test_close_downward_impulse_marks_episode_failed_before_lift():
    t = task(); t.begin_step(torch.zeros((3, 20))); m = loads(t, 4.)
    dt = 1 / 480
    downward = torch.full((3,), -1.0)
    for _ in range(97):
        t.consume(m, ~t.done, dt, vertical_force_n=downward)
    assert t.failure[0].item() != 0
    assert t.phase[0].item() == 1


def test_downward_force_cannot_admit_direct_lift():
    t = task(); t.begin_step(torch.zeros((2, 20))); m = loads(t, 4.)
    dt = 1 / 480
    for _ in range(96):
        t.consume(m, ~t.done, dt, vertical_force_n=torch.tensor([-1.0, 1.0]))
    assert t.phase.tolist() == [1, 3]
```

The first test uses `961 / 480 ≈ 2.002 s`, which exceeds `2.0 N·s` at `-1 N`; the second verifies the signed transition gate independently of the reset threshold.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_dh116_close_downward_penalty.py::test_close_downward_impulse_marks_episode_failed_before_lift tests/test_dh116_skip_hold.py::test_downward_force_cannot_admit_direct_lift
```

Expected: FAIL because the task currently has no close impulse failure state and admits lift without checking `vertical_force_n`.

- [x] **Step 3: Commit the failing tests**

```bash
git add tests/test_dh116_close_downward_penalty.py tests/test_dh116_skip_hold.py
git commit -m "test: specify close downward force reset and lift gate"
```

### Task 2: Implement close force accounting and lift admission

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py:8-130,340-420`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_env.py:100-150`
- Modify: `tests/test_dh116_close_downward_penalty.py`

**Interfaces:**
- Consumes: `vertical_force_n` passed from `TensorEnv.step()` and the existing `palm_reposition` state.
- Produces: `TensorTask.close_downward_impulse`, `CLOSE_DOWNWARD_PENALTY_CONTRACT` with the `2.0 N·s` limit and `4 s` close timeout, and a failure/reset path recognized by `TensorEnv`.

- [x] **Step 1: Add the minimum implementation**

Add a task tensor `close_downward_impulse` initialized and reset to zero. During each valid phase-1 contact sample, add `clamp(-vertical_force_n, min=0) * dt`. Mark `grasp_not_established` when the accumulated value reaches `2.0`. Mark the same failure when the close phase exceeds `4 s`. Add `vertical_force_n >= 0` to the `ready` mask before dwell/transition. Treat an active, aborted, or timed-out repositioner as not ready and mark the same failure when its correction budget is exhausted. Keep the existing one-time event reward for telemetry, but make continuous close force cost apply while `previous_phase < 3` as well.

The task contract should expose:

```python
CLOSE_DOWNWARD_PENALTY_CONTRACT.update(
    version='close_downward_impulse_reset_v3',
    impulse_limit_n_s=2.0,
    close_timeout_s=4.0,
    lift_admission='vertical_force_n >= 0',
)
```

Do not reset physics from `TensorTask`; set the failure code and let `TensorEnv.step()` perform its existing reset on the next step.

- [x] **Step 2: Run the focused tests and verify GREEN**

Run the same two pytest targets from Task 1. Expected: PASS.

- [x] **Step 3: Add reposition failure coverage**

Extend `tests/test_dh116_close_downward_penalty.py` with a case that sets `t.palm_reposition.aborted[:] = True`, consumes a close sample, and asserts failure is set without entering lift.

- [x] **Step 4: Run the extended focused tests**

```bash
pytest -q tests/test_dh116_close_downward_penalty.py tests/test_dh116_skip_hold.py
```

Expected: PASS.

- [x] **Step 5: Commit the task behavior**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_task.py src/go2_pvcnn/control/dh116/tabletop_tensor_env.py tests/test_dh116_close_downward_penalty.py tests/test_dh116_skip_hold.py
git commit -m "feat: gate lift on nonnegative close force"
```

### Task 3: Increase close slew while retaining governors

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py:12,132-150`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py:250-270`
- Modify: `tests/test_dh116_loaded_palm_motion.py`

**Interfaces:**
- Consumes: existing target, coupling, pressure, and envelope calculations.
- Produces: a `0.20 rad/s` close command slew with the same lift and reposition behavior.

- [x] **Step 1: Write the failing slew test**

Add a test that calls coordinated control with a target one radian away at `dt=1/480` and asserts the command delta is `0.20/480`, while the disabled reposition path remains byte-for-byte equal to the baseline for all other tensors.

- [x] **Step 2: Run the test and verify RED**

```bash
pytest -q tests/test_dh116_loaded_palm_motion.py -k slew
```

Expected: FAIL because the current delta is capped at `0.05/480`.

- [x] **Step 3: Implement the single slew change**

Introduce one shared constant, `CLOSE_COMMAND_RATE_RAD_S = 0.20`, and replace both coordinated and legacy tensor command caps with `CLOSE_COMMAND_RATE_RAD_S * dt`. Leave reposition release behavior, pressure relief, coupling limits, and mechanical envelopes unchanged.

- [x] **Step 4: Run focused controller tests**

```bash
pytest -q tests/test_dh116_loaded_palm_motion.py tests/test_dh116_close_downward_penalty.py
```

Expected: PASS.

- [x] **Step 5: Commit the slew change**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py src/go2_pvcnn/control/dh116/tabletop_tensor_task.py tests/test_dh116_loaded_palm_motion.py
git commit -m "feat: speed up coordinated close slew"
```

### Task 4: Update contracts and run full validation

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_training.py:350-395`
- Modify: `tests/test_dh116_tensor_training.py`
- Modify: `docs/superpowers/specs/2026-10-27-close-force-gated-lift-design.md`

**Interfaces:**
- Consumes: the new task and controller contracts.
- Produces: checkpoint compatibility rejection for old behavior and reproducible validation evidence.

- [x] **Step 1: Write the failing contract test**

Assert that a checkpoint contract with the old close penalty version or old command rate is rejected as incompatible with the new runtime contract.

- [x] **Step 2: Run the contract test and verify RED**

```bash
pytest -q tests/test_dh116_tensor_training.py -k 'close or contract'
```

Expected: FAIL because the new contract fields are not yet included in checkpoint comparison.

- [x] **Step 3: Add contract fields and migration rejection**

Include the impulse reset limit, lift force gate, and close command rate in the saved contract. Do not migrate the current checkpoint automatically; require a fresh training run for the changed behavior.

- [x] **Step 4: Run the full relevant suite**

```bash
pytest -q tests/test_dh116_close_downward_penalty.py tests/test_dh116_skip_hold.py tests/test_dh116_loaded_palm_motion.py tests/test_dh116_tensor_training.py tests/test_dh116_vertical_reward.py tests/test_dh116_continuous_vertical_reward.py
```

Expected: PASS.

- [x] **Step 5: Commit the contract and validation updates**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_training.py tests/test_dh116_tensor_training.py docs/superpowers/specs/2026-10-27-close-force-gated-lift-design.md
git commit -m "feat: version close force gated lift contract"
```
