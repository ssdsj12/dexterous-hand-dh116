# PPO Synchronized-Z Force-Phase Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tensor PPO close exploration use one synchronized world-Z height for both hands, allow only a ±0.5 mm low-weight XY refinement, use normal-only force before lift and normal-plus-friction force after lift, and map each force action to 2--12 N.

**Architecture:** Keep the existing tensor task and coordinated controller as the single control path. Add explicit constants and contract fields for the synchronized palm bounds and force range, pass separate close and lift vertical-force channels from the tensor world, and keep close reward/penalty calculations on the existing normal-only channel. Lift stability and physical success consume the combined channel only after phase 3.

**Tech Stack:** Python, PyTorch tensor task/world, Isaac/PhysX contact reports, pytest, JSON checkpoint contracts.

## Global Constraints

- Both hands share one policy Z command during close; after lift, both hands keep the common lift offset.
- Close force is the signed sum of hand-to-object normal Z components only; friction is excluded from close gates, rewards, and penalties.
- Lift force is the signed sum of normal Z plus friction Z and is used only after lift admission for lift stability and physical success.
- Each hand force target maps exactly to `[2.0, 12.0]` N; the nominal total range is `[4.0, 24.0]` N.
- XY palm exploration is capped at ±0.5 mm and its action/regularization weight is 25% of the Z exploration weight.
- Existing close reward, downward penalty, four-second close timeout, safety guards, and reset semantics remain unchanged.
- Checkpoints with the old 0.1--12 N force contract are rejected for ordinary resume; explicit weights-only migration must record the new contract.

---

### Task 1: Add synchronized palm exploration bounds and low-weight XY refinement

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py:35-45,95-105,290-305`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py:24-45`
- Test: `tests/test_dh116_tensor_task.py:21-30`

**Interfaces:**
- `constrain_palm_exploration(actions, phase, mode)` remains the controller entry point and returns `(N,2,3)` normalized palm actions.
- Add module constants `PALM_XY_SEARCH_LIMIT_M = 0.0005`, `PALM_Z_SEARCH_LIMIT_M = 0.004`, and `PALM_XY_EXPLORATION_WEIGHT = 0.25`.
- The `z_down` mode clamps close XY to the weighted ±0.5 mm window, clamps close Z to `[-4 mm, 0]`, and zeros all palm motion after phase 3.

- [ ] **Step 1: Update the existing z-down test and add the new bounds assertions**

```python
def test_z_down_allows_small_weighted_xy_and_shared_z_window():
    actions = torch.tensor([[[1., -1., -1.], [-1., 1., .5]]], dtype=torch.float64)
    phase = torch.tensor([1])
    constrained = constrain_palm_exploration(actions, phase, 'z_down')
    assert constrained[..., 0].abs().max().item() <= .5 / 3.0 * .25 + 1e-12
    assert constrained[..., 1].abs().max().item() <= .5 / 3.0 * .25 + 1e-12
    assert constrained[..., 2].max().item() <= 0.
    assert constrained[..., 2].min().item() >= -1.

def test_z_down_zeroes_xy_and_z_after_lift():
    actions = torch.ones((1, 2, 3), dtype=torch.float64)
    assert torch.equal(constrain_palm_exploration(actions, torch.tensor([3]), 'z_down'),
                       torch.zeros_like(actions))
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_dh116_tensor_task.py -k 'z_down_allows_small_weighted_xy or z_down_zeroes_xy'`

Expected: FAIL because the current `z_down` path zeros XY and does not expose the bounded weighted XY contract.

- [ ] **Step 3: Implement the minimal controller change**

```python
PALM_XY_SEARCH_LIMIT_M = .0005
PALM_Z_SEARCH_LIMIT_M = .004
PALM_XY_EXPLORATION_WEIGHT = .25

def constrain_palm_exploration(actions, phase, mode):
    if mode not in ('xyz', 'z_down'):
        raise ValueError('unsupported palm exploration mode')
    if actions.ndim != 3 or actions.shape[-1] != 3 or phase.shape != actions.shape[:1]:
        raise ValueError('palm actions must be (N, 2, 3) and phase must be (N,)')
    result = actions.clone()
    if mode == 'z_down':
        close = phase < 3
        xy_limit = PALM_XY_SEARCH_LIMIT_M / .003 * PALM_XY_EXPLORATION_WEIGHT
        result[close, :, :2] = result[close, :, :2].clamp(-xy_limit, xy_limit)
        result[close, :, 2] = result[close, :, 2].clamp(-PALM_Z_SEARCH_LIMIT_M / .004, 0.)
        result[~close] = 0.
    return result
```

The coordinated path must call this same function before subtracting its common pad anchor, so both hands receive identical Z action values while retaining only the permitted XY refinement.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_dh116_tensor_task.py -k 'z_down_allows_small_weighted_xy or z_down_zeroes_xy'`

Expected: PASS.

- [ ] **Step 5: Commit the isolated palm exploration change**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_task.py src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py tests/test_dh116_tensor_task.py
git commit -m "feat: bound synchronized palm xy exploration"
```

### Task 2: Route normal-only close force and combined lift force through the tensor task

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_world.py:26-37,180-190,220-230`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_env.py:130-140`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py:365-520`
- Test: `tests/test_dh116_tensor_task.py:69-165`

**Interfaces:**
- Add `signed_hand_lift_vertical_force(vertical_components)` returning `vertical_components[...,0] + vertical_components[...,1]` with the same shape and finite-value contract as the normal helper.
- `TensorTask.consume(..., vertical_force_n=None, lift_vertical_force_n=None, ...)` keeps `vertical_force_n` as the close normal-only channel and accepts the combined channel for phase-3 gates.
- `TensorEnv.step` passes `w.vertical_components[:,:,0].sum(1)` as `vertical_force_n` and `signed_hand_lift_vertical_force(w.vertical_components).sum(1)` as `lift_vertical_force_n`.

- [ ] **Step 1: Write the failing tests**

```python
def test_lift_force_channel_adds_friction_only_after_lift():
    components = torch.tensor([[[1.0, .5], [2.0, -.25]]], dtype=torch.float64)
    assert torch.equal(signed_hand_normal_vertical_force(components), torch.tensor([[1., 2.]]))
    assert torch.equal(signed_hand_lift_vertical_force(components), torch.tensor([[1.5, 1.75]]))

def test_lift_success_uses_combined_force_but_close_penalty_uses_normal_force():
    task = TensorTask(template(), 1, 'cpu')
    task.phase[:] = 3
    normal = torch.tensor([-.1], dtype=torch.float64)
    combined = torch.tensor([1.0], dtype=torch.float64)
    task.consume(measurement(task), torch.tensor([True]), .01,
                 vertical_force_n=normal, lift_vertical_force_n=combined)
    assert task.terms['close_downward'].item() == 0.
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_dh116_tensor_task.py -k 'lift_force_channel or lift_success_uses_combined'`

Expected: FAIL because no combined lift channel exists in the world/task interface.

- [ ] **Step 3: Implement the separate force channels**

```python
def signed_hand_lift_vertical_force(vertical_components):
    if vertical_components.ndim != 3 or vertical_components.shape[-1] != 2:
        raise ValueError('vertical components must be N x 2 x 2')
    return vertical_components[..., 0] + vertical_components[..., 1]
```

In `consume`, validate both channels when supplied, keep close readiness, close continuous scoring, `close_downward_impulse`, `downward_force`, and `close_downward` on `vertical_force_n`, and use `lift_vertical_force_n` only for lift stability/success checks and any existing lift admission check that runs after phase 3. Preserve all existing reward coefficients and phase timing.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_dh116_tensor_task.py -k 'lift_force_channel or lift_success_uses_combined'`

Expected: PASS.

- [ ] **Step 5: Commit the phase-specific force routing**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_world.py src/go2_pvcnn/control/dh116/tabletop_tensor_env.py src/go2_pvcnn/control/dh116/tabletop_tensor_task.py tests/test_dh116_tensor_task.py
git commit -m "feat: separate close and lift vertical force channels"
```

### Task 3: Change PPO force targets to 2--12 N and version the checkpoint/probe contracts

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py:276-285,340-355`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py:58-75`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_control_probe.py:23-28,240-370`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_training.py:390-430,checkpoint_contract_changes`
- Test: `tests/test_dh116_control_probe.py:21-25`
- Test: `tests/test_dh116_tensor_training.py:130-220`

**Interfaces:**
- Add `POLICY_MIN_FORCE_N = 2.0` and `POLICY_MAX_FORCE_N = 12.0`; map every tensor/coordinated force action with `2 + 0.5*(action+1)*(12-2)` and clamp requests to the same interval.
- `force_action(force, maximum)` encodes the new lower bound and rejects targets below 2 N or above the configured maximum.
- The training contract records `force_action_range_n=[2.0,12.0]`, `palm_exploration` version/bounds/XY weight, and phase-specific force semantics.
- `checkpoint_contract_changes` returns a named incompatibility for a saved `[0.1,12.0]` contract; ordinary resume raises the existing contract mismatch error.

- [ ] **Step 1: Write the failing tests**

```python
def test_force_target_roundtrip_and_bounds():
    for force in (2., 4., 8., 12.):
        encoded = p.force_action(force, 12.)
        assert 2. + (encoded + 1.) * 5. == pytest.approx(force)
    with pytest.raises(ValueError):
        p.force_action(1.99, 12.)
```

```python
def test_old_force_range_is_not_resume_compatible():
    old = dict(current_contract, force_action_range_n=[.1, 12.])
    new = dict(current_contract, force_action_range_n=[2., 12.])
    assert checkpoint_contract_changes(old, new) == ['force_action_range_2_to_12']
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_dh116_control_probe.py::test_force_target_roundtrip_and_bounds tests/test_dh116_tensor_training.py -k 'force_range or contract'`

Expected: FAIL because the current mapper starts at 0.1 N and contracts do not record the lower bound.

- [ ] **Step 3: Implement the new mapping and metadata**

```python
POLICY_MIN_FORCE_N = 2.0
POLICY_MAX_FORCE_N = 12.0

nominal = POLICY_MIN_FORCE_N + (actions[:, 18:] + 1.) * .5 * (maximum - POLICY_MIN_FORCE_N)
requested = nominal.clamp(POLICY_MIN_FORCE_N, maximum)
```

Use the same constants in both tensor control paths and the probe encoder. Add a versioned contract field rather than silently rewriting old checkpoints. Keep the existing fresh-policy bias near the new floor and make its expected initial target explicit in the contract.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_dh116_control_probe.py::test_force_target_roundtrip_and_bounds tests/test_dh116_tensor_training.py -k 'force_range or contract'`

Expected: PASS.

- [ ] **Step 5: Commit the force range and contract changes**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_task.py src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py src/go2_pvcnn/control/dh116/tabletop_control_probe.py src/go2_pvcnn/control/dh116/tabletop_tensor_training.py tests/test_dh116_control_probe.py tests/test_dh116_tensor_training.py
git commit -m "feat: bound ppo hand force targets to two to twelve newtons"
```

### Task 4: Run regression checks and a fresh teacher/probe validation

**Files:**
- Modify: `docs/superpowers/specs/2026-10-27-ppo-synchronized-z-force-phase-design.md` only if implementation metadata names differ from the approved spec.
- Create: a new run directory under `runs/` for the validation output; do not overwrite historical runs.

**Interfaces:**
- The validation probe uses the new contract and reports normal close force, combined lift force, per-hand loads, contact centers/validity, phase, failure code, and success.

- [ ] **Step 1: Run focused unit tests**

Run: `pytest -q tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py tests/test_dh116_tensor_training.py`

Expected: all focused tests pass.

- [ ] **Step 2: Run static checks**

Run: `python -m compileall -q src tests && git diff --check`

Expected: exit status 0.

- [ ] **Step 3: Run the teacher contact-Z probe with the latest checkpoint as inference-only input**

Use the existing `tabletop_rigid_20261026_v160_force_direction_ppo4096/tensor_ppo.pt` checkpoint only for inference, write to a new `runs/teacher_position_probe_20261027/...` directory, and verify the report contains the new force range, XY refinement metadata, close normal-force samples, and lift combined-force samples.

- [ ] **Step 4: Record validation evidence**

Confirm that no historical run was overwritten, ordinary resume of the old 0.1--12 N contract is rejected, and the known unrelated tensor observation tolerance failure remains unchanged if it reappears.

- [ ] **Step 5: Commit only implementation and validation metadata**

```bash
git status --short
git diff --check
```

Expected: the final diff contains only the approved implementation, tests, and this plan/spec metadata.
