# Force Direction Reposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a signed-vertical-force release→downward-reposition→reclose controller, validate it in 8/16 Isaac environments, then start a fresh 4096-environment PPO run with an isolated TensorBoard.

**Architecture:** Extend the existing loaded-palm state machine with per-hand signed hand-to-object vertical force gates. The world publishes per-hand vertical force in the device-resident measurement, the coordinated controller passes it to the state machine, and training telemetry records force before release, after release, during motion, and after reclose. The old `release_reposition_reclose_v1` path remains available for preserved runs; the new controller uses a new contract and a new CLI flag, so the fresh PPO run cannot silently resume the old behavior.

**Tech Stack:** Python 3, PyTorch CUDA tensors, Isaac Sim/PhysX, RSL-RL PPO, pytest, TensorBoard.

## Global Constraints

- Close-only correction; the state machine never runs during lift.
- Trigger at signed hand-to-object vertical force `< -0.15 N` with contact load `> 0.1 N`.
- Permit motion only after hand load `≤ 0.05 N` and vertical force `≥ -0.05 N` for `0.05 s`.
- Reclose requires hand load `> 0.1 N` and vertical force `≥ -0.15 N` for `0.05 s`.
- Maximum two attempts per hand; retain existing table-clearance, joint, velocity, mimic, force, NaN, and reset guards.
- Do not change reward weights in this phase.
- Fresh PPO must not load a policy, optimizer, RNG state, or episode counters from v159.
- Use 8-env and 16-env probes before 4096-env training.
- Keep prior runs intact and write the new TensorBoard to a new run directory.

---

### Task 1: Add failing tests for force-gated repositioning

**Files:**
- Modify: `tests/test_dh116_loaded_palm_motion.py`
- Create: `tests/test_dh116_force_direction_reposition.py`

**Interfaces:**
- The new controller must expose `ForceDirectionReposition.apply(target, current, loads, phase, live, dt, vertical_hand_force_n, clearance=None)` and return `(palm_target, active_release_mask)`.
- The controller must expose `state`, `attempts`, `aborted`, and force diagnostics used by training telemetry.

- [ ] **Step 1: Write the failing unit tests**

Add tests with deterministic `torch.float64` tensors:

```python
from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition


def test_negative_vertical_force_releases_before_downward_motion():
    c = ForceDirectionReposition(torch.tensor([True]))
    current = torch.zeros((1, 2, 3), dtype=torch.float64)
    target = current.clone(); target[:, :, 2] = -0.001
    loads = torch.full((1, 2), 1.0, dtype=torch.float64)
    fz = torch.tensor([[-0.30, -0.02]], dtype=torch.float64)
    phase = torch.ones(1, dtype=torch.long); live = torch.ones(1, dtype=torch.bool)
    out, active = c.apply(target, current, loads, phase, live, .01, fz)
    torch.testing.assert_close(out, current)
    assert active.tolist() == [[True, False]]
    assert c.state[0, 0].item() == 1


def test_move_requires_both_load_and_vertical_force_release():
    c = ForceDirectionReposition(torch.tensor([True]))
    current = torch.zeros((1, 2, 3), dtype=torch.float64)
    target = current.clone(); target[:, :, 2] = -0.001
    loads = torch.full((1, 2), 1.0, dtype=torch.float64)
    fz = torch.tensor([[-0.30, -0.02]], dtype=torch.float64)
    phase = torch.ones(1, dtype=torch.long); live = torch.ones(1, dtype=torch.bool)
    c.apply(target, current, loads, phase, live, .01, fz)
    loads[:, 0] = 0.; fz[:, 0] = -0.10
    for _ in range(5):
        out, _ = c.apply(target, current, loads, phase, live, .01, fz)
    torch.testing.assert_close(out[0, 0], current[0, 0])
    assert c.state[0, 0].item() == 1
    fz[:, 0] = 0.
    for _ in range(5):
        out, _ = c.apply(target, current, loads, phase, live, .01, fz)
    assert c.state[0, 0].item() == 2
    torch.testing.assert_close(out[0, 0, 2], torch.tensor(-.001, dtype=torch.float64))


def test_recontact_or_negative_force_stops_motion_and_reclose_needs_force_gate():
    c = ForceDirectionReposition(torch.tensor([True]))
    current = torch.zeros((1, 2, 3), dtype=torch.float64)
    target = current.clone(); target[:, :, 2] = -0.001
    loads = torch.full((1, 2), 1.0, dtype=torch.float64)
    fz = torch.tensor([[-0.30, -0.02]], dtype=torch.float64)
    phase = torch.ones(1, dtype=torch.long); live = torch.ones(1, dtype=torch.bool)
    c.apply(target, current, loads, phase, live, .01, fz)
    loads.zero_(); fz.zero_()
    for _ in range(5): c.apply(target, current, loads, phase, live, .01, fz)
    assert c.state[0, 0].item() == 2
    current[:, :, 2] = -.001
    loads[:, 0] = 1.; fz[:, 0] = -.20
    out, _ = c.apply(target, current, loads, phase, live, .01, fz)
    torch.testing.assert_close(out[0, 0], current[0, 0])
    assert c.state[0, 0].item() == 1
    for _ in range(205): c.apply(target, current, loads, phase, live, .01, fz)
    assert c.aborted[0, 0].item()
```

The test must also cover state 3, completed state 0, and the two-attempt abort using the same deterministic tensors and time steps.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest \
  tests/test_dh116_force_direction_reposition.py -q
```

Expected: collection or assertion failure because `tabletop_force_direction_reposition.py` does not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_dh116_loaded_palm_motion.py tests/test_dh116_force_direction_reposition.py
git commit -m "test: specify force direction reposition gates"
```

### Task 2: Implement the force-direction state machine

**Files:**
- Create: `src/go2_pvcnn/control/dh116/tabletop_force_direction_reposition.py`
- Test: `tests/test_dh116_force_direction_reposition.py`

**Interfaces:**
- `CONTRACT` contains `version='force_direction_release_reposition_v1'`, `trigger_force_n=-.15`, `release_force_n=-.05`, `reclose_force_n=-.15`, `contact_load_n=.1`, `released_load_n=.05`, `released_dwell_s=.05`, `correction_step_m=.001`, `max_downward_step_m=.002`, `arrival_tolerance_m=.0002`, `arrival_dwell_s=.05`, `adjust_timeout_s=2.`, `reclose_timeout_s=1.`, and `max_attempts_per_hand=2`.
- `ForceDirectionReposition(mask)` stores device-resident state and diagnostics, preserves unloaded/upward requests, and never starts when `phase != 1`.
- `apply(...)` uses per-hand signed force. If only one hand is below the trigger, only that hand enters release. If both hands are negative and the aggregate is below the trigger, both enter release. A force-triggered goal is `min(policy_target_z, current_z - correction_step_m)` and is clipped to the existing clearance margin and `max_downward_step_m`.

- [ ] **Step 1: Implement the contract and state tensors**

Use the existing `LoadedPalmReposition` tensor layout, adding `force_before`, `force_release`, `force_move`, `force_reclose`, `release_timeout`, `recontact_abort`, and `reclose_timeout` with shape `(N, 2)`.

- [ ] **Step 2: Implement release, move, and reclose transitions**

Use these exact transition predicates:

```python
selected = mask[:, None] & live[:, None] & (phase == 1)[:, None]
negative = vertical_hand_force_n < CONTRACT['trigger_force_n']
aggregate_negative = vertical_hand_force_n.sum(1, keepdim=True) < CONTRACT['trigger_force_n']
trigger = selected & (negative | (aggregate_negative & (vertical_hand_force_n < 0.))) & (loads > CONTRACT['contact_load_n'])
released = (loads <= CONTRACT['released_load_n']) & (vertical_hand_force_n >= CONTRACT['release_force_n'])
reclose_ready = (loads > CONTRACT['contact_load_n']) & (vertical_hand_force_n >= CONTRACT['reclose_force_n'])
```

Keep existing mimic, pressure, slew, clearance, and table-margin governors outside this module. A recontact or force violation during state 2 returns that hand to state 1 before any palm target is emitted.

- [ ] **Step 3: Run the focused tests to verify they pass**

```bash
CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest \
  tests/test_dh116_force_direction_reposition.py -q
```

Expected: all new state-machine tests pass.

- [ ] **Step 4: Commit the controller**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_force_direction_reposition.py tests/test_dh116_force_direction_reposition.py
git commit -m "feat: gate palm reposition on signed vertical force"
```

### Task 3: Publish per-hand vertical force through the tensor world and coordinated controller

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_world.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py`
- Modify: `tests/test_dh116_tensor_world.py`
- Modify: `tests/test_dh116_coordinated_grasp.py`

**Interfaces:**
- `TensorWorld.measure()` returns `vertical_hand_force_n` with shape `(N, 2)`, computed as `self.vertical_components.sum(dim=2)` using the existing signed hand-to-object convention.
- `TensorTask.keys` includes `vertical_hand_force_n`; reset and finite-state validation treat it like other sensor fields.
- `LoadedPalmReposition` remains available for old runs; `ForceDirectionReposition` is selected only by the new training flag.

- [ ] **Step 1: Add a failing world measurement test**

```python
def test_measure_publishes_signed_per_hand_vertical_force():
    world = make_world_for_test()
    world.vertical_components[:] = torch.tensor([[[1., -2.], [3., -4.]]])
    measured = world.measure()
    torch.testing.assert_close(measured['vertical_hand_force_n'], torch.tensor([[-1., -1.]], dtype=torch.float64))
```

Adapt only the fixture constructor to the existing test helper; do not create a second physics world.

- [ ] **Step 2: Add the measurement field and controller call**

Add `vertical_hand_force_n` to `TensorTask.keys`, copy it through reset/consume, and pass it from coordinated control:

```python
palm, releasing = t.palm_reposition.apply(
    palm, m['adaptive_palm_offset'], loads, t.phase, live, dt,
    m['vertical_hand_force_n'], clearance=m['hand_clearance'])
```

The controller call must remain behind the new `force_direction_reposition` selection so the preserved v159 path is byte-for-byte unchanged when disabled.

- [ ] **Step 3: Add integration tests**

Cover: one-hand selective release, both-hand release for aggregate negative force, no trigger in lift, no motion until both load and force gates pass, and no change when the new flag is disabled.

- [ ] **Step 4: Run world and coordinated tests**

```bash
CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest \
  tests/test_dh116_tensor_world.py tests/test_dh116_coordinated_grasp.py \
  tests/test_dh116_force_direction_reposition.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit sensor/control integration**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_world.py \
  src/go2_pvcnn/control/dh116/tabletop_tensor_task.py \
  src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py \
  tests/test_dh116_tensor_world.py tests/test_dh116_coordinated_grasp.py
git commit -m "feat: feed per-hand vertical force into coordinated control"
```

### Task 4: Add the new training contract, diagnostics, and fresh-run CLI

**Files:**
- Modify: `scripts/train_dh116_tabletop_gpu_ppo.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_training.py`
- Modify: `scripts/train_dh116_tensor_ppo.sh`
- Modify: `tests/test_dh116_tensor_training.py`
- Modify: `tests/test_dh116_control_probe.py`

**Interfaces:**
- Add `--force-direction-reposition`; it requires coordinated tensor training, driven carriage, pressure relief, measured-load direct lift, and zero reset jitter. It cannot be combined with `--resume`, `--initialize-from`, or a control probe checkpoint.
- Training instantiates `ForceDirectionReposition` only for this flag and writes `palm_reposition` contract version `force_direction_release_reposition_v1`.
- A fresh run writes `training_start.json` with `resume_source: null`, `initial_iteration: 0`, `initial_timesteps: 0`, `initial_episodes: 0`, and zero optimizer states.

- [ ] **Step 1: Add failing contract tests**

Test that the new flag produces a distinct contract, that old v159 checkpoints are rejected rather than migrated, and that a fresh launch cannot receive `--resume` or `--initialize-from`.

- [ ] **Step 2: Implement CLI validation and controller selection**

Keep `--release-reposition` unchanged for legacy runs. Add the new flag and validation, then select `ForceDirectionReposition` in `run_tensor_training`.

- [ ] **Step 3: Add per-phase diagnostics**

At each PPO iteration, publish JSON and TensorBoard scalars:

- `reposition/force_before_release_n`
- `reposition/force_after_release_n`
- `reposition/force_during_move_n`
- `reposition/force_after_reclose_n`
- `reposition/release_timeout_fraction`
- `reposition/recontact_abort_fraction`
- `reposition/reclose_timeout_fraction`
- existing state and abort fractions

Do not add or change reward terms in this task.

- [ ] **Step 4: Update the wrapper for a no-resume fresh run**

The wrapper must reject `DH116_RESUME` when `DH116_FORCE_DIRECTION_REPOSITION=1`, pass `--force-direction-reposition`, and default output to a new timestamped directory. It must not delete or overwrite v159.

- [ ] **Step 5: Run training-contract and probe tests**

```bash
CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest \
  tests/test_dh116_tensor_training.py tests/test_dh116_control_probe.py -q
bash -n scripts/train_dh116_tensor_ppo.sh
```

Expected: all tests pass and the shell script has no syntax errors.

- [ ] **Step 6: Commit training integration**

```bash
git add scripts/train_dh116_tabletop_gpu_ppo.py scripts/train_dh116_tensor_ppo.sh \
  src/go2_pvcnn/control/dh116/tabletop_tensor_training.py \
  tests/test_dh116_tensor_training.py tests/test_dh116_control_probe.py
git commit -m "feat: add fresh force direction PPO contract"
```

### Task 5: Run the 8/16 environment probes

**Files:**
- Create: `runs/tabletop_rigid_20261026_force_direction_probe8/` (generated)
- Create: `runs/tabletop_rigid_20261026_force_direction_probe16/` (generated)
- Create: `runs/tabletop_rigid_20261026_force_direction_probe_summary.json` (generated)

**Interfaces:**
- Probe mode uses the new force-direction controller with a deterministic candidate action and no PPO update.
- Probe output must include state ordering, force gate values, per-hand force diagnostics, clearance, solver peaks, and abort causes.

- [ ] **Step 1: Run the 8-env probe**

Use the validated DH116 scene parameters and a new output directory; do not pass `--resume` or `--initialize-from`. Require all requested episodes to complete without hard safety failure.

- [ ] **Step 2: Check the 8-env result**

Require: every completed reposition has release before move, no move while either release gate is false, no lift-phase reposition, no NaN/penetration/over-force, and a measurable decrease in negative vertical force after release.

- [ ] **Step 3: Run the 16-env perturbation probe**

Use the existing counterbalanced probe mechanism plus small downward grasp-point perturbations. Record mean/peak negative force, completed moves, recontacts, timeouts, and oscillation counts.

- [ ] **Step 4: Write the probe summary and commit generated evidence metadata**

The summary must state whether the controller is physically qualified for PPO. Do not claim grasp success unless the object reaches the existing 30 mm/0.5 s success condition.

### Task 6: Start fresh PPO and isolated TensorBoard

**Files:**
- Create: `runs/tabletop_rigid_20261026_v160_force_direction_ppo4096/` (generated)
- Create: `runs/active_tabletop_run.json` (update only after launch)
- Create: `runs/tabletop_rigid_20261026_v160_force_direction_ppo4096/tensorboard/` (generated)

**Interfaces:**
- Start command uses 4096 environments, 24 rollout steps, 480 Hz, direct tensor PPO, coordinated controller, driven carriage, force-direction reposition, existing high-friction rigid cylinder, and 1024 episodes per environment.
- Command contains neither `--resume` nor `--initialize-from`.
- New TensorBoard binds to a free port and points only to the new run directory.

- [ ] **Step 1: Verify no Isaac training/replay process is running**

```bash
pgrep -af '[t]rain_dh116_tabletop_gpu_ppo.py|[r]ecovery_replay.py|[a]ction_capture.py|[t]ensor_fault_trace.py|[t]ensor_terminal_trace.py'
```

Expected: no matching process.

- [ ] **Step 2: Launch from zero**

```bash
DH116_OUTPUT=runs/tabletop_rigid_20261026_v160_force_direction_ppo4096 \
DH116_FORCE_DIRECTION_REPOSITION=1 \
nohup bash scripts/train_dh116_tensor_ppo.sh > runs/tabletop_rigid_20261026_v160_force_direction_ppo4096/train.log 2>&1 &
```

Do not set `DH116_RESUME`.

- [ ] **Step 3: Verify fresh-start evidence**

Check `training_start.json`, `tensor_contract.json`, `monitor.json`, and the process command line. Require null resume source, zero initial counters, the new contract version, and a new TensorBoard event file.

- [ ] **Step 4: Start isolated TensorBoard and report the URL**

```bash
tensorboard --logdir_spec 'force_direction_v160:runs/tabletop_rigid_20261026_v160_force_direction_ppo4096/tensorboard' --port 6009 --host 0.0.0.0
```

Verify the tags endpoint contains only `force_direction_v160` and report the URL only after it responds.

- [ ] **Step 5: Monitor without intervention unless a defined fault appears**

Read `monitor.json` and TensorBoard. Intervene only for process exit, NaN, hard guard, oscillating state machine, clearance violation, or three consecutive 50-iteration windows without improvement in negative force and lift admission. Keep the previous v159 run and all probe evidence.
