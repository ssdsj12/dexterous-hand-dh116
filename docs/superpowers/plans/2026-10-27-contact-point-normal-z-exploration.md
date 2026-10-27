# Contact-Point Normal Z Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the teacher Z probe target and validate real PhysX contact-point geometry and normal directions while preserving a 10 N total normal-load target.

**Architecture:** Add a pure tensor contact-geometry reducer, enable Isaac Lab contact-point tracking, carry measured contact summaries through `TensorWorld`, and make the teacher probe's Z controller use weighted measured contact centers. The existing force controller remains a scalar 5 N-per-hand load target; no external force is injected.

**Tech Stack:** Python 3.11, PyTorch tensors, Isaac Lab `ContactSensor`, pytest, headless Isaac Sim.

## Global Constraints

- Contact source is PhysX/Isaac Lab contact reports; `pad_relative` is telemetry only.
- Target Z candidates remain `[-0.004, -0.003, -0.002, -0.0015, -0.001, -0.0005]` m.
- Normal-load target remains 10 N total, 5 N per hand.
- XY and authored pinch geometry remain fixed; exploration changes world Z only.
- Every force-effective contact must be below the object centerline before lift admission.
- Missing or invalid contact data invalidate the candidate without proxy fallback.
- No object radius, density, reward, or trained checkpoint changes.

---

### Task 1: Add pure contact geometry aggregation

**Files:**
- Create: `src/go2_pvcnn/control/dh116/tabletop_contact_geometry.py`
- Test: `tests/test_dh116_contact_geometry.py`

**Interfaces:**
- `aggregate_contact_geometry(contact_positions_w, normal_forces_w, object_center_w, *, force_threshold_n=0.05, centerline_epsilon_m=1e-6) -> dict`
- Inputs: `(N, 2, L, 3)` contact positions, normal-force vectors, and `(N, 3)` object centers.
- Outputs: `contact_center_w (N,2,3)`, `contact_normal_w (N,2,3)`, `normal_load_n (N,2)`, `valid (N,2)`, `all_below_centerline (N,2)`, `contact_count (N,2)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_weighted_center_and_normal_use_real_contacts():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry
    positions = torch.tensor([[[[.008, 0., -.004], [-.006, 0., -.003]],
                               [[-.008, 0., -.004], [.006, 0., -.003]]]], dtype=torch.float64)
    forces = torch.tensor([[[[3., 0., 0.], [0., 0., 4.]],
                            [[-3., 0., 0.], [0., 0., 4.]]]], dtype=torch.float64)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert result['valid'].tolist() == [[True, True]]
    assert result['contact_count'].tolist() == [[2, 2]]
    assert result['normal_load_n'][0, 0].item() == pytest.approx(7.)
    assert result['contact_center_w'][0, 0, 2].item() == pytest.approx(-.0035714286)
    torch.testing.assert_close(result['contact_normal_w'][0, 0],
                               torch.tensor([3., 0., 4.], dtype=torch.float64)/5.)
    assert result['all_below_centerline'].tolist() == [[True, True]]


def test_invalid_contacts_are_rejected_without_proxy_fallback():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry
    positions = torch.full((1, 2, 2, 3), float('nan'), dtype=torch.float64)
    forces = torch.zeros_like(positions)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert not result['valid'].any()
    assert not result['all_below_centerline'].any()
    assert torch.isnan(result['contact_center_w']).all()


def test_loaded_contact_above_centerline_fails_all_below_gate():
    from go2_pvcnn.control.dh116.tabletop_contact_geometry import aggregate_contact_geometry
    positions = torch.tensor([[[[.008, 0., .001]], [[-.008, 0., -.001]]]], dtype=torch.float64)
    forces = torch.ones_like(positions)
    result = aggregate_contact_geometry(positions, forces, torch.zeros((1, 3), dtype=torch.float64))
    assert result['valid'].tolist() == [[True, True]]
    assert result['all_below_centerline'].tolist() == [[False, True]]
```

- [ ] **Step 2: Run the tests and verify the expected missing-module failure**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_contact_geometry.py`

Expected: FAIL because `tabletop_contact_geometry` does not exist.

- [ ] **Step 3: Implement the reducer**

Validate all shapes and finite object centers. Mark a contact valid only when
position and force are finite and the normal-force magnitude is at least
`force_threshold_n`. Sum valid force magnitudes per hand for `normal_load_n`;
use those magnitudes for the weighted contact center. Normalize the vector sum
of valid normal forces for `contact_normal_w`. Return NaN centers/normals for
invalid hands, zero counts for invalid hands, and require every valid contact
to satisfy `contact_z < center_z - centerline_epsilon_m` for the below-axis
gate.

- [ ] **Step 4: Run the focused tests**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_contact_geometry.py`

Expected: 3 passed.

- [ ] **Step 5: Commit the isolated reducer**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_contact_geometry.py tests/test_dh116_contact_geometry.py
git commit -m "feat: aggregate measured contact geometry"
```

### Task 2: Enable PhysX contact positions and expose TensorWorld measurements

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_world.py`
- Test: `tests/test_dh116_tensor_world.py`

**Interfaces:**
- Sensor configs set `track_contact_points=True` in tensor mode and retain a
  16-contact budget.
- TensorWorld measurement adds `contact_center_w`, `contact_normal_w`,
  `contact_normal_load_n`, `contact_valid`, `contact_all_below_centerline`,
  and `contact_count`.

- [ ] **Step 1: Write failing world measurement tests**

Add a fake sensor data fixture with `contact_pos_w` and assert that
`TensorWorld.measure()` contains the six measured-contact fields and preserves
the existing `pad_relative` field as separate telemetry.

- [ ] **Step 2: Run the test and verify failure**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_world.py`

Expected: FAIL because the new measurement keys are absent.

- [ ] **Step 3: Enable contact reporting**

Add `track_contact_points=bool(getattr(args, 'tensor_pipeline', False))` to
both sensor construction paths. Keep `track_friction_forces` enabled in tensor
mode and keep `max_contact_data_count_per_prim=16` in tensor mode.

- [ ] **Step 4: Aggregate and expose data**

In `TensorWorld.refresh`, stack each sensor's `data.contact_pos_w[:,0,0,:]`
with the same hand/link ordering as `force_matrix_w`. Pass those positions,
normal vectors, and `self.wp` to `aggregate_contact_geometry`. Cache the result
and include it in `measure()`. Reset all cached fields to invalid/NaN/zero for
reset slots.

- [ ] **Step 5: Run focused tests and compile**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_world.py tests/test_dh116_root_pose_writes.py && bash scripts/python.sh -m py_compile src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py src/go2_pvcnn/control/dh116/tabletop_tensor_world.py`

Expected: PASS and no compile output.

- [ ] **Step 6: Commit the sensor/measurement layer**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_gpu_isaac.py src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py src/go2_pvcnn/control/dh116/tabletop_tensor_world.py tests/test_dh116_tensor_world.py
git commit -m "feat: expose PhysX contact geometry in tensor world"
```

### Task 3: Add measured-contact target control and telemetry

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/tabletop_tensor_task.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py`
- Modify: `src/go2_pvcnn/control/dh116/tabletop_control_probe.py`
- Test: `tests/test_dh116_tensor_task.py`
- Test: `tests/test_dh116_control_probe.py`

**Interfaces:**
- `TensorTask` accepts `contact_target_z_m` `(N,2)` only for the measured
  contact teacher probe.
- A pure helper `contact_z_correction(current, measured, target, limit_m, dt)`
  returns an `(N,2,3)` Z-only palm correction and never changes X/Y.
- Probe mode `grasp_teacher_contact_z_sweep` uses the six target values and
  records measured-contact fields.

- [ ] **Step 1: Write failing tests**

Test that a positive/negative contact-center error changes only action Z, is
bounded by the per-tick correction limit, and that a hand with
`contact_valid=False` cannot pass the measured-contact gate. Add a probe-mode
test asserting the six case names are generated and that the probe contract
contains `contact_point_semantics='physx_contact_report_v1'`.

- [ ] **Step 2: Run the tests and verify failure**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py`

Expected: FAIL because the correction helper and probe mode do not exist.

- [ ] **Step 3: Implement Z-only measured-contact correction**

At each close control tick, compute the force-weighted contact-center error in
world Z. Apply a bounded correction through the existing palm-action path,
zeroing X/Y. Keep release/reclose and pressure governors ahead of the motion.
Require valid contacts, all-below gating, target tolerance, and stable 5 N
per-hand loads before marking the candidate ready.

- [ ] **Step 4: Implement the measured-contact probe mode**

Add `grasp_teacher_contact_z_sweep` to the CLI choices and probe dispatch.
Use the same six target values, set the contact-point contract version, and
store the measured contact center, normal, validity, count, normal load, and
vertical normal/friction components in the trace and JSON output. Never use
`pad_relative` for control decisions.

- [ ] **Step 5: Run focused tests**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py tests/test_dh116_force_direction_reposition.py`

Expected: all focused tests pass.

- [ ] **Step 6: Commit the probe/controller layer**

```bash
git add src/go2_pvcnn/control/dh116/tabletop_tensor_task.py src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py src/go2_pvcnn/control/dh116/tabletop_control_probe.py tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py
git commit -m "feat: explore measured contact points along z"
```

### Task 4: Run the six-point Isaac validation and publish evidence

**Files:**
- Create: `runs/teacher_position_probe_20261027/isaac_teacher_contact_z_8mm_10n_v1/` (generated, not committed)
- Create: `runs/teacher_position_probe_20261027/isaac_teacher_contact_z_8mm_10n_v1/contact_geometry_cross_section.png` (generated, not committed)

- [ ] **Step 1: Run the headless six-point probe**

Use the latest force-direction checkpoint and the existing 8 mm cylinder,
12 N-per-hand load profile, extreme-grip object material, coordinated
controller, and zero reset jitter. Select
`--control-probe-mode grasp_teacher_contact_z_sweep` and six probe cases.

- [ ] **Step 2: Verify measured contact semantics**

Read the output contract and assert `contact_valid`, `contact_count`,
`contact_center_w`, and `contact_normal_w` are present. Confirm candidate
control never changes XY and that no candidate falls back to `pad_relative`.

- [ ] **Step 3: Generate and inspect the cross-section figure**

Plot the cylinder, measured force-weighted contact centers, actual normal
vectors, target Z lines, normal-load split, and signed vertical components.

- [ ] **Step 4: Run final focused verification**

Run: `bash scripts/python.sh -m pytest -q tests/test_dh116_contact_geometry.py tests/test_dh116_tensor_world.py tests/test_dh116_tensor_task.py tests/test_dh116_control_probe.py tests/test_dh116_force_direction_reposition.py && git diff --check`

Expected: all selected tests pass and `git diff --check` is clean.

- [ ] **Step 5: Report evidence and limitations**

Report the six measured candidate points, normal vectors, 10 N split, lift
status, and any invalid candidates. Do not describe a candidate as successful
unless the actual lift gates and measured contact criteria pass.
