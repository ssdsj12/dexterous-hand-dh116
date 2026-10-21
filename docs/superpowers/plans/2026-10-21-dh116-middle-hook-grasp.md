# DH116 Middle-Finger Wire Hook Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add measured left/right DH116 middle-finger support contacts so the existing bimanual sequence can close, hold and lift the DeformX wire with six physical contact channels.

**Architecture:** Extend the existing ordered contact-channel contract from four channels to six, keeping thumb/index as the closing pair and treating middle fingers as lift-only support contacts. Generate middle-finger source references and pinned convex hulls from the current DH116 USD, feed all six moving meshes into DeformX, allocate middle-finger Jacobian motion toward the wire underside, and require measured middle support during hold/lift.

**Tech Stack:** Python 3.11, NumPy, SciPy, Open3D, trimesh, Isaac Lab/PhysX, DeformX/PyElastica, pytest.

## Global Constraints

- Preserve source geometry collision protection and reject non-target body collisions.
- Keep the runtime single-threaded on CPU0 through `scripts/python.sh`.
- Use measured DeformX contact impulses and Isaac reaction wrenches; never infer success from commanded pose.
- Keep the current fixed dual Panda + DH116 asset and the current table/wire coordinate frame.
- The final run must record actual close, continuous hold, middle-finger support, wire rise, and whole-wire clearance.

---

### Task 1: Define the six-channel contact contract

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/contact_runtime.py`
- Modify: `src/go2_pvcnn/control/dh116/grasp_sequence.py`
- Modify: `src/go2_pvcnn/control/dh116/touch_first.py`
- Modify: `src/go2_pvcnn/control/dh116/grasp_evidence.py`
- Modify: `src/go2_pvcnn/control/dh116/stage_runtime.py`
- Modify: `src/go2_pvcnn/control/dh116/stage_state.py`
- Modify: `src/go2_pvcnn/control/dh116/material_tracking.py`
- Test: `tests/test_dh116_middle_hook_contract.py`

**Interfaces:**
- `BODY_NAMES` becomes the ordered tuple/mapping `left_thumb`, `left_index`, `left_middle`, `right_thumb`, `right_index`, `right_middle`.
- `TabletopGraspSequence` accepts six target loads but exposes `pinch_indices=(0,1,3,4)` and `support_indices=(2,5)`; close qualification uses pinch indices, hold/lift qualification additionally requires support indices.
- `GraspEvidenceGate.update(time_s, *, loads_n, valid_contacts, normals, anchors_local, station_rise_m, min_surface_clearance_m, safety_ok, active_constraints, lift_alignment_ok=True)` accepts six loads, validity flags, normals, and anchors while retaining two station heights.

- [ ] **Step 1: Write the failing contract tests.** Test exact channel order, six-element shapes, rejection of four-channel state, pinch-only close, and middle-support-required hold/lift:

```python
def test_middle_channels_are_ordered_and_required_for_lift():
    gate = GraspEvidenceGate(
        material_ids=(10, 11, 12, 30, 31, 32),
        approach_qualified=True,
        minimum_hold_loads_n=np.array([.9, .81, .18, .9, .81, .18]),
    )
    normals = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.],
                        [-1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
    evidence = gate.update(
        0.0,
        loads_n=np.array([1., .9, .2, 1., .9, .2]),
        valid_contacts=np.array([True, True, False, True, True, False]),
        normals=normals,
        anchors_local=np.zeros((6, 3)),
        station_rise_m=np.zeros(2),
        min_surface_clearance_m=.01,
        safety_ok=True,
        active_constraints=[],
    )
    assert evidence["stage"] == "close"
    assert not evidence["permit_lift"]
```

- [ ] **Step 2: Run the contract test and verify it fails because the current implementation requires four channels.**

Run: `PYTHONPATH=src bash scripts/python.sh -m pytest -q tests/test_dh116_middle_hook_contract.py`

Expected: FAIL with a six-channel shape/order assertion.

- [ ] **Step 3: Implement the minimal six-channel validation and stage separation.** Keep pinch thresholds and opposing-normal checks on indices `(0,1)` and `(3,4)`; require indices `(2,5)` to be valid with upward normals and load at least `0.9 * target` before `hold` and `lift` can advance.

- [ ] **Step 4: Run the focused tests and the existing stage/restore tests.**

Run: `PYTHONPATH=src bash scripts/python.sh -m pytest -q tests/test_dh116_middle_hook_contract.py tests/test_dh116_staged_sequence.py tests/test_dh116_stage_state.py`

Expected: PASS with no four-channel compatibility path accepted by the guarded sequence.

- [ ] **Step 5: Commit the contract changes.**

```bash
git add src/go2_pvcnn/control/dh116 tests/test_dh116_middle_hook_contract.py
git commit -m "feat: add six-channel DH116 hook contract"
```

### Task 2: Calibrate and pin the DH116 middle-finger geometry

**Files:**
- Modify: `scripts/calibrate_dh116_surfaces.py`
- Modify: `scripts/build_dh116_contact_meshes.py`
- Create: `configs/symmetric_grasp_middle_v1/point_references.json`
- Create: `configs/symmetric_grasp_middle_v1/contact/manifest.json`
- Create: `configs/symmetric_grasp_middle_v1/contact/left_middle.ply`
- Create: `configs/symmetric_grasp_middle_v1/contact/right_middle.ply`
- Create: `scripts/select_dh116_grasp_references.py`
- Test: `tests/test_dh116_middle_hook_geometry.py`

**Interfaces:**
- Calibration emits one reference for each side/digit in `thumb,index,middle` and records `finger33_link` for middle.
- Contact mesh generation accepts the same digit set and writes source SHA, derived mesh SHA, local mesh COM, reference point, and reference normal.

- [ ] **Step 1: Add a failing geometry test** that checks both middle references use `finger33_link`, have finite local points/normals, and have normals oriented toward the wire from below.

- [ ] **Step 2: Run the test to verify the current four-reference bundle fails.**

- [ ] **Step 3: Add `middle` to the calibration/build digit lists and generate the new bundle from the fixed USD.** The selector converts the six calibrated entries into the runner's pinned grasp-reference schema and preserves all source/layer SHA values.

```bash
PYTHONPATH=src bash scripts/python.sh scripts/calibrate_dh116_surfaces.py \
  --asset assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd \
  --output configs/symmetric_grasp_middle_v1/surfaces
PYTHONPATH=src bash scripts/python.sh scripts/build_dh116_contact_meshes.py \
  --references configs/symmetric_grasp_middle_v1/surfaces/references.json \
  --output configs/symmetric_grasp_middle_v1/contact
PYTHONPATH=src bash scripts/python.sh scripts/select_dh116_grasp_references.py \
  --surfaces configs/symmetric_grasp_middle_v1/surfaces/references.json \
  --output configs/symmetric_grasp_middle_v1/point_references.json
```

- [ ] **Step 4: Add the wire-under-fingertip placement data** to the new references: each middle point is below the centerline at the approach pose, and its world normal points upward toward the wire. Reject a reference whose source mesh front partition or table clearance is invalid.

- [ ] **Step 5: Run geometry and source-screen tests.**

Run: `PYTHONPATH=src bash scripts/python.sh -m pytest -q tests/test_dh116_middle_hook_geometry.py tests/test_dh116_usd_contract.py`

Expected: PASS and both middle derived meshes have recorded SHA pins.

- [ ] **Step 6: Commit the pinned geometry bundle.**

```bash
git add scripts/calibrate_dh116_surfaces.py scripts/build_dh116_contact_meshes.py \
  configs/symmetric_grasp_middle_v1 tests/test_dh116_middle_hook_geometry.py
git commit -m "feat: pin DH116 middle-finger contact geometry"
```

### Task 3: Connect six moving meshes, Jacobians, and reaction wrenches

**Files:**
- Modify: `scripts/verify_fixed_dual_panda_dh116.py`
- Modify: `src/go2_pvcnn/control/dh116/coordinated_closure.py`
- Modify: `src/go2_pvcnn/control/dh116/pad_region_guard.py`
- Test: `tests/test_dh116_middle_hook_runtime.py`

**Interfaces:**
- The runner loads the six-channel reference/contact bundle and builds six `BodyState`/`MeshMotion` values per physics step.
- `coordinated_bimanual_increment` receives six normals/Jacobian rows and a six-element active-pad mask.
- Middle channels use a support velocity target toward the wire underside; their reaction wrenches are submitted through the same per-side articulation sink.

- [ ] **Step 1: Write failing runtime tests** for six `BodyState` entries, six contact candidates, middle active-mask behavior, and rejection when a middle mesh/body identity is wrong.

- [ ] **Step 2: Run focused tests and observe failure against the four-channel runner.**

- [ ] **Step 3: Replace hard-coded `('thumb','index')` loops in the runner with the ordered channel metadata.** Use separate control policy: thumb/index close toward the wire; middle target normal is upward and its command is suppressed until close begins.

- [ ] **Step 4: Keep middle support active during hold/lift and route its measured `linear_w`, `angular_com_w`, and `position_w` through the existing reaction submission.** Do not populate loads from targets when no impulse is returned.

- [ ] **Step 5: Add explicit evidence fields** `middle_hook_load_n`, `middle_hook_valid`, `middle_hook_direction_w`, and `middle_hook_contact_duration_s` to every trajectory row and summary.

- [ ] **Step 6: Run runtime unit tests and compile the runner.**

Run: `PYTHONPATH=src bash scripts/python.sh -m pytest -q tests/test_dh116_middle_hook_runtime.py tests/test_dh116_coordinated_closure.py && python3 -m py_compile scripts/verify_fixed_dual_panda_dh116.py`

- [ ] **Step 7: Commit the runtime integration.**

```bash
git add scripts/verify_fixed_dual_panda_dh116.py src/go2_pvcnn/control/dh116 \
  tests/test_dh116_middle_hook_runtime.py
git commit -m "feat: drive DH116 middle-finger wire support"
```

### Task 4: Run direct close, hold, and lift with physical evidence

**Files:**
- Create: `runs/dh116_middle_hook_close_lift_20261021/`
- Modify: `README.md`

**Interfaces:**
- The run uses the six-channel config, `--grasp-sequence`, `--gravity-hold`, `--table-wire`, `--coordinated-closure`, and CPU0/single-thread runtime settings.
- The run starts at the existing measured close checkpoint only after a new six-channel checkpoint is generated; imported state never counts as continuous grasp evidence.

- [ ] **Step 1: Generate a six-channel approach/close stage state** using the existing validated pose and new references; verify the checkpoint pins include the six-channel bundle. Write the close checkpoint to `runs/dh116_middle_hook_stage_20261021/stage_close.json`.

- [ ] **Step 2: Start direct close from that checkpoint with CPU0 and one numerical thread.** Set the pinned inputs immediately before the run:

```bash
STAGE=runs/dh116_middle_hook_stage_20261021/stage_close.json
STAGE_SHA=$(sha256sum "$STAGE" | awk '{print $1}')
CONTACT_SHA=$(sha256sum configs/symmetric_grasp_middle_v1/contact/manifest.json | awk '{print $1}')
```

```bash
CUDA_VISIBLE_DEVICES=0 DH116_CPU_AFFINITY=0 DH116_NUM_THREADS=1 \
PYTHONPATH=src bash scripts/python.sh scripts/verify_fixed_dual_panda_dh116.py \
  --asset-dir assets/fixed_dual_panda_dh116_v1 \
  --output runs/dh116_middle_hook_close_lift_20261021 \
  --grasp-sequence --restore-state "$STAGE" \
  --restore-state-sha256 "$STAGE_SHA" \
  --point-references configs/symmetric_grasp_middle_v1/point_references.json \
  --contact-manifest configs/symmetric_grasp_middle_v1/contact/manifest.json \
  --contact-manifest-sha256 "$CONTACT_SHA" \
  --closure-loads 1.0 0.9 0.2 1.0 0.9 0.2 \
  --gravity-hold --table-wire --coordinated-closure --symmetric-lift \
  --approach-wire-mode frozen --steps 90000 --headless
```

- [ ] **Step 3: Inspect the summary and trajectory.** Require `close`, `hold`, and `lift` PASS; four pinch loads at threshold; both middle loads valid and upward; continuous durations; reaction applied; rise at least 30 mm; whole wire clearance at least 2 mm. A failure remains a failure and is retained for diagnosis.

- [ ] **Step 4: Record the result and command in README.**

- [ ] **Step 5: Commit the verified run metadata and documentation.**

```bash
git add README.md
git commit -m "docs: record DH116 middle-hook grasp evidence"
```
