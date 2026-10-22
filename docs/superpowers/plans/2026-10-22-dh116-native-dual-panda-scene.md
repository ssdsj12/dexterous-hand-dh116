# DH116 Native Dual Panda Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and visibly validate a clean Isaac scene containing two Isaac-native Panda arms, two separately mounted DH116 V2.0 hands, the tabletop wire, and the candidate3 pose followed by direct close→hold→lift execution.

**Architecture:** The scene loader owns asset composition and explicit Panda-flange-to-DH116 root mounting. Each DH116 remains an independent world-root articulation; only its root prim is aligned to the native Panda flange, so PhysX does not create nested articulations. A pose/stage controller owns candidate3 joint mapping and close/hold/lift commands. Evidence code records asset identity, per-side joint state, stage, contact, wire height, and screenshots. The old combined USD and per-link visual pose writer remain available for historical replay but are not called by the new entry point.

**Tech Stack:** IsaacLab `FRANKA_PANDA_CFG`, Isaac Sim USD composition, DH116 V2.0 USD assets under `assets/fixed_dual_panda_dh116_v1/dependencies`, PyTorch/NumPy state mapping, pytest.

## Global Constraints

- Use `runs/forward_feasibility3/pose.json` as the only initial pose source for this scene; do not substitute the 10-21 pose.
- Panda uses IsaacLab's native Franka/Panda articulation; DH116 left and right are independent world-root USD articulations whose roots follow the corresponding native `panda_hand` pose.
- Do not load `assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd` in the new scene.
- Do not call `visual_pose_sync` or write transforms directly to individual robot links.
- Do not run approach; the stage sequence starts at the saved close state and proceeds close→hold→lift.
- Keep one Isaac instance and one CPU process; do not delete or overwrite existing assets or runs.
- A pose preview is not a physics success claim; each stage must emit its own evidence.

---

### Task 1: Define the native scene contract

**Files:**
- Create: `src/go2_pvcnn/control/dh116/native_scene.py`
- Test: `tests/test_dh116_native_scene.py`

**Interfaces:**
- `NativeSceneSpec(root_path: str, panda_paths: dict[str, str], hand_paths: dict[str, str], hand_usd: dict[str, Path], flange_link: str)` — immutable scene identity and paths.
- `candidate3_pose(path: Path) -> dict[str, np.ndarray]` — loads and validates the two 13-DOF side vectors from `runs/forward_feasibility3/pose.json`.
- `validate_scene_spec(spec: NativeSceneSpec) -> None` — rejects combined-USD paths, missing sides, and identical left/right paths.

- [ ] **Step 1: Write the failing tests**

```python
def test_candidate3_loader_returns_distinct_left_and_right_states(tmp_path):
    pose = candidate3_pose(ROOT / "runs/forward_feasibility3/pose.json")
    assert pose["left"].shape == (13,)
    assert pose["right"].shape == (13,)
    assert not np.array_equal(pose["left"], pose["right"])

def test_scene_spec_rejects_combined_asset_and_duplicate_hands():
    spec = NativeSceneSpec(
        "/World/DH116DualPandaScene",
        {"left": "/World/LeftPanda", "right": "/World/RightPanda"},
        {"left": "/World/LeftDH116", "right": "/World/RightDH116"},
        {"left": COMBINED_USD, "right": COMBINED_USD},
        "panda_link8",
    )
    with pytest.raises(ValueError, match="combined|distinct"):
        validate_scene_spec(spec)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_scene.py -q`

Expected: FAIL because the new contract functions do not exist.

- [ ] **Step 3: Implement the contract and candidate3 loader**

Use `json.loads`, require `schema == "dh116-static-pose-v1"`, require both `states.left` and `states.right`, convert with `np.asarray(..., dtype=np.float64)`, and reject non-finite values or shapes other than `(13,)`. `validate_scene_spec` must compare resolved hand paths and reject any filename containing `fixed_dual_panda_dh116.usd`.

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_scene.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/go2_pvcnn/control/dh116/native_scene.py tests/test_dh116_native_scene.py
git commit -m "test: define native dual Panda DH116 scene contract"
```

### Task 2: Compose Isaac-native Panda and separate DH116 assets

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/native_scene.py`
- Create: `scripts/preview_native_dual_panda_dh116.py`
- Test: `tests/test_dh116_native_scene.py`

**Interfaces:**
- `build_native_scene(sim, spec, candidate_pose, table_cfg, wire_cfg) -> NativeSceneRuntime` — spawns two native Panda articulations and two DH116 USD references.
- `NativeSceneRuntime.articulations: dict[str, Articulation]` — `left` and `right` native Panda objects.
- `NativeSceneRuntime.assert_visual_contract(stage) -> dict` — returns resolved prim paths and rejects missing/duplicated assets.

- [ ] **Step 1: Add composition tests before Isaac startup**

```python
def test_native_scene_uses_two_distinct_dh116_usd_paths():
    spec = default_native_scene_spec(ROOT)
    assert spec.hand_usd["left"].name == "DH116_left.usd"
    assert spec.hand_usd["right"].name == "DH116_right.usd"
    assert spec.hand_usd["left"] != spec.hand_usd["right"]
    assert spec.flange_link == "panda_link8"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_scene.py::test_native_scene_uses_two_distinct_dh116_usd_paths -q`

Expected: FAIL because `default_native_scene_spec` is not defined.

- [ ] **Step 3: Implement explicit composition**

Clone `isaaclab_assets.robots.franka.FRANKA_PANDA_CFG` twice with prim paths `/World/LeftPanda` and `/World/RightPanda`, use the candidate3 first seven joints for each articulation, and disable the built-in Panda gripper actuators/visibility so no second hand is displayed. Spawn `DH116_left.usd` and `DH116_right.usd` as separate world-root references at `/World/LeftDH116` and `/World/RightDH116`, create independent DH116 articulations, and update only each hand root prim to the corresponding native `panda_hand` world pose. The new script must never import the combined USD path, create nested articulations, or call `visual_pose_sync`.

- [ ] **Step 4: Run composition tests and compile**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_scene.py -q && python -m py_compile scripts/preview_native_dual_panda_dh116.py src/go2_pvcnn/control/dh116/native_scene.py`

Expected: PASS and no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add src/go2_pvcnn/control/dh116/native_scene.py scripts/preview_native_dual_panda_dh116.py tests/test_dh116_native_scene.py
git commit -m "feat: compose native Panda arms with separate DH116 hands"
```

### Task 3: Add candidate3 direct close→hold→lift controller

**Files:**
- Create: `src/go2_pvcnn/control/dh116/native_stage_controller.py`
- Modify: `scripts/preview_native_dual_panda_dh116.py`
- Test: `tests/test_dh116_native_stage_controller.py`

**Interfaces:**
- `NativeStageController(start_stage: str = "close")` — accepts only `close`, `hold`, or `lift`; rejects `approach`.
- `step(observation: NativeObservation) -> NativeCommand` — returns per-side Panda and DH116 commands without copying one side into the other.
- `stage_evidence() -> dict` — reports stage, contact duration, hold duration, lift displacement, and failure reason.

- [ ] **Step 1: Write failing stage tests**

```python
def test_controller_starts_at_close_and_never_emits_approach():
    controller = NativeStageController("close")
    command = controller.step(observation_at_close())
    assert command.stage == "close"
    assert command.stage != "approach"

def test_lift_moves_both_pandas_by_equal_positive_z_delta():
    controller = NativeStageController("lift")
    command = controller.step(observation_with_stable_hold())
    assert command.panda_delta_z["left"] > 0.0
    assert command.panda_delta_z["left"] == command.panda_delta_z["right"]
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_stage_controller.py -q`

Expected: FAIL because the controller module is absent.

- [ ] **Step 3: Implement the minimal stage machine**

Use candidate3 as the immutable base state; close updates only each side's 12 active DH116 coordinates, hold freezes Panda targets and requires 1 s stable contact, and lift adds the same positive world-Z Panda displacement to both sides while handing the previous DH116 grasp target through unchanged. If contact or slip fails, set `failure_reason`, stop advancing, and serialize the current state.

- [ ] **Step 4: Run stage tests**

Run: `PYTHONPATH=src pytest tests/test_dh116_native_stage_controller.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/go2_pvcnn/control/dh116/native_stage_controller.py scripts/preview_native_dual_panda_dh116.py tests/test_dh116_native_stage_controller.py
git commit -m "feat: start DH116 native scene at close and lift symmetrically"
```

### Task 4: Run the visible candidate3 pose and physical stages

**Files:**
- Modify: `scripts/preview_native_dual_panda_dh116.py`
- Create: `runs/native_dual_panda_candidate3/` (generated output only; do not commit)

- [ ] **Step 1: Run the static target-pose gate**

```bash
cd "/home/xk/coding/dexterous hand"
PYTHONPATH=src /home/xk/miniconda3/envs/go2/bin/python scripts/preview_native_dual_panda_dh116.py \
  --pose runs/forward_feasibility3/pose.json \
  --output runs/native_dual_panda_candidate3 \
  --stage static
```

Expected: Isaac opens with two distinct Panda arms, two distinct DH116 hands, table, and wire. The output must include a screenshot and `static_pose.json` with both side paths and the candidate3 SHA256.

- [ ] **Step 2: Inspect the screenshot before motion**

Check that both arms are visible, neither arm is duplicated, both palms point down, both finger groups point forward, and the wire is on the tabletop. If any check fails, stop without starting close.

- [ ] **Step 3: Run direct close→hold→lift**

```bash
PYTHONPATH=src /home/xk/miniconda3/envs/go2/bin/python scripts/preview_native_dual_panda_dh116.py \
  --pose runs/forward_feasibility3/pose.json \
  --output runs/native_dual_panda_candidate3 \
  --stage close --run-hold --run-lift
```

Expected: one Isaac instance advances through close, hold, and lift or stops at the first failed gate; it writes `trajectory.jsonl`, `summary.json`, and a final screenshot. No approach frames are allowed in the trajectory.

- [ ] **Step 4: Verify evidence**

Run: `python - <<'PY'` to load `summary.json` and assert `asset_mode == "native_panda_plus_separate_dh116"`, `initial_pose_source` ends with `runs/forward_feasibility3/pose.json`, and `stages[0] == "close"`. Report the actual stage reached and do not call lift successful unless the 30 mm/1 s/wire-clearance conditions are all true.

- [ ] **Step 5: Commit only source changes**

```bash
git add scripts/preview_native_dual_panda_dh116.py src/go2_pvcnn/control/dh116/native_stage_controller.py
git commit -m "feat: add visible candidate3 close hold lift runner"
```

Generated Isaac screenshots, logs, and runs stay uncommitted.

## Self-Review Checklist

- Candidate3 is used everywhere; no 10-21 pose is referenced.
- The new scene has two native Panda articulations and two distinct DH116 USD paths.
- Combined USD and manual per-link transform writes are excluded.
- The stage machine starts at close and never emits approach.
- Close, hold, and lift have separate evidence and failure stopping rules.
- Tests cover asset identity, left/right separation, stage entry, and symmetric lift.
