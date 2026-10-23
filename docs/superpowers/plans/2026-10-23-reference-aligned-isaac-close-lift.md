# Reference-Aligned Isaac Close/Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Isaac scene use the reference image's dual-Panda/DH116/world-wire layout and produce real `close → hold → lift` evidence with matching camera views.

**Architecture:** A small reference-view module owns the shared world layout and perspective/top/side camera presets. The native Isaac preview consumes those constants for its table, wire, mounting and stage screenshots; the integrated physics runner consumes the same perspective camera and writes stage snapshots without changing its physical guards. The stage runner moves both Panda roots together during lift and records the actual stage reached.

**Tech Stack:** Python, NumPy, IsaacLab/Isaac Sim USD, pytest, existing DH116 evidence JSON.

## Global Constraints

- Preserve `runs/forward_feasibility3/pose.json` as the candidate3 pose source.
- Keep the wire along world `+Y` at `x=0.43`, tabletop top near `z=0.30`, and palms/fingers in the reference orientation.
- Keep left/right Panda and DH116 identities distinct; never reuse one side's state for the other.
- Do not claim physics or grasp success from a static preview; report the first failed close/hold/lift gate.
- Keep one Isaac instance, one CPU process, and do not overwrite existing runs/assets.

---

### Task 1: Shared reference layout and camera contract

**Files:**
- Create: `src/go2_pvcnn/control/dh116/reference_view.py`
- Create: `tests/test_dh116_reference_view.py`

**Interfaces:**
- `REFERENCE_TABLE_CENTER`, `REFERENCE_TABLE_SIZE`, `REFERENCE_WIRE_START`, `REFERENCE_WIRE_END` — immutable numeric tuples.
- `camera_preset(name: str) -> CameraPreset` for `perspective`, `top`, and `side`.
- `reference_wire_points(count: int = 41) -> np.ndarray` returning `(3, count)` points.

- [ ] **Step 1: Write the failing tests** for the wire axis/height, three camera names, and finite camera vectors.
- [ ] **Step 2: Run `PYTHONPATH=src pytest tests/test_dh116_reference_view.py -q` and confirm the missing-module failure.**
- [ ] **Step 3: Implement the constants and validation, rejecting unknown camera names and counts below two.**
- [ ] **Step 4: Re-run the focused tests and confirm they pass.**

### Task 2: Reference-aligned native Isaac preview and stage sequence

**Files:**
- Modify: `scripts/preview_native_dual_panda_dh116.py`
- Modify: `src/go2_pvcnn/control/dh116/native_scene.py`
- Modify: `src/go2_pvcnn/control/dh116/native_stage_controller.py`
- Create: `tests/test_dh116_reference_runner.py`

**Interfaces:**
- `--view {perspective,top,side}` and `--capture-views` select the shared camera presets.
- `--sequence` runs close, hold, and lift in one Isaac process and writes per-stage screenshots/JSON.
- `NativeSceneRuntime.apply_root_delta_z(delta_z)` moves both Panda/DH116 roots in world Z while preserving each side's independent hand targets.

- [ ] **Step 1: Add failing source-contract tests** requiring the shared camera module, reference wire points, explicit `--sequence`, and a root-Z lift path.
- [ ] **Step 2: Run the focused test and confirm it fails against the current script.**
- [ ] **Step 3: Implement shared table/wire/camera usage, per-view capture, sequence timing, and synchronized root translation.**
- [ ] **Step 4: Run focused tests and `python -m py_compile` on the changed modules.**

### Task 3: Use the reference camera in the real integrated physics runner

**Files:**
- Modify: `scripts/verify_fixed_dual_panda_dh116.py`
- Modify: `tests/test_dh116_gui_render.py`

- [ ] **Step 1: Add a failing AST test** requiring `camera_preset("perspective")` and stage snapshot paths in the integrated runner.
- [ ] **Step 2: Run the test to verify it fails.**
- [ ] **Step 3: Replace the hard-coded camera tuple with the shared perspective preset and schedule stage-labelled captures at close/hold/lift transitions.**
- [ ] **Step 4: Run the GUI/render-focused tests and compile the runner.**

### Task 4: Run and inspect the actual Isaac outputs

**Files:**
- Generated only: `runs/reference_aligned_native_20261023/`, `runs/reference_aligned_physics_20261023/`

- [ ] **Step 1: Run the native candidate3 sequence in one Isaac instance with all three reference views.**
- [ ] **Step 2: Inspect the perspective/top/side captures for the reference layout before interpreting stages.**
- [ ] **Step 3: Run the integrated physical close→hold→lift command using the existing pinned assets/guards and the same perspective camera.**
- [ ] **Step 4: Read both summaries and report the actual stage reached, first failed criterion, and file paths; never promote a failed lift to success.**

## Verification checklist

- Shared constants match the reference world frame and wire axis.
- Isaac perspective/top/side images show both distinct Panda arms and DH116 hands in the same layout.
- Close/hold/lift evidence is from the current run, not copied from historical runs.
- Tests and compile checks are fresh before reporting results.
