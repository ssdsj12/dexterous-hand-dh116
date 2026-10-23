# Candidate 3 MPC Pose Lock Implementation Plan

> **For agentic workers:** Inline execution is selected; execute each task with a failing test, minimal implementation, and verification.

**Goal:** Make candidate 3 the only Close→Hold→Lift MPC starting configuration and keep planned hand commands inside a bounded, non-singular local neighborhood of that pose.

**Architecture:** A strict pose-lock helper loads the reviewed pose artifact and exposes the 12 active DH116 reference coordinates. Parallel contact MPC receives that reference on every planning call, intersects its command limits with a local neighborhood, and reports the lock in diagnostics. Existing callers that do not opt into a lock retain their current behavior for compatibility.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing DH116 pose artifact and contact MPC modules.

## Global Constraints

- Candidate 3 remains an offline geometry candidate until physical clearance and contact checks pass.
- No Approach phase is imported or credited; the lock only initializes a fresh Close-stage episode.
- Panda coordinates remain outside the 12D Hand MPC action.
- Commands must remain finite, within DH116 active limits, and within the configured local reference neighborhood.

### Task 1: Strict candidate pose lock

**Files:**
- Create: `src/go2_pvcnn/control/dh116/pose_lock.py`
- Test: `tests/test_dh116_pose_lock.py`

**Interfaces:**
- `load_pose_lock(path, sha256, asset_sha256, layers, tolerance_rad=0.12) -> PoseLock`
- `PoseLock.active_reference -> np.ndarray` with shape `(12,)`
- `PoseLock.active_bounds(lower, upper) -> tuple[np.ndarray, np.ndarray]`

- [x] Write tests for valid loading, hash/identity rejection, and bounded active neighborhoods.
- [x] Run the focused tests and observe the expected missing-module failure.
- [x] Implement strict JSON/schema validation using `load_active_pose` and expose reference/bounds.
- [x] Run focused tests until green.

### Task 2: Pose-locked contact MPC

**Files:**
- Modify: `src/go2_pvcnn/control/dh116/contact_mpc.py`
- Modify: `src/go2_pvcnn/control/dh116/integrated_hand_mpc.py`
- Test: `tests/test_dh116_contact_mpc.py`

**Interfaces:**
- `ParallelContactMPC.plan_contact(..., reference_active_joints=None, reference_tolerance_rad=None)`
- `IntegratedHandMPC.plan(..., reference_active_joints=None, reference_tolerance_rad=None)`

- [x] Add a failing test proving a command cannot leave the local candidate neighborhood.
- [x] Add reference-bound projection and a diagnostic `pose_lock` record.
- [x] Keep Panda out of the action and preserve existing unlocked behavior.
- [x] Run the focused MPC tests.

### Task 3: Wire the lock into fresh direct-Close execution

**Files:**
- Modify: `scripts/verify_fixed_dual_panda_dh116.py`
- Test: `tests/test_dh116_sequence_runtime.py`

- [x] Add a failing source-contract test requiring the pose reference to be passed to contact MPC for Close, Hold, and Lift.
- [x] Load the pinned candidate pose for a fresh `--grasp-entry close` run, validate identity, initialize both sides from it, and pass the active reference into every hand-MPC call.
- [x] Record pose hash and local-bound diagnostics in the trajectory manifest.
- [x] Run source-contract and focused unit tests; do not start a long physics run until these pass.

### Task 4: Verification artifact

- [x] Generate a pose-lock record in the run manifest with asset/pose hashes and reference bounds.
- [x] Run focused tests and a short direct-Close dry initialization.
- [x] Confirm no old V23 trajectory or Approach frame is used by the new command.

### Follow-up P0 fixes completed in this continuation

- [x] `panda_feedback_increment` adds measured palm position feedback, damped solve, residual/reachability reports, velocity caps, and joint-limit clipping.
- [x] `GraspEvidenceGate` keeps Lift in progress until whole-wire clearance is continuously present for the dwell.
- [x] `CompleteContactPolicy`, `ParallelContactMPC`, and `IntegratedHandMPC` expose episode reset hooks.
- [ ] Candidate 3 still needs a contact-preserving pose correction; clearance-only random candidates are not promoted because their pad contact is not validated.
