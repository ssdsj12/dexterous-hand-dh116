# Task 3 report — PPO force range and checkpoint/probe contracts

Status: IMPLEMENTED

The tensor and coordinated grasp controllers now map policy force channels over the explicit 2–12 N range, clamp requests to that range (while respecting a lower configured mechanical maximum), and retain the existing slew/governor behavior. The probe encoder uses the same 2 N floor, rejects nonfinite/too-low/too-high targets, and records normal-only force telemetry with friction as a separate field.

Training contracts now record `force_action_range_n=[2.0,12.0]`, the phase-specific close/hold/lift force semantics, the initial force-action bias and expected target, and synchronized palm exploration version/bounds/weight. `checkpoint_contract_changes` reports the named `force_action_range_2_to_12` incompatibility for old `[0.1,12.0]` contracts; ordinary resume rejects that migration with the existing contract-mismatch error. Existing inference-only teacher probe migration for the force-source contract remains explicit.

Verification:

```text
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
  tests/test_dh116_control_probe.py::test_force_target_roundtrip_and_bounds \
  tests/test_dh116_tensor_training.py::test_old_force_range_is_not_resume_compatible
2 passed

DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
  tests/test_dh116_tensor_task.py -k 'control or force' \
  tests/test_dh116_control_probe.py -k 'force or probe_contract'
9 passed, 23 deselected

python3 -m py_compile src/go2_pvcnn/control/dh116/tabletop_tensor_task.py \
  src/go2_pvcnn/control/dh116/tabletop_coordinated_grasp.py \
  src/go2_pvcnn/control/dh116/tabletop_control_probe.py \
  src/go2_pvcnn/control/dh116/tabletop_tensor_training.py
```

The broader requested tensor-training filter has one pre-existing shared-tree failure: `test_close_force_behavior_requires_new_controller_contract` expects the separate close-penalty v4 contract while this checkout still defines v3. The Task 3 force-range and resume tests pass.

Fix review evidence (2026-10-27):

- `BatchTask.control` now uses the PPO force policy constants (`POLICY_MIN_FORCE_N=2.0`, `POLICY_MAX_FORCE_N=12.0`) for full-hand GPU batches, with request clamping to the policy range and configured mechanical maximum. Legacy low-capacity pad-load profiles retain differential behavior.
- Added `test_batch_force_policy_action_maps_to_two_to_twelve_newtons`, asserting batched actions -1 and +1 request exactly 2 N and 12 N.
- Restored `tests/test_dh116_tensor_training.py::test_close_force_behavior_requires_new_controller_contract` to the source-defined `close_downward_impulse_reset_v3` contract.

Verification:

```text
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
  tests/test_dh116_batched_task.py tests/test_dh116_tabletop_object_contact.py \
  tests/test_dh116_tensor_training.py
53 passed, 1 warning

DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
  tests/test_dh116_batched_task.py::test_batch_force_policy_action_maps_to_two_to_twelve_newtons \
  tests/test_dh116_tensor_training.py::test_close_force_behavior_requires_new_controller_contract
2 passed

python3 -m py_compile src/go2_pvcnn/control/dh116/tabletop_batched_task.py
```

Instant-force regression fix (2026-10-27):

- Full-hand tensor and batched controllers now seed and reset `f_target`/`f_requested` at the 2 N policy floor when `max_force_n >= 2`; legacy low-capacity profiles retain the 0.1 N seed.
- Updated tensor action mapping assertions to the 2–12 N range, including inactive-row and post-reset targets. Scalar `ReactionForceController` assertions remain at 0.1–12 N.

Verification (CUDA was available; the parametrized file exercised both CPU and CUDA):

```text
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q tests/test_dh116_instant_force_target.py
4 passed

DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
  tests/test_dh116_tensor_task.py -k 'control or force' \
  tests/test_dh116_control_probe.py -k 'force or probe_contract' \
  tests/test_dh116_batched_task.py
12 passed, 41 deselected
```
