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
