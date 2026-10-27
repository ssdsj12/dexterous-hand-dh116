# Task 2 report — DH116 middle-finger contact geometry

Status: DONE

Calibration already exposed `finger33_link`; the builder and selector now
carry that digit through the same pinned source/layer SHA path as thumb and
index. A new `symmetric_grasp_middle_v1` bundle was generated from the fixed
dual-Panda DH116 USD with six references and six quantized convex-hull contact
meshes. The bundle explicitly remains simulation geometry only:
`physics_contact_qualified=false`, `hardware_calibrated=false`.

Verification:

```text
PYTHONPATH=src bash scripts/python.sh -m pytest -q \
  tests/test_dh116_middle_hook_geometry.py tests/test_dh116_usd_contract.py
3 passed
```

The generated middle entries use `finger33_link`, finite unit normals, and
record source mesh, derived mesh, reference, layer, and asset SHA values.

## Task 2 — separate close and lift vertical force channels

Status: DONE

Implemented `signed_hand_lift_vertical_force`, routed `TensorEnv.step` through
normal-only close force and combined normal-plus-friction lift force, and added
`TensorTask.consume(..., lift_vertical_force_n=...)` validation and lift gating.
Close readiness, continuous close scoring, downward impulse/force penalties,
and close transition penalties continue to use `vertical_force_n`.

Verification:

```text
bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_task.py -k \
  'lift_force_channel or lift_success_uses_combined'
2 passed, 14 deselected

bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_task.py
15 passed, 1 failed
```

The remaining tensor-task failure is the existing scalar-vs-batch control
comparison (`test_observation_matches_scalar_and_control_matches_batch`),
which predates this force-channel change and reflects synchronized palm work
already present in the shared dirty tree. Broader close-downward tests also
retain unrelated baseline failures in the shared tree.

Commit: `73495c2` (`feat: separate close and lift vertical force channels`)

### Reviewer fix — close grasp-ready channel

Status: FIXED

The close-phase `grasp_ready` event now requires `vertical_force_n > 0`, keeping
friction-inclusive `lift_vertical_force_n` restricted to phase-3 lift
stability/success checks. Added a regression covering negative normal force
with positive combined lift force; it receives no close grasp-ready credit.

Verification:

```text
bash scripts/python.sh -m pytest -q tests/test_dh116_vertical_reward.py -k \
  'close_grasp_ready_requires_positive_normal_force or signed_force_replaces_unsigned'
8 passed, 8 deselected

bash scripts/python.sh -m pytest -q tests/test_dh116_tensor_task.py -k \
  'lift_force_channel or lift_success_uses_combined'
2 passed, 14 deselected
```
