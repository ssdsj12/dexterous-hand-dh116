# Numeric CUDA-view reset state correction

Base b0ce300c91b45f40c577cd340fd8482ff2f33c70; independent branch codex/mesh-reset-numeric-state-20261114 and worktree /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-reset-numeric-state-20261114. This package adds one opt-in Python module, CPU tests and this report. No frozen source/recipe changes, backend compilation, Isaac launch, solver step, training or publication.

## Concrete failure and source diagnosis

The consumed mesh-reset Pinch attempt stopped at SDK2, actual time 0.004166666883975267 s, before any measured row. The error was mesh-only reset changed rod nodes/velocity/directors/parameters.

Original output: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-mesh-reset-pinch-20261114-pt9_jx1f/native-runs/flex.f8-mesh-reset-health-projected-pinch-native/attempt-kjuzues5/native.

| Original file | SHA256 |
| --- | --- |
| result.json | 8eff90e46cf772fc520fb790354ec9990ea5abc5757a992154e5782861aba4d5 |
| physical.jsonl | 101e3b526b035d644d663876d848f96b6f1b8595f968bc632347a2e1f9d891ed |
| mesh_reset_state.json | e2f5302ed2dabbb3916b438fb7e3c74bfaa6f7c96e1b4ab5133fdf6794983e41 |

The failure artifact has after=None and completed_reset_receipt=None. Internal before/after invariant states were discarded by v1 finally cleanup. It cannot establish whether native numeric rod state changed; missing values remain unknown.

The actual compiled 881701 source has:
- csrc/bindings.cpp:715–732: CudaView exposes CUDA Array Interface, ptr and shape, without NumPy array/buffer conversion.
- bindings.cpp:846–860: positions, velocities and quaternions each return a fresh CudaView over existing device output.
- csrc/batch.h:79–81: getters expose d_outPos, d_outVel and d_outQuat.
- bindings.cpp SHA256 a3c87883962663358c90ccbacdb85596da965d979e8d12baef9c32bf90e30e09.

Frozen tabletop_mesh_reset_order._state:214–221 passes these views into tabletop_deformable_bimanual_step._owned. That helper only handles an existing .detach() specially; otherwise np.array(value, copy=True) creates a zero-dimensional object array for this protocol. Frozen _encoded hashes ndarray.tobytes(): for object arrays these are wrapper-pointer bytes, not device numeric data. The first arrays retain their wrappers, so later fresh views have different identities.

A pure CPU reproduction executes these exact frozen Python functions with fresh source-shaped CAI wrappers over the same fake pointer/metadata. All p/v/q fields are dtype object, shape [], and their hashes differ; parameter hashes remain equal. No .so/Isaac import or device allocation occurs. This proves a Python representation bug capable of triggering the consumed guard; it does not reconstruct missing native numeric values.

Independent C++ read-only review found no direct contact-parameter mutation in clear/add/write_meta. sdf.cu:247–265 writes SDF/pair metadata, not radius/friction/gravity/time. Fixing Python conversion requires no new backend compilation or diagnostic-only native rerun.

## Independent repair and API

New module: src/go2_pvcnn/control/dh116/tabletop_mesh_reset_numeric_state.py.

- numeric_reset_state_scope(*, sim, rod, record=None): verifies actual loaded source/instance, temporarily binds only frozen reset_order._state, yields an owned audit, restores the old function on every exit, then invokes record(owned_audit) once.
- validate_numeric_reset_audit(audit, *, reset_receipt): independently decodes disk data and validates sources, owner, actual clocks, both raw states against original mesh_reset.native_state_before/native_state_after, and direct mesh-readback owner.
- state_differences(before, after): per-field exact byte equality, numeric equality, maximum absolute difference, changed-element count and signed-zero-only classification.

The existing material_director_readback._copy_endpoint supplies p/q conversion: explicit CUDA device/default stream, synchronization without stepping, torch.as_tensor(CudaView), detach/clone, blocking CPU copy and owned NumPy arrays. Velocity follows the same real conversion. Strict shapes/dtype are float32 (1,41,3) positions/velocities and (1,41,4) quaternions, all finite. Forty quaternion entries are segment directors; the last is finite padding, not an extra material frame.

parameter_snapshot remains the real owned getter and its original schema validator. Each capture brackets actual SDK/runtime/native solver time. Two calls cover the invariant interval. The second actual clock retains clock_valid=False during mesh mutation. Actual timestamps are not rounded. No reset/gather/commit/physical step is added.

The old bitwise invariant guard remains unchanged. Actual p/v/q, radius and signed-zero byte changes still trigger its original error. Signed-zero numeric equivalence is only explanatory; no tolerance replaces the guard. Object arrays, wrong shape/dtype, nonfinite values, changed/default-stream violations, clock drift and solver errors never become valid numeric evidence.

Evidence is bounded to two samples, 64 numeric leaves and 1024 elements per float32 array. When both captures complete, raw owned state and every field difference survive even if the original guard raises. An incomplete capture retains an error and completed-field names; it does not invent raw values. Native aliases and capturing callbacks are cleared from exception frames. Recorder failure propagates after reader restoration and does not perform another physics mutation/reset.

The successor wrapper must surround its actual reset transaction with this scope and durably write mesh_reset_numeric_state.json. It must fingerprint this module, its original copy-helper dependency and tests. The consumed v1 recipe remains unchanged. Any invalid numeric audit raises; the validator never merely returns a false flag. Success returns numeric_state_unchanged while native_qualification and teacher_admission remain false.

## CPU verification

Final exact command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -x -q tests/test_mesh_reset_numeric_state.py tests/test_mesh_reset_order.py tests/test_mesh_observation_profile.py tests/test_dh116_material_director_readback.py tests/test_dh116_native_parameter_endpoints.py

Actual result: 130 passed in 29.75 seconds (23 new, 107 original regressions). The actual RebuiltMeshReset Python transaction runs with explicitly simulated CAI/Torch transport. There is no native/CUDA allocation or Isaac success claim. Tests cover the original object-pointer hash, owned copies and disk codec, unchanged guard under real value/parameter/signed-zero changes, shape/dtype/nonfinite/stream/clock errors, nesting refusal, disk tampering, writer failure and native-alias cleanup.

The older reset fixture had three nodes. Initial setup failed (1 reproduction pass, 19 fixture errors); the new fixture was upgraded explicitly to 41 nodes and real parameter shapes. A later NaN case was rejected by the original reset capture before the new getter; its assertion now retains operation_error with zero samples, rather than inventing capture_error. These failures remain archived. Root review also caught verify_runtime returning None; final code invokes it separately and stores source_basis(), tested with the real verify_runtime function. No production shape/source requirement was relaxed.

Evidence directory: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-reset-numeric-cpu-ig3ec4ll.

| Evidence | Actual result | SHA256 |
| --- | --- | --- |
| red.log | reproduction passed; 19 setup errors retained | 68e98dfc4f7611d0032683caf195b4a2af08b32b806bf7def3f3189cdc7e1ceb |
| second.log | 9 passed / 1 fixture assertion failure | 834821f5cfcdd20b6d27379ea34d78d973d83f3eba242ea2447af70437ac8e14 |
| third.log | 23 passed / 6.47 s | d13d69ecbc1c50b3b59a46aa134c70d7cd2c08d01e82a044bfb136d428dd3f9c |
| regression.log | 130 passed / 29.75 s | 0eba6b009082b39980d82f3f02d1dbfa3a46f7c8e80a8c3bdd9ff27a059e9c2f |
| cai-reproduction.json | old Python CAI conversion reproduced, no native import | 5e38bb23bafa8948c4c601f87be5d23ac62442bc9a87d28d70d9e6026d6c95f3 |
| protection.json | 528 protected files + 2 Checkpoints passed | 51d623a63a12eb6c55584334fb821d2859bfcc8dc4f3b7e66bda222a55f6f16c |

Module SHA256 e02853b0e7be2d38f9887d35549f335430fc92ae7380857be931d07b7dda0346.
Test SHA256 869262fcfd40a56f3759d16cc65a476b9a79e8b40f032790c43d9cc06be81d54.

A future separately frozen run must still verify actual numeric invariance, eleven mesh current/target/solver poses, raw pairs, health and process exit. This package claims CPU/source repair preparation only.
