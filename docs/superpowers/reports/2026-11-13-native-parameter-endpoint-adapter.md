# 2026-11-13 — Completed-boundary native parameter adapter

Base: 4ccea90e183e2a7f003b787e883bd845b4a67f35.
Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/native-parameter-runtime-20261113.

## Result and scope

The new opt-in NativeParameterEndpoints implements actual completed-reset/480Hz endpoint readback for the isolated parameter-getter backend. It reuses the existing reset/episode sequencing guard, while overriding the old binary-locked read implementation. No original source, default backend, controller, reward or model has been replaced.

The source lock enumerates all 72 tracked DeformX/YarnBall files. Inventory SHA256:
3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472.
Every new source file is verified; every non-bindings file must also equal the corresponding original pair-backend file. Removing the exact reviewed getter from bindings.cpp must reproduce the old post-pair bindings SHA. This verifies the read-only delta instead of claiming a general API or solver upgrade.

Standalone native probe qualification is separately bound to receipt SHA:
85f0b6e89c25eaf9f8e8d3d003eab1bc91cd7b2376459f969b0b66b79ee63ad5.
It completed three owned CUDA snapshots, 0 dynamics/PhysX/PPO steps, outer natural exit 0 in 2.897244 seconds. That is historical evidence of the new getter, **not** this adapter's Isaac task qualification.

## Runtime checks actually implemented

- Exact isolated backend file/SHA and RopeBatch type; 1 environment, 41 nodes, free root orientation.
- Same rod owner via weak reference, not an owning native reference.
- Same run/env/episode/reset/rod identity, one SDK step and 12 rod substeps per subsequent endpoint; a failed capture poisons the sequence until a fresh actual reset.
- SDK step/time, solver time and solver error checked before and after the complete read. Explicit default CUDA stream and blocking owned endpoint copies.
- Device rest lengths reconciled to the actual float32 reset nodes, native inverse masses reconciled to the unchanged linear density, no hidden fixed nodes.
- Native radius, gravity, friction, contact kappa/slip scale, EA and EI reconciled to the unchanged existing candidate configuration. Configuration is only the comparison target, not the reported source.
- Native host metadata, device copies and derived quantities retain separate provenance; uncertainty remains unknown.
- The returned radius sample carries its own full clock, backend SHA and unqualified task-physics status. No existing R0 consumer is automatically switched to it.
- Endpoint directors are retained as endpoint measurements only. Continuous contact/material station and tangential slip remain unavailable.

## Files/functions

- src/go2_pvcnn/control/dh116/tabletop_native_parameter_endpoints.py: source_basis, _locked_source_tree, NativeParameterEndpoints._read.
- docs/contracts/deformx_parameter_getter_source_lock_v1.json: complete pinned source inventory.
- tests/test_dh116_native_parameter_endpoints.py: sequence, provenance, parameter, fixed-node, stream, retention and real source-lock checks.

## Tests

New tests first failed on the absent module; log:
evidence/native-parameter-endpoints-red-hdSSAN/pytest.log.
The full-source-inventory test then failed before the inventory validation was added.

123 CPU tests passed, 0 failed, exit 0 in 7.49s:
tests/test_dh116_native_parameter_endpoints.py
tests/test_dh116_material_director_readback.py
tests/test_dh116_deformx_parameter_getters.py
tests/test_flexible_parameter_readback.py
tests/test_dh116_parameter_packet_validation.py

Command prefix:
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q

Full log:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-parameter-endpoints-regression-gZzmr7/pytest.log

CPU fixtures do not count as physical records. This work package executed zero new native steps and zero PPO updates.

## Next dependency

Prepare a separately versioned physical-owner integration and post-exit auditor for this adapter and the new backend. Preserve the old task/control path, fixed reset profile and 2+8+8 bounded schedule. Before a new run, freeze the new source/config/binary fingerprints and standing-authorization record; verify resource lock, natural exit and every raw endpoint. Do not rerun the consumed zero-step probe.

R0 coefficient/domain qualification, bilateral contact, material anchors and complete Teacher state sources remain blocked. Pinch/Tripod/Power Teacher admission remains false. Passing this adapter's CPU tests does not qualify R0 or training.
