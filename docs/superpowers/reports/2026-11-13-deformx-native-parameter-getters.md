# 2026-11-13 — Independent native solver parameter getters

Base: 9c4f25551d23e4deb91627518003d1c07a5bde2d. Ubuntu worktree: parameter-native-getters-20261113. Scope: optional read-only backend extension and zero-step CUDA verification preparation. No original controller, radius, coefficients, SDK limits, checkpoint or default backend changed.

## Actual change

- scripts/prepare_dh116_deformx_parameter_getters.py validates a clean, separate DeformX worktree at 96b8514c7c55c724f1abba1fbf6ed6642b7db72b, applies the existing pair-readback delta, verifies the exact post-pair bindings SHA, then inserts RopeBatch.parameter_snapshot(). Every source anchor is checked before writes.
- The C++ getter locks the batch, synchronizes its solver stream, takes owned blocking D2H copies of d_verts AND d_qRests, and copies host metadata. It never steps, resets, uploads, returns native views or changes physics.
- Critical source finding: upstream Sim::download() copies vertices, velocities and quaternions, **not d_qRests**. Therefore a host qRests mirror was deliberately not used as evidence of current device constitutive coefficients.
- Parameter provenance: contactRadius, gravity and contact friction are native host metadata; rest lengths, inverse masses and raw constitutive coefficients are device copies. EA=kStretch/lRest and EI=norm(qRest)*lRest/4 are explicitly derived values. The trailing node is excluded from segment arrays. These are simulation parameters, not measured hardware/material calibration.
- tabletop_deformx_parameter_snapshot.validate_snapshot() checks shape, exact solver clock, provenance, finite/positive values, fixed nodes and EA/EI reconciliation. Missing/calibrated uncertainty is not invented; teacher_qualified remains false.
- build_dh116_deformx_parameter_getters.py reuses the locked CUDA build tools and existing source inventory audit. Exact path trust is passed only in the audit child environment; global Git settings remain unchanged.
- verify_dh116_deformx_parameter_getters.py and its worker reuse the already tested native_bounded transport, unique intent identity and single-instance lock. Raw packets are saved before validation and independently reopened. CPU subprocess tests verify mismatched HEAD/fingerprint/output/GPU rejection.
- The new opt-in recipe constructs one three-node free rod and reads three snapshots, including a mutation of the first returned host arrays to verify isolation. It has zero solver integration steps, zero PhysX steps and zero PPO updates. Gather/readback kernels are not counted as dynamics steps. This is not contact or grasp qualification.

## Build actually completed

Independent source: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu/source
Independent build/package: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu/build

Python 3.10.12; PyTorch 2.7.0+cu128; nvcc 12.8.61; C++/CUDA17; SM89.
Extension SHA256: 2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec
Source delta SHA256: 0bf1ed22749f11049e8d6a83da6e024a685d8482f3e2208a1bcc163705d80ed8

Configure, compile, install and cuobjdump SM89 checks exited 0. Original upstream format/LTO warnings are retained in build.log, not silently suppressed. No pip install or environment replacement occurred.

The first build attempt failed BEFORE patching because the mounted checkout is root-owned and the inventory audit Git child lacked scoped safe.directory. Log retained at evidence/parameter-getters-cpu-XNocbm/build-driver.log. Corrected child-scoped trust has a failing-then-passing regression. The successful driver log is build-driver-safe-path.log in the same directory.

Unchanged original post-pair bindings SHA: 89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5.
Unchanged original backend .so SHA: 92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0.
The new binary is not automatically selected by existing native recipes.

## CPU results and commands

87 tests passed, 0 failed, exit 0 in 42.24s:
- tests/test_dh116_deformx_parameter_getters.py
- tests/test_dh116_deformx_parameter_probe.py
- tests/test_dh116_deformx_cuda_build.py
- tests/test_dh116_deformx_pair_capture.py
- tests/test_dh116_deformx_pair_worker_v2.py

Command prefix: DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q
Full log: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/parameter-getters-regression-Eka1et/pytest.log.
Initial missing-module red test: evidence/parameter-getters-red-69KnNj/pytest.log. Build wrapper/path-trust tests also failed on the absent implementation before it was added. CPU fixtures never count as native executions.

Rebuild (new clean independent source/output required):
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m scripts.build_dh116_deformx_parameter_getters --source NEW_CLEAN_WORKTREE --output NEW_BUILD --toolkit EXISTING_SDK_12_8_61 --cmake EXISTING_CMAKE --ninja EXISTING_NINJA --pybind EXISTING_PYBIND_DIR

## Frozen next execution, not yet run in this report

Recipe: configs/dh116_execution/deformx_parameter_getter_readback_v1.json
Spec SHA: 8339551f8c5329d6786c16099cec85ac8e61f02f414e137caaba7a4da55d9319
Code/config SHA: ca509b7593c281b093e51ce078f17dbb25b58d1795b0fbd2c0640dbaddb0f63c

GPU3, one native process, one environment, at most once/90 seconds, 0 physics/0 PPO. After publication bind the current standing user approval to these exact fingerprints. Old native attempts remain consumed. Natural outer exit 0 is required; no forced success or retry.

No new Isaac/contact/grasp evidence yet. Pinch bilateral contacts, Tripod/Power qualification, reset/state completeness and R0 domain/coefficient admission remain unresolved. An isolated getter passing will not grant Teacher PPO admission or justify replacing the old backend in a physical task.
