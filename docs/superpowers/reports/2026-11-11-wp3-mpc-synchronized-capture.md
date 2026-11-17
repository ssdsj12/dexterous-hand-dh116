# WP3 synchronized MPC pre-action capture

The original constrained MPC collector can now be invoked through an independent capture entry point. The entry point defaults to read-only preflight.

- State comes from the original TensorEnv reset and step outputs after validating the versioned 371D surface contract. The observer clones the state before the next existing control call. The existing collector owns every reset and physics step.
- The 371D validity mask compares the existing q/qd prefix against the current measurement and existing normalization. A mismatch remains present in the captured state with validity false. Other fields retain original values and are unqualified.
- Each row binds the actual 24D command submitted to TensorEnv, the separate historical reference action, the camera frame, environment, physical episode, reset generation, and both World and SDK clocks. A normal interval requires eight consecutive 480 Hz samples. A real terminal truncates the interval.
- The raw RGB-D chunk is SHA-checked. The record carries the SHA of the original per-environment intrinsics and camera pose with its resolution and coordinate convention. The source is a fixed monocular RGB-D camera.
- The physical sidecars retain SDK targets and physical readback. The canonical issued 24D vector is the TensorEnv input.
- The output artifact is explicitly unqualified for BC. This short MPC diagnostic performs zero PPO updates and grants no Student success or new expert qualification.

## Ubuntu verification

- CPU regression passed 104 tests and 17 subtests, including 16 new parameterized cases. Initial failures for the missing collector and camera binding are saved on the server.
- Read-only inspection of the historical Stage7 sample found 1,440 rows of width 3,844. It matched 64 issued actions and camera frames across the two grasp styles to 512 original physical substeps. Historical raw camera manifest SHA is 8717fbc00a272e3295a664435583151e3442aa145dfb37e8cb0b1ba7da2e96d0. This check provides no historical 371D labels.
- Preflight verified the existing baseline plan SHA and ten SHA-bound read-only input files. The resolved limits are two environments, one of each grasp, and at most 32 policy steps. Preflight created no native output directory.
- All 528 protected source, script, and config bytes matched their recorded SHA. Historical Teacher checkpoint SHAs are ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511 and c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999.
- Logs are held separately under /tmp/dh116-wp3-mpc-20261111/ on Ubuntu. This package executed zero Isaac sessions.

    PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      CUDA_VISIBLE_DEVICES="" /mnt/mydisk/xk/envs/dh116/bin/python -m pytest \
      -q -p no:cacheprovider tests/test_mpc_privileged_capture.py \
      tests/test_dh116_privileged_action_audit.py tests/test_bimanual_mpc_capture.py \
      tests/test_dh116_privileged_teacher.py tests/test_surface_contract.py

## Native verification pending approval

On the Ubuntu worktree, run the following as a read-only preflight. After separate approval and a GPU availability check, add --execute-native to run the same bounded plan under the project's single-instance lock.

    baseline=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/smoke180_v01/command.json
    destination=/mnt/mydisk/xk/dh116/runs/wp3_mpc_privileged_sync32_v01
    CUDA_VISIBLE_DEVICES=3 DH116_NUM_THREADS=2 \
      DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
      bash scripts/python.sh scripts/collect_dh116_mpc_privileged.py \
      --baseline-plan "$baseline" \
      --baseline-plan-sha256 0561d7fa70845e7ea85b386d7befa35f35febb16ab44f384d0c85ac65adf1c31 \
      --output "$destination"

Confirm launcher exit code, isolated output, actual 371D state and joint-mask counts, RGB-D calibration SHA, source SHA, separate issued/reference actions, and continuous physical sample intervals. If capture fails, preserve the original raw sidecars and failure report. Formal BC, Teacher success evaluation, and same-initial-state policy comparisons require separate evidence.
