# DH116 Teacher PPO with 64 environments per grasp

Pinch, Tripod and Power each have a 64-lane CPU integration regression. Each regression issues sampled 24D actions through the shared physical exchange with CPU doubles, runs eight 480 Hz ticks per lane, settles one reward per lane, and calls the existing structured PPO trainer. The writer checks every action readback, the pre-action and post-action reward stamps, and the SDK and rod tick sequence. It rejects a crossed reward owner, a different issued action, and a critic value identified with a post-Reset state. True terminations omit bootstrap and timeouts use the same pre-Reset endpoint. Test fixtures do not confer native source qualification.

The material, closure geometry and solver health readers now attach to the step-completion boundary used by both the shared and original single-lane exchange. Tripod's contact diagnostics use the same boundary. The Tripod reward endpoint now forwards the same optional SDK hand readback argument as the base endpoint, fixing a Reset-time TypeError. A source error in any lane closes the shared window before PPO recording, including at the final tick. Actual native training fingerprints that include this source must be regenerated for a new run. Historical receipts refer to their original source.

## Native launch status

No native Teacher training ran in this work package. The earlier 64-lane native capacity checks held the seed pose and exercised lane-local Reset without policy queries. The current native collector and actual update worker still require one environment. The grouped writer needs a live scene/episode provider that binds the original hands, rod, geometry reader, reward, history and critic observation for all lanes. The release worker also needs lane-local Reset and a journal that can verify 64 distinct sources per decision. The CPU integration tests supply bootstrap values and scope labels as fixtures. They do not prove native critic observation capture.

All four RTX 4090s had at least 24,027 MiB free at preflight. The current scheduler owns a single `/tmp/dh116-isaac-single-instance.lock`. Concurrent Pinch, Tripod and Power launches require a supported per-GPU reservation and isolated Isaac process state and journals. Other compute jobs subsequently occupied all four GPUs, each leaving over 22,000 MiB free. The 64-lane capacity receipts were recorded one Isaac process at a time. The worker retained its existing resource behavior in this package.

Next run the new owner provider against two live lanes and demonstrate distinct physical actions, same-owner reaction and reward, lane-only Reset and natural exit. Once this passes, run one bounded update at 64 lanes per grasp, checking each native record and each process exit. A 32-control-step rollout at 64 lanes will contain 2,048 policy transitions and 16,384 measured lane ticks. Twelve solver substeps per tick yield 196,608 solver substeps per grasp, separate from setup and Reset verification. The original Lift, Hold and Slip qualification still applies.

## Regression

Current CPU results are 394 batch and PPO regressions, 84 source and model regressions, 64 contact regressions (63 plus the corrected assertion rerun), and 60 registry regressions. Regression log SHA-256 values are recorded in the readiness JSON. Two additional historical Tripod source tests reject the old Reset manifest with a backend-profile mismatch. Both fail identically with the baseline source and with this package. Logs remain outside Git. Use the project's Ubuntu Python with GPU disabled for these tests.

```bash
CUDA_VISIBLE_DEVICES= python -m pytest -q --tb=short \
  tests/test_teacher_batch_rollout.py tests/test_teacher_batch_inputs.py \
  tests/test_teacher_batch_window.py tests/test_teacher_physical_group.py \
  tests/test_dh116_flexible_physical_exchange.py tests/test_dh116_poststep_clearance_guard.py \
  tests/test_native_teacher_reward.py tests/test_dh116_privileged_teacher.py \
  tests/test_dh116_deformable_rollout_admission.py tests/test_dh116_isaac_batch_scene.py
```

The 528 protected source hashes match their inventory. The same two historical temporary Checkpoint paths remain absent.
