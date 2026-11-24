# Tensor-native checkpoint resume CPU execution proof

- Date: 2026-11-24
- Worktree: `/mnt/mydisk/xk/projects/dh116-goal-ppo-native-20261123`
- Scope: independent `tabletop_tensor_checkpoint.py` and CPU tests only. Existing entry, collector, PPO algorithm, historical checkpoints and raw data are unchanged by this work package. No GPU or Isaac instance was started, and no commit was created.

## Behavior and interface

`save_checkpoint(path, trainer, *, recipe_sha256, receipt_sha256, backend_profile, seed_sha256, policy_seed)` returns a report with `sha256`, identity, counters and policy version. Save validates the serialized policy, complete Adam state and RNG on disposable copies, checks the complete serialized tree, publishes exclusively, and leaves the live trainer/session/rollout and RNG unchanged.

`load_checkpoint(path, trainer, *, expected_sha256, recipe_sha256, receipt_sha256, backend_profile, seed_sha256, policy_seed)` verifies the complete file SHA and identity before restoring. The v2 namespace intentionally rejects the entry's previous serialization-only v1 format. Load restores policy, all Adam values/settings, counters, policy version, CPU Torch/Python/NumPy RNG and all previously initialized CUDA default RNG streams. It creates a new session, write capability and empty device rollout with the restored policy version.

Identity locks the actual TensorNativeTeacher/DeviceDeformablePPOTrainer types, grasp-specific checkpoint namespace, model metadata, full recipe/receipt/seed SHA, policy seed, verified flat-backend profile, PPO configuration, rollout topology, device type, CUDA RNG device count, PyTorch version and current driver/model/collector/PPO/reward/checkpoint-validator source hashes. Policy metadata is also checked inside `_extra_state`. Invalid input is validated on disposable model/optimizer/generator copies before any live state changes.

This is a policy/optimizer/RNG restart, not a physical snapshot. Identity and reports explicitly retain `physical_state_saved=False`, `restart_requires_verified_reset=True`, `history_reset=clear`, `teacher_admission=False` and `grasp_qualified=False`. The caller must rebuild and verify native physical Reset and source/reward histories before collecting. No saved physical state, episode or reaction history is claimed or restored.

## Test evidence

TDD RED command (before creating the implementation):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_tensor_checkpoint.py -x
```

Result: `1 failed in 2.56s`; expected assertion: `tensor checkpoint resume module is missing`.

Focused GREEN: the same command without `-x` returned `16 passed in 9.31s`.

Related CPU regression:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_tensor_checkpoint.py tests/test_dh116_tensor_collector.py tests/test_dh116_device_ppo.py tests/test_dh116_tensor_native_inputs.py
```

Result: `44 passed, 3 skipped in 23.94s`. All skips are explicit operator-allocation CUDA tests.

The real CPU TensorNativeTeacher + DeviceDeformablePPOTrainer test performs an accepted PPO update, saves, continues through another collect/record/update, then loads the saved checkpoint into a different live trainer and repeats that next update. The test compares samples, PPO metrics, model tensors, complete Adam tensors/settings, counters and Torch/Python/NumPy RNG exactly against uninterrupted continuation. It also checks that save leaves the original session/rollout/capability/RNG unchanged and load renews the session/rollout/capability consistently.

Negative tests cover a real Power model trying to load a Pinch checkpoint, namespace/source/physical-state tampering, inconsistent policy/counter/rollout coverage, incomplete Adam moments, changed Adam settings, damaged RNG bytes, mismatched recipe/receipt/profile/seed/policy seed, bad file SHA, exclusive-path publication and pending-transition boundaries.

## Limits and next integration

- CPU fixture profile verification is explicitly substituted as in the existing tensor collector fixture; these tests do not qualify a native backend or mechanical grasp.
- CUDA generator validation/restoration paths are implemented using the existing validation contract and local disposable CUDA generators, but were not exercised on a GPU in this work package.
- Entry integration remains with the coordinating task, which must compute the full canonical recipe SHA and supply the actual receipt/profile/seed identities. Existing readiness/changelog/report integration is also owned by that task to avoid concurrent edits.

Root integration adds actual CUDA continuation evidence: one GPU3 test saves after
an update, restores CPU/CUDA RNG and Adam, then matches next sampled actions,
rollout, metrics, policy, optimizer and final CUDA RNG exactly (1 passed,4.90s).
The native entry save_snapshot now delegates to v2. Source identity also seals
controller, geometry, articulation/setup, rod/pair/mesh and Tripod/wrist adapters.
No physical state is saved or claimed restored.
