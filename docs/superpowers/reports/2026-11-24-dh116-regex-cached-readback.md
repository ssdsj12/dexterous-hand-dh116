# DH116 regex official-cache readback candidate — 2026-11-24

Scope: add an isolated adapter and CPU tests only. The existing Regex owners,
collector, controller, source geometry, COM arithmetic and guards are unchanged
by this work package. No GPU, AppLauncher, Isaac instance, commit or push was run.
The parent task owns integration and native qualification.

## Audited installed producer

- Ubuntu tree: `/mnt/mydisk/xk/projects/dh116-goal-ppo-native-20261123`.
- Inspected branch/HEAD: `11-17`, `9ddb3ce37168e21b05799671978fe70b759d1176`.
- Installed editable Isaac Lab: `0.54.4`.
- Source root: `/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab`.
- This installed source directory has no Git metadata; file SHA256 values are
  the reproducible source identities. No repository HEAD is claimed for it.

- `assets/articulation/articulation_data.py`: `d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a`
- `assets/articulation/articulation.py`: `fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309`
- `utils/math.py`: `881ab27758816475e26d6721bab22c934fd70d0c064392266a90b06f5bbe4149`
- `utils/buffers/timestamped_buffer.py`: `1020269322e63b6383c9782f8cf369cbe8b55f52e081bccd79d03ae52712bd59`
- `utils/wrench_composer.py`: `095ab1fcf374e4bce96b61430f810c83f3519b7225476ce6e0a6ceb403a5f6a8`

## Actual duplicate-read chain

1. `Articulation.update(dt)` delegates to `ArticulationData.update(dt)`
   (`articulation.py:266`, `articulation_data.py:97`). This advances the SDK
   cache timestamp and eagerly evaluates `joint_acc`. `joint_acc:772` evaluates
   `joint_vel:763`, acquiring `get_dof_velocities()` once and filling its cache.
2. Original `RegexArticulationSide.read()` bypasses all those caches, acquiring
   `get_dof_positions`, `get_dof_velocities`, root pose/velocity and link
   pose/velocity directly. The qd acquisition therefore repeats immediately.
3. The next `TensorPalmController.submit()` resets its permanent composer and
   sets world-frame wrenches. `WrenchComposer` gets `data.body_link_pos_w` and
   `data.body_link_quat_w` (`wrench_composer.py:42`, `:291`). Their shared
   `body_link_pose_w` cache is empty, causing another `get_link_transforms`.
4. `write_data_to_sim()` calls `_apply_actuator_model()`; the implicit actuator
   receives `data.joint_pos` and `data.joint_vel` (`articulation.py:1886`). Empty
   q cache causes another `get_dof_positions`; qd reuses the earlier update cache.
5. `source_guards()` reads `data.root_pose_w`, alias of `root_link_pose_w`;
   the original raw root read did not populate that cache, so root is also reread.

## Safe data mapping

| Original raw field | Official getter | Conversion |
| --- | --- | --- |
| q | `joint_pos` | owned row selection |
| qd | `joint_vel` | reuse eager SDK update acquisition |
| root pose | `root_link_pose_w` | WXYZ/XYZW component permutation only |
| root velocity | `root_com_vel_w` | same raw COM velocity, no arithmetic |
| link pose | `body_link_pose_w` | WXYZ/XYZW component permutation only |
| link velocity | `body_com_vel_w` | same raw COM velocity, no arithmetic |
| actual targets | raw `get_dof_position_targets` | no actual-backend official cache |

`root_link_pose_w:470` and `body_link_pose_w:583` clone raw tensors and use
`convert_quat`, whose installed implementation is `torch.roll` with no
normalization. Inverse permutation preserves the original component bit patterns.
`body_link_vel_w` and `root_link_vel_w` change velocity reference points and must
not replace the original COM velocity. `body_com_pose_w` is not raw link pose.
The task's existing startup-only local COM cache and FP64 raw-quaternion world
COM computation remain authoritative.

The official body-link getter calls `update_articulations_kinematic()` before
its SDK transform read. CPU tests cannot prove this is physically equivalent to
raw reads in a native run; the parent task must validate same-step values and
read counts before adopting this candidate.

## Integration contract

Create `CachedSideReadback(owner)` once per bound side after startup. Construction
checks the exact installed version and all five source digests without reading
backend state. After each physical SDK step retain exactly one existing
`hand.update(dt)`, then `reader.read()` or
`read_cached_side(owner, reader=reader)`. All returned fields use `index_select`
owned storage in the original environment mapping. No application cache based
on an asserted step number is introduced.

After Reset or an external state write, invoke
`reader.invalidate(reason="...")` after all writes and before the fresh read.
This invalidates whole-view world pose/velocity and joint-position/velocity
buffers plus their derived world-state caches, and clears only the composers'
`_link_poses_updated` flags. It preserves physical peers, composed forces,
active flags, targets, cache clock, immutable local COM and joint derivative
history. Raw velocity writers must manage derivative history using official SDK
writers. `restore_regex_seed` already runs an FK update and `batch.update(DT)`
without advancing physics; explicit post-write invalidation still ensures a raw
fresh acquisition rather than trusting a setter's submitted-value cache.

Reader ownership checks reject rebound hand/data/view objects. `metadata()`
records actual hand/data/view Python types and device. Its qualification remains
`installed_sdk_source_audit_only`, with `native_equivalence_verified=False`, even
when source getters execute against a CPU fixture or later a GPU producer.

## CPU evidence

Interpreter: `/mnt/mydisk/xk/envs/dh116/bin/python`,
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.`.

- New tests first failed: **5 failed** because the helper was absent.
- New tests after implementation: **5 passed** in 2.46 seconds.
- Focused suite command:
  `python -m pytest -q tests/test_dh116_regex_cached_readback.py tests/test_dh116_regex_articulation.py tests/test_dh116_regex_task_setup.py tests/test_dh116_regex_native_preflight.py`
  Result: **21 passed, 1 skipped** in 2.73 seconds. The existing task-setup CUDA
  test requires `DH116_TENSOR_SETUP_CUDA=1` and was intentionally not enabled.
- `git diff --check`: passed for the working tree's tracked diff.

The tests execute the installed full `ArticulationData` class source and installed
`convert_quat` function via AST compilation against a CPU spy backend. They do
not replace the lazy-getter implementation. The independent oracle is the
existing raw `RegexArticulationSide.read()`. Tests verify single q/qd/link getter
acquisitions across SDK update, adapter capture and downstream cache consumers;
permuted environment rows; bitwise equality including signed zero, non-unit
quaternions and distinct NaN payloads; owned captured-post immutability after
next-update refresh; same-clock explicit external-write invalidation; source and
version drift rejection; and owner/data/view rebinding rejection.

Native equivalence, native read counts, GPU performance, actual grasp outcomes
and full training are **not validated by this work package**. The original
1 micrometer and 30 N guards are not modified.

## Coordinating-task native follow-up

The final current-source two-lane model-free contact replay completed218 SDK
steps and27 windows with436 before/cache/after raw side snapshots,zero mismatches,
selected Reset+next-window continuation and natural exit0. This is the explicit
native cohort needed to enable this adapter in the separate N128 timing run.
It does not grant grasp, Student/camera or general native qualification.
See [shared runtime report](2026-11-24-dh116-shared-runtime-lifecycle-native.md).
