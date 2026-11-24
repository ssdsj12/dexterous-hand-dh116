# DH116 shared physical runtime — CPU candidate, 2026-11-24

This work package extracts the existing physical owner into
`NativeTensorPhysicsRuntime` and leaves `NativeTensorCollector` as a Teacher
policy/PPO adapter. The parent task approved the minimal inheritance design.
The initial worktree was branch `11-17`, HEAD
`9ddb3ce37168e21b05799671978fe70b759d1176`; existing unrelated edits were retained.
No commit, push, GPU context or Isaac process was started by this work package.

## Files and exact shared dependencies

- New `src/go2_pvcnn/control/dh116/tabletop_tensor_physics_runtime.py` (662 lines):
  physical setup, original source validation, batch/control owner, raw clearance,
  native mesh bridge/Reset, original geometry construction, rod/source identity,
  measured frame, unchanged hard guards, reward window, selected Reset/peer proof
  and optional capture observation.
- `src/go2_pvcnn/control/dh116/tabletop_tensor_collector.py` is now 87 lines:
  original model/PPO construction, Teacher observation conversion and
  policy/value/rollout adaptation against the shared window API. Existing public
  collector constructor, `sdk`, `window`, `reset`, probes and state fields remain.
- New `tests/test_dh116_tensor_physics_runtime.py`: CPU producer-owned action,
  window, capture, Reset and failure contracts.
- Updated `tests/test_dh116_tensor_collector.py`: keep original reward/PPO checks
  on the inherited shared substep wrapper, patch the relocated clearance owner,
  and check the parent task's compiled-runtime rejection for both training and
  control probes.

The shared runtime does not import or construct `TensorNativeTeacher`,
`DeviceDeformablePPOTrainer` or `PPOConfig`. A fresh-process CPU test verifies the
Tensor Teacher and device PPO modules are absent after importing the runtime.
The original measured reward engine is shared because it owns physical endpoint
and action-window guard/settlement semantics; it is not a model or critic.
The existing source modules retain historical Teacher names and reuse material
validation/feature helpers; no model is instantiated through that shared path.

## Preserved numerical chain

Physical initialization is moved verbatim into `_configure_physics`; the common
`validate_runtime_execution()` check is added before it. Teacher model/trainer
construction stays at the original point before reward/reference initialization.
`_initialize_episode` retains original reward, ages, controls and measured Reset
reference setup, with only bookkeeping initialization added.

There is one physical substep implementation, `_sdk_impl`, used by the inherited
`sdk` wrapper for every action producer:

1. Source boundary → original control prepare → original reaction acquisition.
2. Original wrench/target submit → one SDK step → one batch cache update.
3. Actual regex readback → raw source clearance → original source/cache guards.
4. Original mesh submit → rod advance/export → source boundary → control finish.
5. Original wrist/projection/30 N guards → measured distance/frame → reward
   observe → original optional cold parity snapshot.

The intermediate-terminal guard now lives in the common `sdk` wrapper so a
caller cannot bypass it by manually requesting individual substeps. It executes
once per intermediate SDK, with the same original exception text. Any physical,
source, hook or transaction error marks the scope failed and propagates; no
later SDK, Reset or action window may recover that scope in place.

AST comparison against the pre-extraction file proved all following method
bodies exactly equal, including arithmetic, branching and ordering. Location
attributes and method renames are excluded from the digest:

| Original → shared method | SHA256 of equal normalized body AST |
| --- | --- |
| `source_guards` → `source_guards` | `4e199f497881a1a0eb23e4377ecf1740d79c53e9f7f3bb721f26cc63a62d930e` |
| `arm_probe` → `arm_probe` | `c82d9a781a27a15e39ebaafbb4edd0343f0eeb4034637020e38065ce10f2f614` |
| `snapshot` → `snapshot` | `b9bf64b46d4ab058316d139e35f1acf68b99d5275c8e89379401b8dd87515e25` |
| `rod_state` → `rod_state` | `a6667c16736152804003c7d8205cfc06f8f681988f069a73262b0a5a00629f62` |
| `distance` → `distance` | `a3af97381a66214be87c3adcb7eea701d6eb00e118984cffa17619773f3fe58e` |
| `frame` → `frame` | `862708f78548498a0761a15cc3d7a7c4289a1e40e0411b7bcfbde6701d7e794f` |
| `_reset_refs` → `_reset_refs` | `52e7beeaa7100e8d4ba8812d65e219aea37d245888627a666b874e400fb76400` |
| `sdk` → `_sdk_impl` | `5fd81879eac6dc92161f42e06b662c366df2b1f48485b1234729be6ed5dc15c1` |
| `peer_state` → `peer_state` | `cb1432435ab2506a6774ed8fd99e7b7f091c58888476661f2b8be74daa87be61` |
| `reset` → `_reset_impl` | `a88bf1849c088aafd56f844d5e112f128e42dd1bc55b333fbbf2fece4189bb4c` |

Separate comparisons also passed for original physical initialization, original
Teacher model/trainer initialization, reward/reference initialization,
`RawTableClearance`, `mesh_reset`, `probe_copy` and `require_substep_guard`.
The pre-extraction local audit copy is
`/tmp/dh116-collector-before-shared-runtime.py`; it is not a repository artifact.
The 1 micrometer decision margin, raw quaternion math, FP64 COM arithmetic,
source material/routing validation and 30 N guard are not changed here.

## Producer-neutral window and Reset contract

A caller supplies its own same-device actions, batch token and policy identities:

```python
physical.begin_action_window(actions, token=token,
    policy_uids=policy_uids, end=budget_end)
physical.advance_action_window(actions, budget_end)
physical.record_window_request()
post = physical.measurement("post")
packet = physical.complete_action_window(buffer_full=rollout_full)
# Only packet.done's selected IDs may now call physical.reset(selected_ids).
```

Teacher uses these same methods. Its post-state value is still evaluated after
actual request history is recorded and before reward settlement/Reset. The same
24D submitted actions are bound to the eight-step transaction and mutation or
replacement is rejected. Student policy identities are accepted without a
Teacher model or checkpoint. Original 32-control timeout, reward packet numeric
semantics, bootstrap masks and PPO rollout writing are retained.

Only a settled eight-step window authorizes its actual `done` IDs to Reset, at
an unchanged source boundary. Mid-window Reset, unselected IDs and stepping
before pending done lanes are reset are rejected. The unchanged physical Reset
still proves peer mechanical/rod/reaction/controller/source/reward isolation.
The shared wrapper additionally clears capture history for selected IDs only.

## Optional capture contract, currently no camera

`attach_capture(observer, cadence="window", history_length=2)` is explicit and
must run before the first action window. No observer is attached by default.
Window cadence produces measured pre/post events at the 60 Hz action boundary;
per-SDK capture requires explicit `cadence="substep"`.

The observer implements `on_measurement(event)` returning a mapping from actual
env IDs to plain data/device tensors, and `reset_selected(ids,event)` for local
history reset. Runtime history is bounded per lane and stored/returned as owned
copies. The event contains detached device copies of measured state, rod state,
clearance, submitted action labels and source/token/policy identity. It contains
no model, critic, reference context, authored seed or SDK/simulation handle.
It is a measured-label interface, not an actor input adapter.

Before and after every callback the runtime compares source identity, SDK/rod
clocks, control transaction/epoch, actual hand/data/view identities and tracked
physical tensor mutation counters. Reentrant physical API calls are forbidden,
including calls whose exception is caught inside the hook. Hook errors propagate
and poison the physical scope. Mutating a provided snapshot only changes that
owned snapshot. Callback code is trusted Python: this is not an arbitrary-code
sandbox and cannot detect an independently captured raw native handle writing
and concealing a change without any observable clock/version/source change.
No such raw handle is supplied by the API.

## CPU evidence and limitations

Environment: `/mnt/mydisk/xk/envs/dh116/bin/python`,
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.`.

- Initial new contract suite: **8 failed**, expected helper-module absence.
- Initial frozen shared-runtime/Teacher tests: **15 passed** in **7.03 seconds**.
- A subsequent Startup leakage regression first failed because `state` aliases
  the complete initial packet and contained seed/reference keys. The capture
  measurement was changed to an explicit measured-field whitelist; this changed
  only observer payload selection, with no physical/guard method changes.
- Focused final whitelist revision: **16 passed** in **7.91 seconds**.
- Expanded command:
  `python -m pytest -q tests/test_dh116_tensor_physics_runtime.py tests/test_dh116_tensor_collector.py tests/test_tensor_source_owner.py tests/test_dh116_tensor_native_inputs.py tests/test_dh116_tensor_control.py tests/test_tensor_training_reward.py tests/test_dh116_tensor_checkpoint.py tests/test_dh116_runtime_execution.py`
  Initial frozen revision result: **130 passed, 2 skipped** in **34.88 seconds**.
  Final whitelist revision result: **131 passed, 2 skipped** in **35.48 seconds**.
- Skips are existing explicit CUDA gates:
  `DH116_RUN_DEVICE_TESTS` and `DH116_TENSOR_CHECKPOINT_CUDA`.
- `git diff --check`: passed for tracked changes.

CPU tests cover ordered single physical chain, Student-owned action/policy
identity, shared reward timeout/settlement, owned measured/capture data, default
window cadence, selected capture history/Reset isolation, early guard failure,
mid-window Reset rejection, hook clock/transaction/tensor mutations, hook
exceptions and sticky failure. Existing real Teacher/PPO CPU tests retain action
and pre-Reset value/rollout validation.

The parent task is adapting the independent contact replay entry to instantiate
this model-free runtime and use the common action-window API. Entry changes,
source sealing, public lifecycle/exit handling, geometry work and all native
readback/oracle runs belong to the parent task or the other assigned work package.

This package has not executed Student inference, Student camera capture,
TiledStereoCapture, image history/optical flow, Student native training, GPU
performance or actual grasp outcomes. It does not qualify a Student native
source, introduce an actor path for critic/reference tensors, or change native
admission flags. The optional hook is a tested observation attachment point,
not a declaration that a camera/Student physical pipeline has been validated.


## Freeze revisions and source/evidence separation

- Teacher adapter SHA256 remains
  `acf0579b89bac2195e9a9e070d0e7a4f5163176144ccdd64d1d16a03adda342a`.
- Initial shared runtime freeze SHA256 was
  `8d6d01340abd968e79bc86737a71fd5e789938bbdf7026209b86bf405c29cb57`.
- Final whitelist shared runtime SHA256 is
  `de269bbcf3ea7e9e841b23a935e86b20893703f58d65bb8862f2588d6870f4b1`.
- The source change overlapped the parent task starting a native audit of the
  earlier freeze. The parent was immediately informed of the exact revision
  mismatch. That earlier run must not be represented as validation of the final
  whitelist revision; native source sealing/evidence separation and revalidation
  remain the parent's responsibility. No source has been changed since reporting
  the final whitelist SHA.
