# PhysX tensor lifetime: Pinch failure and successful Thumb

Base: 7e98f864fc09aec09e1500a5359d994846c25850.
This package starts no Isaac process, debugger, CUDA workload, physics step or training.
No frozen candidate, runtime, scheduler, recipe, SDK or checkpoint is changed.

## Finding and limit

The failed Pinch retained the hand, articulation tensor view, composer and
RopeBatch at its nominal owner-drop checkpoint and after app.close returned.
Successful corrected Thumb released those four resource classes before STOP.
Both runs reported the SimulationManager global view as None after STOP.
That global readback is therefore insufficient evidence of all tensor ownership
having ended.

The failed process's first stack and mappings place its fault in SimulationView
release. Installed ELF RTTI and instructions strengthen this localization.
They do not identify the retaining Python holder or whether the final indirect
call used a null, stale or otherwise invalid interface pointer. No fix or
causal proof is claimed. These runs also differ in geometry, materials, solver
objects, failure path and resource scopes; they are not a lifetime-only A/B test.

## Immutable evidence

All private paths below are relative to
/home/xk/.local/share/dh116-autonomous-3c80166/evidence.

| Property | Pinch | Corrected Thumb |
|---|---|---|
| Package | flex.f4-free-pinch-left-candidate-native | flex.f4-left-contact-native |
| Attempt | attempt-crp1pahm | attempt-_7cva3ff |
| Tested HEAD | 1e2ab725cdbccdea7e6210fa4dfe29caeac27420 | 3f065c7eb44ce597d8fd38f3d7726df5944cc231 |
| Code/config SHA | d0ec5e2f6146c82e637331acd8dac173fd46cc80425e769578d36e7ce98273de | 0918aca3f52b88c3fe24fb6d3d457eed81d9f8cb5f272125f9e453debc7b69fb |
| Receipt SHA | 601652996a62e791edb49b0b75f3d4815be36090903308872ad919a319bacefb | 460ce27e089ae3178b912dd22e72286122c4eee4d11769b154ef27051d0dd6ca |
| Native log SHA | 8fa58be7b169b0d3218b4b49ef67d38f5674479b3ba6b98c09d016d853bb91b7 | 412be78388c5748411521b30f33d0e957ee989dee415ce9f54e50cd643705684 |
| Ownership SHA | bcb65c896a335eaabda612bc04dbeba2a3183d99af7acd30fe692c7ae91b82c4 | 3802e2297b3973beabcca293692dbb44307a058592012638d9f2ef3226c6726f |
| Outer / inferior | 255 / signal 11 | 0 / exit 0 |
| Wall seconds | 33.041884151054546 | 32.02661002299283 |
| Physics | setup 2 + measurement 1 | setup 2 + measurement 238 |
| Hand/view/composer/rope before STOP | all live | all dead |
| Hand/view/composer/rope after close | all live | all dead |
| Simulation Python weakref after close | live | live |

Receipts: native-attempts/<package>/<code-config SHA>/<attempt>/receipt.json.
Pinch output:
approved-c5-pinch-20261112-v9l5ajv3/native-runs/flex.f4-free-pinch-left-candidate-native/attempt-crp1pahm/native.
Thumb output:
continuation-p0-approved-93c17beb/thumb/flex.f4-left-contact-native/attempt-_7cva3ff/native.

Read-only derived audit:
physx-tensor-lifetime-20261113/audit-v2.json, SHA
5ffa06e55b92bdddea1391472ed22abf936156f8b84c036237a0ae6d9f8ef455.
It independently checks receipt and ownership SHA pins and each receipt's log
SHA, stores the audit script/helper hashes, and marks native_execution_performed=false.
The initial audit.json is preserved; no prior evidence was overwritten.

## Same-process stack, maps and installed ELF

The first Pinch fatal capture contains bt32, all-thread bt20, shared libraries
and GDB process mappings. The new parser requires ordered command boundaries,
actual stack/thread/library contents and a matching first-PC mapping.
Fatal markers or empty command sections do not establish a complete capture.

| Frame | Actual PC | Module | File offset |
|---|---|---|---|
| 0 | 0x7ffa27459718 | libomni.physx.tensors.plugin.so | 0x59718 |
| 1 | 0x7ffa274ba7d8 | libomni.physx.tensors.plugin.so | 0xba7d8 |
| 2 | 0x7ffa2ca2c1de | _physicsTensors.cpython-310-x86_64-linux-gnu.so | 0x2c1de |
| 3 | 0x7fffeec4213d | _carb.cpython-310-x86_64-linux-gnu.so | 0x4213d |
| 4 | 0x7fffeec425de | _carb.cpython-310-x86_64-linux-gnu.so | 0x425de |

Each offset is PC minus matching mapping start plus mapping file offset.
Py_FinalizeEx is frame 13. This differs from C4's first _extensions+0x1e445
fault. Shared Carb/Python frames do not establish a common root cause.
Pinch has mappings in its GDB log; there is no invented standalone full
/proc/PID/maps or normal-finalizer stack.

Installed plugin path:
/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physx.tensors/bin/libomni.physx.tensors.plugin.so.
SHA c54757543c9f621a77d213fce5aed7e74c6fe8127eba2b7206d9fa417f474745;
Build ID db5129baa214dc094b7113c931a85a339474bf90.
Python binding SHA:
d822c4889d1d09c355b093cce7b958ede5f4cfa75e41acc822ba65013ea9c29e.

Read-only ELF inspection:

- Code at 0xba7a0 writes vtable 0xb8c248. Its RTTI type is
  N4omni5physx7tensors17GpuSimulationViewE. It calls 0x596f0 at 0xba7d3,
  returns at observed 0xba7d8, then tail-calls sized operator delete (size 0x350).
- Code at 0x596f0 writes vtable 0xb8aff8. Its RTTI type is
  N4omni5physx7tensors18BaseSimulationViewE.
- At 0x59707 it loads the address of global slot 0xba0740; at 0x59715 it loads
  that slot into RAX. The faulting instruction is call *0x1a8(%rax).
  This supports a SimulationView release/destruction path calling a service
  interface.
- No archived registers, slot memory or complete debug symbols identify that
  service or distinguish null pointer from use-after-free. The nearest exported
  STL symbol printed by objdump is not proof of the containing function name.

physx-tensor-lifetime-20261113/installed-sdk.json, SHA
63ab652878e9ca6c32646965be4a7b700a87cb8e0299d0180b6bf574e64158a3,
stores 11 installed source/binary paths and hashes, RTTI reads and exact
readelf/objdump command output. These are current installed-file hashes.
The old receipt did not independently pin these PhysX binary hashes; matching
mapping paths alone cannot establish historical bytes.

## Actual SDK ownership and release ordering

IsaacLab source root:
/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab.
Isaac Sim root:
/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim.

| Creator / code | Strong ownership | Release behavior relevant here |
|---|---|---|
| SimulationManager PHYSICS_WARMUP | global _physics_sim_view | default-order STOP calls invalidate, then clears this class attribute |
| physics/tensors/impl/api.py | SimulationView owns _backend and _frontend | invalidate marks unusable; it does not drop every Python alias |
| Articulation._initialize_impl | _physics_sim_view alias and _root_physx_view | priority-10 STOP clears root view, but leaves _physics_sim_view |
| ArticulationData | weak proxy to root view | does not make Articulation's own references weak |
| Isaac 4.5 tendon compatibility | four root-view getter lambdas capture hand | creates hand/view cycles; warning-only setter does not capture hand |
| WrenchComposer | strong _asset; two default-argument getter aliases | reset clears command state, not ownership |
| WrenchComposer buffers | Warp force/torque/index buffers plus Torch aliases | Python death and CUDA completion are separate observations |
| FreeRodRuntime/provider | runtime owns RopeBatch; provider owns a bound runtime reader | no global PhysX ownership conclusion follows from dropping registry entries |
| AssetBase callbacks | weak proxies in PLAY/STOP/prim-deletion callbacks | destructor unsubscribes handles |

Source details: articulation.py lines 1508/1547/1559-1570 and 1698;
asset_base.py lines 103 and 270-359; articulation_data.py line 51;
utils/wrench_composer.py lines 37-65; SimulationManager lines 106-128;
tensor api.py lines 5-110 and 160-175. Exact file hashes are in installed-sdk.json.

Pinch's frozen sequence is: _run_impl returns a host result; composer.reset;
torch.cuda.synchronize(device); clear completed exception frames if operation
raised; pop provider/runtime/view/composer/hand from the registry; gc.collect;
weakref checkpoint. run_app then freezes Kit physics, clears its owned
SimulationContext singleton/callbacks, sets its sim local to None, collects,
sends STOP, performs a frozen app.update to deliver events, checks the manager,
and closes the App.

The actual _run_impl returned failure=mimic_error normally. run_app raised that
host failure only after the all-live checkpoint. Its final error traceback
therefore cannot by itself explain the earlier retention. Hidden prior SDK
exceptions or externally retained completed frames remain hypotheses.
The source invokes CUDA synchronization; the old receipt does not independently
prove every backend stream or callback was idle.

Thumb explicitly deletes native locals and snapshot/mesh/fk helpers inside its
operation before collecting, and collects again after STOP. Pinch relies on
leaving its operation and clearing a separate registry. The different scopes
are actionable observation targets, not a demonstrated root cause.

No installed public owner-detach API was found that justifies clearing arbitrary
native wrappers. Global tensor reset resets buffers; invalidate marks a view
invalid. This package calls neither, does not invoke private _release and does
not overwrite _backend or another owner's SDK fields.

## Versioned diagnostic and CPU regression

The implementation plan, approved within the parent task, was to add failing
ownership/evidence tests, implement a scalar-only helper, audit immutable data
and document limits before one scoped commit. All steps are complete.

tabletop_native_lifetime_audit.snapshot_weakrefs accepts only weakrefs or explicit
unsupported None: 1..32 resources and 1..64 returned referrers each. Dict scans
inspect at most 128 entries and emit at most eight matching string keys.
It returns scalar type/identity, alias-key names, closure-cell classification
or frame/function source location. It never reads frame.f_locals, calls repr
on owners, invokes resource getters, collects/releases objects or returns
native objects/frames. gc.get_referrers itself enumerates Python references;
these are output bounds, not a constant-time guarantee. C++ holders may be hidden.

Real Python fixtures cover aliases, cycles, stored closures and retained
exception frames. Reports remain alive while actual holders are removed, and
weakrefs still die. A dedicated negative test caught the helper's own list
comprehension creating a closure cell around its target; the implementation now
uses a normal loop. Another peer review caught empty thread/library sections;
two failing tests preceded the complete-content check.

An AST surrogate extracts the original _run_impl snapshot and mesh_poses
definitions, builds SDK-shaped strong cycles, and invokes the frozen run_owned_call
with fake Python resources. All resources die on normal host-result return.
Thus those two closures/cycles alone do not necessarily leak. The fixture does
not reproduce the actual SDK graph or the native crash.

Validation:
- Clean baseline relevant regression: 20 passed.
- Initial helper tests: 11 expected missing-helper failures and one baseline pass.
- Observer self-cell: one expected failure before correction.
- Empty thread/library capture: two expected failures before correction.
- Final run: 74 passed, 0 failed in 7.46s (16 focused + 58 prior regression).
- One intermediate fixture incorrectly wrote literal backslash-n; fixing the
  fixture newline resolved it without changing the runtime.
- 528 protected source files and two historical Checkpoints passed verify_protection.
- Only four new files are added; all previously tracked files are unchanged.

Final CPU command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_physx_tensor_lifetime.py tests/test_deformable_grasp_runtime.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_dh116_pinch_scheduler.py tests/test_dh116_deformx_thumb_contact.py

The audit CLI takes explicit --pinch-receipt/--thumb-receipt paths, each
--<name>-receipt-sha and --<name>-ownership-sha, and exclusive new --output.
All pinned values and paths are listed above.

## Next condition

The missing evidence is the direct retaining-owner observation at the failed
owner-drop boundary. A future separately versioned candidate can record this
helper's scalar result from NativeOwnershipTrace.references after its existing
collection and after STOP. If it identifies a completed operation frame or
closure, release that specific owner in the new candidate and reproduce the
mechanism in a CPU fixture. If only unknown/native references remain, localize
that owner instead of speculatively resetting global SDK state.

Actual liveness and physical contact-pair capture remain gates on any new
candidate recipe. Python owner death is necessary evidence, not an outer-exit
guarantee. Outer/inferior status, complete capture and exact physical budget
still require independent verification. No new native recipe, attempt, native
qualification or grasp claim is created here. Future native work requires its
own frozen scope and explicit approval; old budgets remain consumed.
