# Pinch ownership lifecycle comparison: CPU preparation only

Baseline: 07db9eb461f8db067beb294c12c077d1ea1b824e.
Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/physx-lifetime-probe-20261113.
No Isaac instance, native physics probe, camera attempt or PPO update was run for this package.
The only executed GDB inferiors were tiny ordinary CPU Python fixtures.

## Evidence and hypothesis

The preceding immutable audit, docs/superpowers/reports/2026-11-13-physx-tensor-lifetime-audit.md,
records that failed Pinch retained hand/view/composer/rope weakrefs through close while successful
Thumb dropped them before close. The actual failed Pinch stack maps its first two frames to
PhysX tensor offsets 0x59718 / 0xba7d8 on the Base/GpuSimulationView destruction path.
That archive lacks registers; it does not establish a particular invalid pointer or Python holder.
Installed Articulation retains a root SimulationView alias separately from the manager's alias;
STOP invalidating/clearing the manager alone does not prove all aliases are gone.
No installed public per-owner detach API was identified that justifies changing the SDK.

This package tests one conditional ownership hypothesis. Both cases acquire the same actual
hand, root ArticulationView, permanent WrenchComposer, asset and manager SimulationView aliases,
FreeRodRuntime, RopeBatch and DeformableRodStateProvider bound reader. The registry case returns
normally and relies on the existing run_owned_call cleanup. The explicit case deletes its local
native aliases, bundle and snapshot closure before returning to exactly the same cleanup.
It does not mutate native SDK internals.

A CPU surrogate demonstrates that ordinary SDK-shaped cycles are collected in both cases.
If an external fixture deliberately retains the completed snapshot/acquisition frame, the registry
case retains those locals while explicit deletion releases them. Removing that artificial holder
releases both cases. This proves a conditional Python ownership mechanism, not that the archived
SDK actually held that frame or that this change fixes the native fault.

## Execution scope prepared for later exact approval

Each recipe independently permits one attempt, one environment, GPU 3, 180 seconds qualification
time, 2 actual SDK initialization physics steps, 0 measured physics steps, 0 rod solver steps and
0 PPO updates. There are no cameras or created render products and no explicit render/measurement
loop. The installed SimulationContext.reset() internally invokes render() twice; the probe
intentionally preserves that SDK initialization path. Thus “no rendering at all” is not claimed.
There is one explicit frozen app.update() for delivery of STOP callbacks and one actual
app.close(wait_for_replicator=False).

The original left hand/USD/material/mimic profile, original pose reference, five locked cooked
pad SDFs, rod construction and provider are reused. No PhysX table collider is authored in this
lifecycle-only scene. FreeRodRuntime still owns its finite table SDF. No provider.snapshot is
called because that API requires a measured solver step. A pair may therefore show no retention
in either case; it does not reproduce original Pinch contact, mimic failure or measured-step state.
Neither successful lifecycle exit nor static state acquisition grants contact, grasp, material,
full state371 or Teacher admission.

The two JSONL rows are real observations immediately before and after reset (0 and 2), explicitly
initialization boundaries, not invented per-step telemetry. The custom outer validator additionally
checks their clock delta, actual rod tick/time, frozen available SDK clock observations, STOP manager
release, actual weakref liveness, actual app-close return and independently observed process exit.
If a wrapper cannot expose a weakref, its state stays unknown and cannot qualify as released.

The shared scheduler is unchanged. Its 180-second limit is a qualification threshold: an overdue
child writes an overdue record and fails qualification, but the frozen scheduler waits for exit
instead of forcibly killing it. A future operator must supervise an overdue process; this preparation
does not claim a hard SIGKILL wall-clock cap. Neither timeout nor signal permits an automatic retry.

The two cases must have separate package IDs/scopes and exact recipe/code approvals. They must not
depend on each other's native success. If a batch explicitly approves both, a failed or SIGSEGV
first case may be followed by the separately approved second case only after the first inferior/GDB/
worker has exited and the single-instance reservation and GPU resources are released. No second
attempt, additional measurement step, changed case, expanded time/step budget or third probe is
authorized by that rule. Without approval for both, only the approved case may execute.

## Capture and independent qualification

The dedicated worker re-enters scripts/run_dh116_physx_lifetime_work_packages.py in its real child.
A scoped worker marker binds case, canonical output and current entry/wrapper hashes; the probe
checks it and GDB's marker before mkdir, preflight or SDK import. Recipes cannot supply this marker.

Normal Py_FinalizeEx and SIGSEGV/SIGBUS callbacks collect full raw /proc/PID/maps plus bt,
thread apply all bt, registers, instructions, shared libraries and GDB mappings. The normal
breakpoint automatically continues, so it cannot consume a budget by waiting for interactive input.
Every saved capture artifact is SHA-bound, and the independent validator requires meaningful
frames/registers/library/mapping output and attributes RIP to the captured full maps. Markers alone,
empty sections or changed files do not qualify.

The producer always says process_exit_verified=false. Only the independent outer process can
verify actual outer/inferior zero exit, native lock, log hash and elapsed time. Generic qualification
is withdrawn before the new audit and restored only if both generic and independent gates pass.
Old prerequisite receipts retain their original validators. The independent audit also inventories
partial files on failed cases, preserving comparison evidence without qualifying failure.

## Verification and frozen recipes

CPU workflow began with failing missing-entry/helper/validator/worker/recipe tests, then implemented
the smallest corresponding paths. Intermediate transport-fixture mistakes (output under source and
omitted native_ready stub key) were corrected in CPU test setup; no SDK run was involved.

Final command, from this worktree:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_physx_lifetime_probe.py tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_scheduler.py tests/test_dh116_physx_tensor_lifetime.py tests/test_deformable_grasp_runtime.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_dh116_pinch_scheduler.py tests/test_dh116_deformx_thumb_contact.py

113 passed, 0 failed, 9.89 seconds: 39 new and 74 existing regressions.
The two real CPU GDB fixtures verify normal exit and actual SIGSEGV capture through the independent
capture validator. They are not Isaac validation. Real CPU subprocess tests prove worker reentry,
root/private/output consistency, inherited marker, chosen GDB script and audit-before-receipt.
528 protected source files and two historical Checkpoints passed verify_protection.

Private CPU evidence:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/physx-lifetime-probe-cpu-20261113/pytest.log
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/physx-lifetime-probe-cpu-20261113/fingerprints.json

Recipes freeze all new runtime/capture/validator/test files, transitive original acquisition files,
installed SDK ownership/callback files, tensor binaries, original cooked/SDF/backend inputs,
no-camera experience and /usr/bin/gdb. The unchanged candidate preflight independently rechecks
its original source/config identities. Report/registry/README integration does not alter these pins.

| Case | Recipe | command_spec_sha256 | code_config_sha256 |
| --- | --- | --- | --- |
| registry | configs/dh116_execution/physx_lifetime_registry_v1.json | 726e6d06c39011bdd962e00f1d60f319b2465b09c91993d0d6fd7eef3039718a | 354bf1cb8419f156e3b594f10c0760cc127adbc0d46c39c1a36fe044d26b310b |
| explicit | configs/dh116_execution/physx_lifetime_explicit_v1.json | f10ad7bfa1625d0ba239613488233397bb6a8da43a93da63ff902d882e0348c4 | 7880b81229cf40611d4dd367e05a5c3dad75d21ce40fc883df9e88d14e4bab25 |

The fingerprints include these tests (already also in source_files):
tests/test_dh116_physx_lifetime_probe.py,
tests/test_dh116_physx_lifetime_validation.py,
tests/test_dh116_physx_lifetime_scheduler.py.
Adding different tests to a registry package changes its code fingerprint and requires recomputation.

All eleven scoped files are additions. Frozen camera C0–C5, previous Pinch runtime/recipe, shared
scheduler, SDK source, original geometry, model/Adam/RNG, data and historical Checkpoints are unchanged.
Native outcomes remain pending exact new approval; no ownership or contact root cause is claimed.
