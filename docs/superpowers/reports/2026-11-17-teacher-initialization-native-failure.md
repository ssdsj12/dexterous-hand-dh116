# Initialization-only native attempt: readback passed, capture and exit failed

## Actual scope
One user-approved GPU2 attempt,1 environment, frozen SHA806f802d9541d1c3e33ae43eb0fd54de882a77b76cecb6c689a290fbf37c8290, executed commit ff16a5c7f9ca4814ac8a5370f2432f7c4435b101.
Exactly2 SDK initialization steps;0 control/rod-solver/Teacher query/model/PPO steps. Cameras/WebRTC off. Once-only budget consumed; no retry.

| Actual measurement | Left | Right |
|---|---:|---:|
| Initialization readback | passed | passed |
| Maximum finger position error, rad | 3.722351e-8 | 4.131126e-8 |
| Base position error, m | 1.080613e-8 | 1.928457e-8 |
| Base rotation error, rad | 2.346637e-7 | 2.632070e-7 |
| Maximum finger speed, rad/s | 4.415199e-9 | 2.552802e-9 |

Readbacks are actual SDK tick2 before any post-reset seed overwrite. This validates only the original supported Pinch initialization, not acquisition candidates, contact, settling, Power or grasp.

The SDK clock remained2 and0.004166666883975267s through STOP. One actual get_link_transforms read per hand. Protected34-owner preflight rejected with "finite rigid actual body-link matrix required":0 cooked collider rows,0 stored-cooked getter calls. Source audit34 is not actual cooked capture34.

## Evidence gap and isolated diagnostic repair
The previous worker discarded raw body pose arrays and derived matrices on protected preflight rejection. Thus the exact failing body is unknown. Near-unit float32 quaternion conversion without normalization is an unconfirmed hypothesis; both saved base poses themselves pass the rigid-matrix tolerance. No old raw packet was fabricated or recovered using another native run.

capture_once now provides on_snapshot after the same two SDK reads, before conversion/collector validation. The native worker writes and fsyncs raw XYZW arrays, source/actual body bindings and SDK clocks. Nonfinite values remain tagged. A durable-write failure prevents cooked getters. The result also retains derived source matrices after collector rejection. No quaternion algorithm, validation tolerance, asset, actuator, force or control was changed. This diagnostic repair has not been run natively.

Three new tests first failed, then passed: raw persistence before collector rejection without rereads; nonfinite raw persistence before conversion failure; failed write blocks getters. Full relevant CPU regression:377 passed,0 failed,0 skipped. Fixtures do not qualify native behavior.

## Shutdown and ownership evidence
app.close returned; SimulationManager view getter became None. During CPython Py_FinalizeEx, SIGSEGV occurred in:
PhysX Tensor plugin -> _physicsTensors binding -> carb binding -> Py_FinalizeEx.
All-thread8-frame stack, main40-frame stack and shared-library mappings are saved. GDB py-bt is unavailable; no missing Python stack is invented.

After STOP, both hand ArticulationView weakrefs were gone, but SimulationContext and both Articulation Python objects survived in generated create_runtime locals/closure cells, asset structures and an installed SimulationContext initializer frame. Incomplete Python owner release is observed; the exact late native destructor remains unresolved. Manager None, object/view release, CUDA synchronization, app.close return and natural process exit are distinct claims.

Worker result was saved after37.180s. GDB paused the fault; the180s soft marker then elapsed without simulation progress or automatic termination. After separate explicit user cleanup authorization, only inferior481611 was SIGKILLed and GDB481481 was SIGTERMed. Both are gone, GPU2 returned to9MiB and the single-instance lock is free. This is a failed, manually cleaned experiment. GDB returned0; the supervisor returned1. GDB code0 is NOT a native natural exit0; native exit code remains unavailable. No new Isaac was launched.

## Evidence and protection
docs/contracts/dh116_teacher_initialization_native_attempt_20261117.json records paths, SHA/size, approval, actual scope, cleanup receipts and CPU command/environment. Raw output remains on /mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence and is excluded from Git.

All528 protected hashes and three original candidate checkpoint hashes match. Two already-missing historical diagnostic checkpoints remain missing; none recreated. The protected collector remains byte-identical at ddf703d192958b2a1a88634c8a8433f46d43e43c15575cd02cee0a5803a75219. Ubuntu commits are published via Windows; PR2 remains Draft, unmerged.

## Next gates
1. Resolve owned runtime/factory aliases and native view destruction before shutdown; do not clear arbitrary installed frames or add blind gc loops.
2. A fresh, separately authorized initialization-only attempt must preserve all34 raw poses and identify the exact matrix rejection. Do not relax the protected validator.
3. Obtain34 cooked sources and live attachment evidence before full Power surface qualification.
4. Validate opposed force acquisition, fixed-wrist holding, support clearing and net Lift before Teacher PPO admission.

No additional native budget or training was consumed by the offline repair. Lift30mm/Hold5s/fixed-anchor Slip0.5mm, bilateral opposed contact and no external support/attachment/upward force are unchanged.
