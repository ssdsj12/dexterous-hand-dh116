# Four-resolution native batch: first-case failure and clock evidence repair

Approved source: 5b3a76633de0d1d8051b80b9022ec36efb46a240.
Approved request SHA: 24f1f97e8edde406d39a6cc186744cdee192d2b8760b67e456e2d61090d0eba5.
Execution: 2026-11-16 05:40:22–05:41:39 UTC, GPU0 shared, one Isaac instance.
PR #2 remains Draft. No automatic retry or forced termination occurred.

## Actual batch result

| Resolution | Executed | Accepted RGB pairs | Result |
|---|---|---:|---|
| 224x96 | Once | 0 | Render/SDK timestamp mismatch, then native shutdown SIGSEGV |
| 320x240 | No | NA | Stopped by approved common-path-failure rule |
| 640x360 | No | NA | Stopped by approved common-path-failure rule |
| 640x480 | No | NA | Stopped by approved common-path-failure rule |

The attempted process took 76.09s. Inferior signal=11; actual outer child exit=255. The scheduling process returning normally is not native exit success.
Two initialization physics steps completed, SDK time=0.004166666883975267s. No measured physics step, solver measurement substep or PPO update completed.
The physical JSONL has six complete records: header, two setup records, reset, left initial geometry and right initial geometry. **Six rows do not mean six physics steps.** There are no incomplete physical or stage lines.
The authored calibration is 224x96. Raw RGB readback returned past the former Warp owner-keyword failure; the packet was rejected before saving a complete frame. This does not qualify native RGB, stereo accuracy or a final resolution.

## Acquisition fault: what is and is not known

RGBPairBridge._capture rejected its render-time/physical-time check. That version's exception omitted the two clock values. The run proves a failure at this boundary; it cannot establish a numeric offset, distinguish an old frame from separate clock origins, or justify changing the tolerance.
Actual render time is therefore NA, not guessed as zero.

The installed IsaacReadTimes node reads simulationTime from Fabric render results. SimulationContext.current_time is maintained by the physics-step callback. IsaacLab render() flushes Fabric and pumps App with playSimulations disabled. These installed-source semantics make a domain/phase discrepancy worth investigating, but are not a proven cause for this run.
No timeline rewrite, arbitrary clock offset, threshold relaxation or relabelling of the frame was added.

## Independent shutdown evidence

The first GDB fatal frames are in libomni.kit.livestream.plugin.so, followed by Carb events and renderer frames. Full C++ stack, dynamic-library list and mappings remain in the private native log.
Log SHA: a1ce6ac218d9c939c8d8e6455b52d7a54e6a3cd4283876a3c815d5f4b89142bc.

Python liveness before close:

| Resource | Python object still alive |
|---|---|
| SimulationContext | Yes |
| Left/right Articulation | No |
| Left/right ArticulationView | No |
| Left/right asset SimulationView | Yes |
| Left/right wrench composers | No |
| Dual runtime and RopeBatch | Yes |
| Episode and material reader | No |
| RGB bridge | Yes |

These are weakref observations, not proof of C++ ownership or native destruction. The RGB bridge close routine returned and clears internal reader/simulation references; a live bridge object does not by itself prove its reader is live. Manager and ownership guards failed. Their relation to the livestream crash is unresolved; the stack does not prove that PhysX Tensor View caused this crash.
No forced exit, extra gc loop or success marker was used.

## Actual small code repair

tabletop_stereo_rgb_bridge.py now preserves last_capture_diagnostic at the rejected boundary:
SDK step/time, returned render frame/time, absolute delta, unchanged 1e-6s tolerance, bounded raw reader and render metadata.
Only scalars and bounded text are retained. RGB tensors and native handles are excluded. The mismatch still raises; rejected frames do not advance last_frame or last_epoch.
This repairs missing evidence, **not** the underlying timestamp mismatch or shutdown.

Two failing CPU tests were observed before implementation. Final related CPU regression: **160 passed, 0 failed**. No additional Isaac run followed this repair.

## Reproduction and next gate

CPU command: DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q tests/test_preview_capture_clock_evidence.py tests/test_preview_resolution.py tests/test_preview_warp_readback.py tests/test_preview_camera_clock_startup.py tests/test_preview_contract.py tests/test_dh116_stereo_rgb_bridge.py tests/test_dh116_stereo_rgb_resources.py tests/test_preview_owner_release.py tests/test_preview_debugger.py tests/test_dh116_stereo_resolution.py tests/test_dh116_stereo_resolution_benchmark.py

Native commands and exact contract SHA are preserved beside the approval receipt and raw logs. Remaining three original cases are unspent but stopped, not passed or silently reusable with changed source. A revised diagnostic requires a new frozen source/contract and explicit resume of that stopped batch.
Next evidence needed: raw same-frame Fabric and SDK timestamps, frame epochs and non-advancing physics guard; then resolve renderer/physics clock wiring without falsifying identities. The livestream teardown path needs separate native qualification.
Depth MAE/RMSE/P95, wire completeness, per-resolution GPU memory and training throughput remain NA. Teacher and Student qualification remain false.
