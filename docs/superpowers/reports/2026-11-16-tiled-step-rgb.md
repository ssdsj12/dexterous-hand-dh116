# Independent physics-boundary TiledCamera RGB path

Baseline: d86bd5503052af60005cab9ee84b04fe15e38602.
Reference: NVlabs/DEXTRAH main at ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50.
Result: **264 CPU tests passed, 0 failed, 0 skipped**. **No new Isaac/GPU run or PPO update.**

## What changed

The existing preview now has an opt-in frozen backend, tiled_step. It follows the [DEXTRAH sequential-view acquisition pattern](https://github.com/NVlabs/DEXTRAH/blob/ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50/dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_kuka_allegro_env.py#L717): obtain a left view, move the owned camera to the right pose, render/update, obtain the right view, then restore the left pose without advancing physics.

This uses the current server's TiledCamera rather than copying or replacing its SDK. It is a capture change only: no DEXTRAH 11D/Fabrics control, Teacher network, Student network or training algorithm was transplanted.

Concrete files/functions:

- tabletop_tiled_stereo_capture.TiledStereoCapture: bounds render attempts; fences SDK/Rod/state/Reset identity around both views; requires fresh epochs; rejects partial pairs, changed state, stale metadata or incorrect calibration.
- tabletop_tiled_stereo_native.TiledNativeStereoReader: actually constructs the installed TiledCamera; calls set_world_poses and update(0, force_recompute=True), with sim.render performed by the enclosing capture coordinator; reads only uint8 RGB, returning owned float RGB copies.
- tabletop_tiled_stereo_resources.TiledResources: tracks the real newly created HydraTexture during initialization; owns only this sensor's callbacks, annotators, render product and marked private rig; preserves cleanup errors.
- tabletop_preview_contract.freeze/verify/create_preview_camera: v3 contract binds camera_backend and six actual installed-source identities. legacy_view remains available/default; tiled_step is explicit opt-in.
- preview_dh116_physical_webrtc.py: --camera-backend is freeze-only; calibration and frame records include backend identity.

All four resolution profiles and custom 512x384 passed CPU wiring tests. Recalculate K at each native size. Keep the existing 80mm baseline and 75-degree horizontal FOV. DEXTRAH's rig-specific right-image flip is not copied to our parallel optical rig. No final resolution is selected.

## Clock contract

The old RGBPairBridge remains unchanged, including its original render/physical timestamp comparison.
The independent path preserves raw_left_clock and raw_right_clock with actual render frame identifiers, original renderer timestamps and bounded original annotator metadata. The SDK physical timestamp is a separate field. It never fabricates a shared raw frame ID or timestamp.

A pair is admitted by the new software transaction only when:
1. Both view epochs are fresh and stable during each image read.
2. SDK physical step/time, Rod/Reset/episode/state identity remain unchanged throughout.
3. Both camera intrinsics and optical transforms match calibration.
4. Left-pose and player-setting restoration succeeds, followed by one final physical identity check.

Only then are capture_id and last_epoch committed. capture_id is explicitly a pair transaction ID, not a renderer frame ID.
Every packet still has native_synchronization_qualified=false. Software guards do not, by themselves, prove that native pixels correspond to the intended physical view.

## Actual installed-API review and fixes

Independent code review found three concrete native blockers before any run:
- TiledCamera's XformPrimView requires standard translate/orient/scale ops. The adapter authors them on its new private Camera.
- The installed Fabric kernel expects float32 pose arrays. Position, orientation and K are supplied as float32.
- TiledCamera's owned view default does not mirror Fabric pose changes to renderer-facing USD. Enable mirroring on only that owned view, matching the installed normal Camera behavior. Cross-check a fresh USD XformCache optical transform against Fabric pose readback.

These issues were reproduced with four failing CPU regressions, then fixed. The USD tests use real in-memory USD stages, not an Isaac physics run. Source SHA guards include TiledCamera, Camera, SensorBase, Replicator create, XformPrimView and Fabric utilities. Installed files are read-only.

The current native adapter intentionally supports one preview environment. Generalizing it to parallel training environments is not claimed complete.

## Verification and evidence

New paths were tested first: 18 initial capture failures, 6 resource failures, backend and native-method failures; later transaction and review failures were also preserved. Raw RED/GREEN logs remain outside Git.

Final command prefix:
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q

Test files:

- test_tiled_stereo_capture.py
- test_tiled_stereo_resources.py
- test_tiled_stereo_native_adapter.py
- test_tiled_preview_backend.py
- test_preview_capture_clock_evidence.py
- test_preview_resolution.py
- test_preview_warp_readback.py
- test_preview_camera_clock_startup.py
- test_preview_contract.py
- test_dh116_stereo_rgb_bridge.py
- test_dh116_stereo_rgb_resources.py
- test_preview_owner_release.py
- test_preview_debugger.py
- test_dh116_stereo_resolution.py
- test_dh116_stereo_resolution_benchmark.py
- test_dh116_live_deformx_display.py
- test_physical_preview.py
- test_preview_presentation.py
- test_preview_durable_stages.py
- test_preview_launcher.py
- test_preview_toolbar.py
- test_preview_device.py

All paths above are under tests/. Result: **264 passed in 13.89s**. Independent final source review found no remaining confirmed Critical/Important blocker; the reviewer ran no native instance and did not claim native qualification.
Readiness records source hashes and log digest. The normal publisher verifies protected files/checkpoints and explicit commit scope. No raw images, binary models or native logs are uploaded.

## Native gate and next action

No new native attempt was run. The prior four-resolution batch stopped after its first legacy-path clock failure; the other three old-source cases are unspent but not automatically resumed with new code.

After publication, freeze a fresh tiled_step contract, initially 320x240 to test the new API path. Request one explicit native diagnostic with the existing per-run ceiling: GPU0 shared, one instance, 130 physical steps (2 initialization plus 128 measured), 1536 measured solver substeps, 162 explicit renders, 180s soft deadline, zero PPO. No retries or automatic forced termination.

Pending: native pixels versus intended left/right poses, frame freshness, actual Reset behavior, Warp/CUDA interop, resource release and outer natural exit=0. The known livestream plugin SIGSEGV has not been proven fixed. No Teacher/Student admission, success pool, depth qualification, or flexible grasp success is granted.
