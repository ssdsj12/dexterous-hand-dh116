# Preview multi-resolution execution wiring

Baseline: c3e99e39e3c66b07f832d0b56a64c15370902816. Date: 2026-11-16.
CPU result: **158 passed, 0 failed**. New tests failed **19/19 before implementation**.
No Isaac instance, GPU computation, PPO update or new native attempt was started in this package.

## Concrete fault and changes

The general stereo stack already supported multiple resolutions. The actual preview_dh116_physical_webrtc.py entry omitted width/height when calling RGBPairBridge.from_native, so every preview used its 640x480 defaults regardless of the desired comparison. This package fixes that execution gap; it does not replace the stereo stack.

- tabletop_preview_contract.freeze/verify: independent v2 contract with an exact shared left/right capture_resolution; validate positive integer sizes against existing resource guards, retain physical/render budgets.
- tabletop_preview_contract.create_preview_camera: actual preview factory uses the frozen size and existing rig; recalculates both intrinsic matrices at 75-degree horizontal FOV, preserving 80mm baseline and both optical transforms.
- preview_dh116_physical_webrtc.main: --resolution WIDTHxHEIGHT is freeze-only. Native execution cannot silently override an authorised contract.
- The scene calls that factory. capture_calibration.json stores calibration/SHA before first read; frame metadata includes dimensions, intrinsics and calibration SHA.
- preview_resolution_matrix_v1.json: 224x96, 320x240, 640x360, 640x480 plus a custom-size entry point. Old profiles remain unchanged. 640x480 remains the main accuracy default, not a mandatory size or maximum.
- CPU tests also exercise custom 512x384, malformed dimensions and both lenses' extrinsics.

Fixed horizontal FOV plus square pixels changes vertical FOV with aspect ratio. These are new native lenses, not resized images. A future comparison must account for visible-ROI differences instead of changing its reference denominator.

## Executed checks

Ubuntu command: DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q followed by:

- tests/test_preview_resolution.py
- tests/test_preview_warp_readback.py
- tests/test_preview_camera_clock_startup.py
- tests/test_preview_contract.py
- tests/test_dh116_stereo_rgb_bridge.py
- tests/test_dh116_stereo_rgb_resources.py
- tests/test_preview_owner_release.py
- tests/test_preview_debugger.py
- tests/test_dh116_stereo_resolution.py
- tests/test_dh116_stereo_resolution_benchmark.py

Actual result: **158 passed in 10.38s**. This includes clock/readback, resource ownership, raw RGB identity/synchronisation, subpixel disparity and benchmark evidence gates. Native reader creation is replaced in CPU tests; CPU tests do not prove renderer output or native closure.

The existing estimator was also executed at all four full resolutions with an analytic 1m plane and a shared seeded synthetic texture. Fixture values below are debugging measurements, not actual scene accuracy, wire accuracy or a resolution-selection result:

| Size | fx (px) | CPU fixture valid pixels | Fixture depth MAE / RMSE / P95 (mm) | One CPU compute (ms) |
|---|---:|---:|---:|---:|
| 224x96 | 145.961242 | 14322 | 17.465 / 17.865 / 23.381 | 31.32 |
| 320x240 | 208.516060 | 61754 | 12.313 / 12.589 / 16.453 | 115.07 |
| 640x360 | 417.032119 | 197618 | 6.319 / 6.521 / 8.881 | 512.78 |
| 640x480 | 417.032119 | 267804 | 6.324 / 6.519 / 8.838 | 670.80 |

All four had mean disparity error below 0.5px. Fractional resampling/block-match bias is observable in this synthetic fixture; these values do not satisfy or waive any native 5mm point qualification. Timing is a single CPU diagnostic, excludes Isaac capture/physics and is not training throughput.
Fixture artifact SHA: 4150686891fe532dc56f513cf0100d6cbeeba8d9bede0175ba7e896e1abf2365.

## Native limitations and failure evidence

No new native camera images exist for this package. Per-size native depth MAE/RMSE/P95, wire completeness, GPU memory and throughput are **NA**, and no size is selected.

The prior authorised two-attempt batch is exhausted. Its second run reached raw RGB pointer readback and failed on the installed old owner keyword. The isolated compatibility fix is published separately and still requires native verification.
That run's real GDB fatal stack begins in libomni.kit.livestream.plugin.so, passes through Carb events and the renderer; the final inferior signal is 11 and outer exit 255. Manager/ownership readbacks failed before close. This locates the crash boundary but does not prove a unique root cause or successful release. Changing resolution is not presented as a remedy for that shutdown fault.

Next: freeze four exact contracts after publication; obtain a new bounded native budget. Execute serially, one instance, no PPO, independent outputs; retain original physics/control and stop repeating a common failure. Old 528 protected files, checkpoints, assets and success thresholds are checked by the normal publisher. PR #2 remains Draft.
