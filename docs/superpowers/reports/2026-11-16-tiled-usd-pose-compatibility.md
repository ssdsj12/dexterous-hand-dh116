# TiledCamera native result and private USD-pose compatibility

Actual native source: 11e16c8cec1ebda9f9ecda3d035d4d72e63e0795.
Contract SHA: d3edb9811686e8c27bc4be3c08849f81593ac14cdc548c60213d0e8456b35ce3.
The user explicitly approved one 320x240 tiled_step run. It ran once on shared GPU0, with one environment and zero PPO. No retry or forced termination occurred.

## Actual native result

- Two initialization physics steps completed; zero measured physics steps or measured solver substeps.
- Six complete physical JSONL records, including setup/reset/initial geometry; no incomplete physical or stage JSONL lines.
- Zero accepted RGB pairs; failure occurred before the first camera transaction.
- Acquisition failed in TiledCamera._create_buffers -> Camera._update_poses -> XformPrimView._initialize_fabric:
  AttributeError: module 'usdrt' has no attribute 'hierarchy'.
- Inferior signal=11 during close; outer child exit=255. Wall time=73.95s.
- The first fatal frames again lie in libomni.kit.livestream.plugin.so followed by Carb events and renderer. Full C++ stack and mappings remain private.
- Log SHA: be2647462a82b1d5a8f525479b98495938ac51e9f097bbd622ecd6ca4209c311.

The scheduling process ending normally is not native exit success. The per-run authorization is exhausted; the other original resolution cases were not started.

## Root-cause boundary and source investigation

The loaded usdrt module lacked the hierarchy attribute required by the installed XformPrimView. Inspection found hierarchy bindings under the installed omni.hydra.usdrt_delegate extension, so this is **not proof that the binary package is absent**. Extension/module availability in the current Kit context is unresolved.

The installed XformPrimView already has a USD pose read/write implementation. Using that route for one renderer-facing camera does not require the failing Fabric hierarchy API. No environment upgrade, dependency installation or global Fabric toggle was attempted.

## Actual code fix

tabletop_tiled_stereo_native.prepare_renderer_view now selects the existing USD pose route on only the privately owned camera view.
The ExplicitTiledCamera._create_buffers override applies this before upstream's first pose read. Applying it after _initialize_impl returns would be too late for this failure.
Camera input dtypes, intrinsic checks, standard USD ops and independent fresh USD/camera pose comparison are retained.

GPU PhysX, the CUDA rod solver, CUDA RGB rendering, control frequency and physical parameters are unchanged. Camera pose matrices use the USD route; **this revision does not claim a separate Fabric pose measurement for that camera**. Raw frame/time metadata and physical-state fences remain intact. Each pair records camera_pose_backend=usd_camera_prim_isaac45.

The previous revision's report describing Fabric/USD mirroring remains historical evidence. This compatibility revision deliberately avoids the unqualified camera Fabric path instead of enabling extensions or editing the SDK installation.

## Tests and review

Two new CPU tests failed before implementation:
1. The private view must use USD while a separate view retains its Fabric mode.
2. Executing the actual owned subclass override must switch the route before its base buffer initializer reads poses.

Both now pass. Expanded related regression: **266 passed, 0 failed, 0 skipped**.
Independent read-only review found no new confirmed Critical/Important issue in the change. It verified object-only scope and initialization order; no Isaac/GPU test was performed by the reviewer.

Command: use the same 22 test paths recorded in the previous tiled-step report, including the two added cases in tests/test_tiled_stereo_native_adapter.py, with:
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q

No new native attempt followed the fix. Native synchronization, actual pixel/view correspondence, Reset, GPU resource release and normal exit remain false/pending. The livestream shutdown crash is not claimed resolved.

## Publication and next gate

Source, tests, this report and readiness are published through the existing Ubuntu commit -> Windows relay -> GitHub verification flow. The normal publisher checks the protected inventory and historical checkpoints. Raw logs/models/data are not uploaded, PR #2 remains Draft.

A fresh source fingerprint and explicit single-run native authorization are required to test this compatibility revision. Do not reuse the already-consumed one-time contract, launch other resolutions automatically or call this Teacher/Student success.
