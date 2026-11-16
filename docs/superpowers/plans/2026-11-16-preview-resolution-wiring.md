# Native preview resolution wiring implementation plan

**Goal:** Execute the existing camera/calibration code at configurable native sizes without changing physics or choosing a final resolution prematurely.
**Architecture:** Reuse StereoCalibration/rerender_calibration, RGBPairBridge and the existing depth estimator. Freeze a single shared left/right size in an independent v2 preview contract. Keep all old camera profiles and checkpoint contracts.
**Execution:** Inline in the Ubuntu package worktree; Windows relays the completed commit.

- [x] Check current branch, original checkout, remote receipt and clean worktree. Baseline CPU regression: 81 passed.
- [x] Add tests for four sizes, custom 512x384, invalid dimensions, CLI and actual factory wiring. Observe 19 failures.
- [x] Add capture_resolution to the frozen v2 contract. Runtime cannot override resolution through CLI.
- [x] Route preview creation through create_preview_camera; reauthor both lenses using square pixels and the existing 75-degree horizontal FOV. Preserve optical transforms and 80mm baseline.
- [x] Record authored calibration and SHA before first capture, and each frame's intrinsics/resolution.
- [x] Add an independent four-profile comparison config; retain 640x480 as the main accuracy default, 224x96 as a low-resolution test only.
- [x] Run subpixel depth computation on all four full-size CPU fixtures. Record its synthetic-only scope and do not select a native winner.
- [x] Run related regression: 158 passed. Preserve red/green/raw fixture evidence outside Git.
- [x] Write readiness/report and request publication through the existing protected-file publisher.
- [ ] After a new native budget is approved, serially verify four sizes with independent output directories, common initial task, per-resolution lens readback, synchronous RGB, resource evidence and normal process exit. Stop on a repeated common-path failure; no unauthorised retry.

Native depth accuracy, wire completeness, GPU memory and training throughput remain unmeasured. No PPO or success-pool admission is included.
