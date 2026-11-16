# TiledCamera physics-boundary RGB implementation plan

**Goal:** Add the user-approved DEXTRAH-style sequential dual-view capture to the existing DH116 preview without changing physics, SDK installation, actors or old capture semantics.
**Architecture:** An independent capture coordinator fences one physical/rod/reset identity around both views. An owned native reader calls the installed TiledCamera, preserving original render clocks separately. The legacy bridge keeps its strict equality test. A frozen preview backend selector makes activation explicit.
**Tech stack:** Existing Python3.10, PyTorch, IsaacSim4.5, installed IsaacLab TiledCamera and Replicator; no dependency replacement.
**Execution:** Inline Ubuntu isolated worktree, CPU tests first, Windows publication. No new native attempt is included in implementation approval.

- [x] Write failing tests for valid independent raw clocks, stale frames, physical/rod/reset changes, incorrect extrinsics, partial pair rejection, restore-on-error, and four sizes/custom size.
- [x] Implement tabletop_tiled_stereo_capture.py. Return a pair transaction identity plus both original raw frame/time records; never invent a shared raw timestamp.
- [x] Write resource ownership failures first. Implement tabletop_tiled_stereo_native.py and tabletop_tiled_stereo_resources.py using actual installed API and source SHA guards.
- [x] Register ownership before native initialization, track the actual newly created HydraTexture, detach own callbacks/annotators and destroy only own render product/rig. Preserve cleanup failures; no native calls from garbage collection.
- [x] Add frozen camera_backend=legacy_view/tiled_step and CLI flag. Default legacy behavior is preserved; the new path is opt-in until native-qualified. Keep physical and render budgets and multi-resolution entry points.
- [x] Run new and old CPU regressions, preserve red/green logs, update readiness and report with pending native status.
- [x] Verify protected inventory and historical checkpoints and prepare the explicit publication file set. Commit/push/attestation are recorded by the release manifest, not inferred from this checklist.
- [ ] Prepare a new frozen native diagnostic for tiled_step. The stopped prior batch and three unused old-source contracts are not automatically resumed.

Source review found and corrected standard USD ops, float32 Fabric inputs, and renderer-visible USD mirroring. Real CPU USD crosschecks were added. Final related regression: 264 passed. Native qualifications remain false.
