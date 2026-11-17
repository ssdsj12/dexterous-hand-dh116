# DEXTRAH-style TiledCamera capture design

User approved this scope with “确定” on 2026-11-16 after reviewing the proposed physics-boundary capture change.

Use the existing installed TiledCamera as an independent sequential left/right reader. During one capture transaction, the SDK physical step/time, Rod clock, Reset/episode identity and state SHA must remain unchanged. Both views require new render epochs from their own shared render product. Preserve the two raw renderer timestamps independently, together with the physical boundary identity. Do not rewrite one clock to look like another or remove the old bridge's validation.

The legacy_view backend remains available and default for compatibility. tiled_step is an explicit frozen v3 preview option until native-qualified. Resolution is independently configurable; preserve 640x480 accuracy default, four comparison sizes and a custom size. Keep the original optical rig, 80mm baseline, 75-degree horizontal FOV, SDK installation, flexible rod, physics, controller and actor/checkpoint contracts.

The installed TiledCamera differs from DEXTRAH's target release. The private adapter must author standard Camera xform ops, feed float32 poses to Fabric, mirror only its owned camera's Fabric writes to USD, and independently compare USD and Fabric pose readback.

Own all camera callbacks, extra annotators, actual HydraTexture and private rig explicitly. GPU synchronization precedes release. Release failures are preserved; no GC-triggered native cleanup, process kill, forced exit or simulated success is used.

CPU tests cover the software contract. Native left/right image identity, Reset, GPU interop and normal process exit remain independent gates. The previous stopped native batch is not implicitly resumed by this implementation approval.
