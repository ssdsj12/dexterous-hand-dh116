# DeformX Object Import Implementation Plan

> **For agentic workers:** Use subagent-driven-development for isolated asset work, then integrate and verify the simulator adapter. Preserve existing uncommitted work; no commit/push is requested.

**Goal:** Import the original DeformX flexible asset into the current Isaac DH116 scene and drive actual deformation and contact.

**Architecture:** Portable USD bundle and a per-instance skeleton driver form the rendering boundary. A separate single-environment Isaac executable couples the existing CUDA Cosserat solver to hand meshes and reaction wrenches, without changing the rigid PPO executable.

**Tech Stack:** USD/UsdSkel, NumPy, Torch CUDA Cosserat, IsaacLab.

## Global Constraints

- One Isaac instance; CPU0 and one numeric thread.
- Preserve current DH116 controls, existing rigid PPO and checkpoints.
- Original orange mesh, length 1 m, radius 0.005 m, 20 bones/elements.
- Density 700 kg/m^3, E=200000 Pa, G=E/3, damping=0.1, rod dt=1e-5 s; example parameters, not calibration.
- No fixed attachment, no rigid wire collision duplicate, no claims of 512-environment flexible PPO or grasp success without evidence.

### Task 1: Portable asset and skeleton driver

Files: create `src/go2_pvcnn/control/dh116/deformx_wire_asset.py`, `scripts/build_deformx_wire_asset.py`, `tests/test_deformx_wire_asset.py`, `assets/deformx_wire_v1/`.

Interfaces: `build_asset(source: Path, output: Path) -> dict` writes `wire.usda`, source USDC and dependencies, plus `manifest.json`; `SkeletonWire(stage, prim_path, asset_path)` imports the bundle and `update(positions, directors)` accepts world-space arrays (3,21) and (3,3,20).

- [x] Write failing tests for portable relative-reference composition, original mesh/bone counts, no physics on wire, and independent animations for two instances.
- [x] Verify straight and bent skeleton skinning numerically through UsdSkel, including a transformed ancestor; no change outside object subtree.
- [x] Implement bundle builder and per-instance skeleton driver. Copy provenance and applicable license; suppress bundled light, retain mesh/material/skin. Reject shape/nonfinite inputs.
- [x] Run `DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_deformx_wire_asset.py -q`; build bundle from `/home/xk/coding/DeformX/usd/wire/wire_orange_s20_r0.005_l1_smooth.usdc`.
- [x] Review exact new files and report source hash, import/deformation evidence and limitations.

### Task 2: Flexible object Isaac integration

Files: new `src/go2_pvcnn/control/dh116/deformx_wire_runtime.py`, `scripts/demo_dh116_deformx_wire.py`, focused runtime tests and documentation.

- [x] Test fixed material-region measurement and force/torque routing with independent expected values.
- [x] Build a visual-only wire branch in scene spawning that does not create rigid-object contact sensors. Retain the default rigid path unchanged.
- [x] Reuse hand mesh world transforms, integrate the CUDA rod at dt<=1e-5, send integrated reaction wrenches back to each contacting hand body about its COM. Use all hand collision meshes. Update the original USD skeleton from the solved state.
- [x] Validate local material-region height and log global minimum separately; abort on nonfinite/divergent state.
- [x] Save/stop the old trainer before launching one short headless Isaac validation. Record actual import, motion and contacts without calling a scripted or zero-contact run a successful grasp. Retain the saved rigid checkpoint as rollback; do not automatically train the old object after the user requested replacement.
- [x] Update README and CHANGELOG with exact commands, measured results and remaining vectorized-training limitation.


Task 2 implementation details:

- `spawn_scene(..., *, external_wire=False)` defaults to existing rigid behavior. When true, create neither wire nor Wire-filtered sensors/material; return wire=None, sensors=[] while preserving hands and table. The new executable explicitly imports SkeletonWire under the freed Wire path.
- `deformx_wire_runtime.py` is independent of Isaac imports and exposes `paper_rod_config(center, table_center, table_size)`, `outward_hull(points)`, `reaction_at_com(linear_impulse, angular_origin_impulse, com, dt)`, `GraspRegions(initial_positions, grasp_points, half_width=.02).measure(positions, velocity, radius)` and `WireRuntime(config, meshes, grasp_points, device)`.
- Mesh dictionary order defines unique side/body/shape channels; preserve all 34 original convex hand mesh envelopes. Outward triangle winding must be corrected against scipy ConvexHull outward normals. Mesh coordinates are body link local, so motion velocity at link origin is `v_com + omega cross (link_origin - com)`.
- Use `advance_with_meshes` and its `channels[name].mesh_angular_origin_w` to calculate reaction torque about actual hand body COM: `(L_origin - cross(com,J))/dt`. Sum multiple shapes per body, apply at COM in world coordinates through `root_physx_view.apply_forces_and_torques_at_position(..., is_global=True)` each Isaac tick. No clipping hidden from diagnostics; abort nonfinite or >100 N per-body mean reaction during import validation.
- Retain high-friction choice explicitly as Coulomb coefficient 1.0 in external solver; this is an approximation, not an identical PhysX static/dynamic contact law. Set contact_subsamples=10 (5 mm sample spacing along 50 mm elements), k=10000 N/m and normal damping=10 Ns/m; expose these in report. State no rope self-contact support.
- The CLI loads the same pose summary, grasp initialization and load profile as the active trainer, validates provenance using existing adaptive_initialization, then overrides only wire geometry/position to length 1 m, radius .005, horizontal Y, center height table top+.005. Keep original hand roots/joints; report resulting gaps and do not assert successful grasp.
- Default CLI `--steps 12`, optional `--headless`, `--output` new directory, defaults for current pose/initialization/load profile and built asset; physics fixed 480 Hz. Only root controller may launch Isaac after stopping active training. Agent may run offline tests and `py_compile`.
- Each step holds the current hand targets, integrates rod using measured mesh motions, applies returned reactions, advances PhysX and updates visual skeleton. Report `passed` only for import/finite dynamics/balanced reactions, separately contact_detected, grasp_qualified=false, policy_executed=false. Export initial and final scene USD and full node/frame state for reproducibility. Write incremental monitor during validation.
- Fixed grasp coordinates use projection of initial pad-pair centers onto initial rod material coordinates; include interval endpoints and all interior material nodes for accurate piecewise-linear minimum. Global lowest height remains separate. Height criterion alone must not be labelled grasp success.
- No new PPO or 512-environment claim in this import task. This executable is the real object replacement scene; existing rigid training is retained until separate vectorization.

Execution decision: the latest request is object replacement/import. The old rigid trainer is saved and stopped, not resumed against the old object. Flexible vector PPO remains explicitly outside this import implementation, with its absence reported.
