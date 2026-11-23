# DH116 three-grasp native coupling at 32 and 64 lanes

Baseline `10e1113f94f554adfb5c024a84a179b7f58254e1`. Ubuntu, one Isaac instance at a time on GPU 2. Each grasp ran its 32-lane case before its 64-lane case. The held-pose probe uses 2 setup SDK ticks and 8 measured 480 Hz ticks, with 12 solver substeps per measured tick and rod. It does not ask the policy for actions or perform PPO updates.

Antigravity recommended sharing the Isaac clock while keeping independent hands, rods and reactions. Codex checked the current linked source and the three archived Reset receipts before implementing the probe. Pinch uses the exact 11-slot pair source, while Tripod and Power independently load their own 17-slot files with both end supports, palms, and proximal contact links. The hold probe reuses the original table, materials and wrist clearance guard. It checks the exact lane and side of each live PhysX link before reading its pose and COM. A measured rod reaction is submitted only to the owning composers before the next SDK tick.

| Grasp | Lanes | Outer exit | Solver rows | Native hand-pair samples | Table-pair samples | Support-pair samples | GPU memory at final readback (MiB) | Outer time (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pinch | 32 | 0 | 256 | 0 | 125,952 | 0 | 2,974 | 98.24 |
| Pinch | 64 | 0 | 512 | 0 | 251,904 | 0 | 3,328 | 164.31 |
| Tripod | 32 | 0 | 256 | 0 | 0 | 6,144 | 3,624 | 91.85 |
| Tripod | 64 | 0 | 512 | 0 | 0 | 12,288 | 4,628 | 149.54 |
| Power | 32 | 0 | 256 | 0 | 0 | 6,144 | 3,632 | 94.79 |
| Power | 64 | 0 | 512 | 0 | 0 | 12,288 | 4,636 | 145.14 |

Pair counts are individual solver-substep samples. They do not count completed grasps. Successful cases emitted all eight windows of pair rows with no overflow. A zero-time hand-and-rod Reset of the last lane at SDK tick 6 left the other lanes' numerical snapshots and clocks unchanged. The Reset rebuilt the original archived SDF slots and checked CUDA device copies of current, target, and solver positions and orientations. The ownership trace and outer GDB exit were checked separately. GPU memory is an observed endpoint, not a peak or a PPO allocation estimate.

## Retained failures

The initial batch used the supported-grasp table key for Pinch. The next Pinch attempt called a Reset method absent from the installed binary. The rebuilt mesh was subsequently verified by the SDK, but the first comparison used a NumPy conversion of a CUDA view object and therefore hashed its object representation. The repair uses the project's CUDA numerical endpoint readers. Later runs use new source fingerprints, with one case per width and grasp.

- Pinch 32 lanes, exit 1. `KeyError: 'static_pose'` Receipt `2aafdbba237ec4adb12b1ce084920d559f133c3f70da445d01945239d5822428`.
- Pinch 32 lanes, exit 1. `AttributeError: 'ropesim._ropesim.RopeBatch' object has no attribute 'commit_mesh_reset_pose'` Receipt `c2c04701566e0fc9d3d3ba60be92cdd1e7fac56cddbb4af35bf971b64b67de34`.
- Pinch 32 lanes, exit 1. `ValueError: mesh reconstruction changed actual node or material state` Receipt `ca5336cc165381e61d3854cfc0677dd099e751af1f7daeebf4693a4b9776507f`.
- Tripod 32 lanes, exit 1. `ValueError: mesh reconstruction changed actual node or material state` Receipt `95aec6f445e533a5f5aa23f21724cb7afceeea9fe1b8b36b5660d28c92b25acb`.
- Power 32 lanes, exit 1. `ValueError: mesh reconstruction changed actual node or material state` Receipt `4946b6c6d8f8ba1318af3ceb6cbd29227587ff146e656708d16f50a1a3d156f2`.

## CPU regression

129 passed. The new tests cover archived layout, translated colliders, exact reconstructed mesh readback and regression against hashing CUDA view objects.

```bash
python -m pytest -q tests/test_dh116_isaac_batch_scene.py tests/test_dh116_power_solver_capacity.py \
  tests/test_teacher_batch_inputs.py tests/test_teacher_physical_group.py tests/test_dual_hand_rod_runtime.py \
  tests/test_dh116_power_end_supported_rod.py tests/test_native_teacher_supported_reset.py \
  tests/test_dh116_batch_geometry.py tests/test_native_teacher_collector.py \
  tests/test_native_teacher_actual_update.py tests/test_native_teacher_actual_factories.py
```

The native probe uses the appropriate source Reset receipt for each grasp, a fresh output directory, and a fresh supervised Isaac process. Use the SHA from the readiness record, verify the outer process close, and keep the logs on the server.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 bash scripts/python.sh -B -m scripts.audit_dh116_isaac_batch_scene \
  --style pinch --coupled --num-envs 32 --gpu 2 \
  --receipt /absolute/path/to/pinch-reset/outer.json \
  --receipt-sha256 d8014a7464e43976e7c4e05dae5ca45cd8dbced3dd26092ca97ae636adff38f2 \
  --output /absolute/path/to/new-local-evidence-directory
```

## Training boundary

The diagnostics cover ownership, held-pose dynamics, and isolated Reset. They did not submit a 24D movement request, show Lift, or train a policy. The 32-lane control barrier and batched observation packer still need a native lane owner and a terminal and Reset path for episode recovery. The production PPO worker currently uses the one-environment owner. Validate those pieces with real per-lane pre-action observations, issued commands, reward endpoints and terminal bootstrap before launching a 32-lane PPO update. Keep Pinch, Tripod, and Power separate.

All 528 protected source digests passed. Both legacy checkpoint paths under `/tmp/dh116-wp3-teacher128-v02` were already recorded as absent on November 17 and are still absent in this check. Their expected digests remain in the project registry. No checkpoint file was changed or staged.

See the [readiness record](../../contracts/dh116_teacher_ppo32_readiness_v1.json) for the frozen source digests and proof hashes. The raw logs and model files remain local.
