# DH116 compact GPU contact reduction

The opt-in compact reduction consumes the shared DeformX pair ABI. Float64 GPU index reductions reconcile each environment against its original solver node forces and mesh wrenches, preserving signs, step averaging, atol=3e-4 and rtol=5e-5. It avoids per-contact Python dictionaries.

The compact_pair_capture function checks ordered environment IDs, capacity, overflow, native error, shapes, finite data, unique substep/node/mesh keys, unit normals, compression direction and aggregate forces. Returned tensors own their data. All-empty batches must still reconcile. The atomic reader binds each read to the supplied SDK step, separate rod ticks, mesh names and expected native owner time. Repeated calls remain repeated reads. The future live exchange consumer must enforce single consumption, solver/episode/reset identity and next-step reaction delay. This layer grants no training admission.

## Verification

- Both new module and reader tests failed before implementation. Red logs are saved locally.
- Scoped current Teacher batch and pair CPU regression passed 73 tests with 2 CUDA skips.
- Related pair suites on GPU 3 passed 47 tests, including CPU/GPU parity, empty lanes and all-empty batches, corruption rejection, mutable-buffer isolation and atomic read.
- GPU 3 CUDA probes ran Pinch, Tripod and Power with 128 lanes each, retaining the existing three-step/local-Reset and separate overflow checks.
- Heated compact validation times were 12.07 ms, 6.02 ms and 6.04 ms. Pinch first CUDA validation incurred 216.40 ms of initialization.
- These probes include zero Isaac environments and zero PPO updates. Native readback originates in host arrays, then transfers to CUDA. End-to-end training throughput remains unmeasured.
- Shared versus independent layout differences persist. Reward/guard/action equivalence at 1e-5 remains open.
- Broad wildcard test collection failed on missing co_sim path, then missing open3d after adding the vendor path. Logs are retained. Scoped current Teacher group regression is recorded in readiness.
- All 528 protected file hashes matched. The two historical checkpoint paths in /tmp no longer exist, so their contents could not be reverified. This work wrote no checkpoints.

## Review and remaining integration

Antigravity reviewed the supplied current ABI excerpt and design. Its readonly MCP remains at old HEAD 125f2664, so this was not a current-tree code review. Follow-up agreed to preserve the existing tolerances and highlighted single-consumer, per-reset episode, solver-instance and delayed reaction ownership. These belong in the live shared exchange.

Next connect compact data to actual Isaac per-lane composers and local Reset, then validate reward/guard/action parity. Pinch second-window finger13 clearance failure requires the translated policy-input comparison. Three-grasp 128/512-lane Isaac PPO training remains incomplete.
