# Locked RopeBatch material-director endpoint readback

This independent opt-in package prepares an actual existing readback path. It does not modify or launch the solver, rebuild the extension, change physics, update the frozen closure recipe, or qualify native material Slip. The base is published commit `a4a5af8`.

Only three new files belong to this package: `tabletop_material_director_readback.py`, its CPU tests, and this report. Existing marker, runtime, provider, R0, action, reset and checkpoint files remain unchanged.

## Exact available API and synchronization

The already locked extension is:

`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so`

SHA256: `92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0`.

The extension was read as bytes, not imported or executed. Its existing build-tree sources and Python package were SHA-checked; this does not create a new compilation attestation.

- `csrc/bindings.cpp:846–860` exposes `RopeBatch.positions()` and `RopeBatch.quaternions()` as CUDA views, respectively `[N,V,3]` and `[N,V,4]`. The existing `CudaView` uses CUDA array interface version 2 and carries no stream field (`715–725`). A view retains its owner and is not an owned snapshot.
- `csrc/batch.cu:163–174` gathers positions, velocities and quaternions from the same solver arrays in one kernel. The quaternion order is **XYZW**. Entry `i` is the director of segment `i→i+1`; for 41 nodes only entries 0–39 are real segment frames. Entry 40 is unused padding.
- `batch.cu:388–399` executes all solver substeps, calls `SimAccess::finish`, calls `sdf.end_step` and then queues `gather_nolock`. `YarnBall.cu:311–316` shows that the error check synchronizes the solver stream; **the final gather is queued afterwards**, so the earlier error check alone does not prove that the gathered output has finished.
- `batch.cu:422–426` queues that gather on the solver stream. `YarnBall.cu:231` creates an ordinary CUDA stream. The installed `python/ropesim/__init__.py` explicitly requires Torch's default stream and documents legacy-stream ordering.
- The new readback explicitly rejects a nondefault Torch stream, calls `torch.cuda.synchronize(device)`, obtains the existing views, checks their actual device and float32 dtype, and performs owned clones followed by blocking CPU copies. Synchronization and copying do not advance SDK or solver physics. No `gather`, `step`, `step_free`, `reset_envs`, setter or force method is called.

Pinned sources include:

| Source | SHA256 |
| --- | --- |
| `csrc/bindings.cpp` | `89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5` |
| `csrc/batch.cu` | `899e95da5aab86a40442be8effbcadbd6d2923a27b9554475d38f6fef631d868` |
| `csrc/batch.h` | `7b057d1e2bbfaa812f4a8c0dd201229e54a24e7a2d6a2dbfb017126a7f1d7010` |
| installed `python/ropesim/__init__.py` | `adc864e6bed274b88a0b15f44dcf8c2628cae5d6913518d147d5d8b4ac0d8419` |

The existing marker source contract also checks the exact Cosserat and YarnBall sources. Local +X is the segment director axis; its local Y/Z rotation retains material spin instead of reconstructing it from the geometric centerline tangent.

## Important rest-length provenance correction

`rest_lengths()` is bound on **RopeSim**, not **RopeBatch**. The historical marker report's reference to an existing rest-length getter must not be interpreted as a getter on the currently used batched object.

`FreeRodRuntime.__init__:147–148` stores float32 construction nodes and computes `self.rest` as their neighboring-node distance. `RopeBatch` constructs the same input nodes before `Sim::configure` (`batch.cu:211–240`), but the current Python binding does not expose the resulting native material rest lengths.

The new adapter freezes those constructor nodes/rest values, verifies the actual reset positions against them, and compares every supplied runtime rest array to that fixed reference. It explicitly outputs `native_rest_lengths_readback_m=None` and names the constructor provenance. It never recomputes rest length from the current stretched centerline and never calls an invented batched getter.

If direct native rest introspection becomes necessary, a separate read-only `RopeBatch.rest_lengths` binding could copy the existing host `sim->verts[e*V+i].lRest` under `b.mutex()` into `[N,V-1]`, analogously to existing per-vertex introspection. That would require a new source/binary identity and separate validation; no such patch or compilation was performed here.

## Small opt-in interface

`MaterialDirectorReadback(constructor_nodes_m, device="cuda:0")` retains only owned reference arrays, source metadata, scalar identities and a weak reference to the same actual RopeBatch instance.

- `capture_reset(row, material_reference_uid=..., rope=..., backend=..., sim=..., read_clock=...)` reads an **already completed** actual reset. It does not perform a reset. A new episode/material UID, increasing reset ID, zero rod tick, invalid-until-stepped contact state and exact reset node values are required.
- `capture(row, rope=..., backend=..., sim=..., read_clock=...)` reads each subsequent completed SDK endpoint. Exactly one SDK step and 12 solver substeps must have elapsed, with stable run/env/episode/reset/rod and material reference.
- The actual loaded backend file path/hash and RopeBatch type are checked. Subsequent calls must borrow the same actual RopeBatch instance; a different object with numerically identical state is rejected.
- Before and after the copies, the existing ten-field SDK/rod clock validator, actual SimulationContext step/time, actual `rope.time` and `last_error` must agree. Copied positions must equal the same-clock completed row. Real segment quaternions must be finite and unit length; the finite padding value is saved separately and excluded from segment validity.
- A failed capture poisons ordinary continuation. A subsequent reset requires a fresh episode/material reference and monotonic SDK/solver clocks relative to **both** the most recent reset attempt and the most recent successful sample.
- Borrowed solver/backend/simulation/callback/view/tensor aliases are cleared in finally blocks. Outputs contain owned CPU arrays and scalars. `source_instance_id` is process-local identity, not a cross-run identity.

A future explicitly scoped owner can call reset capture after `DualHandRodRuntime.reset` returns and ordinary capture after its completed `step` returns. It must supply the existing runtime's exact `rope` and `clock`, and the actual simulation context. Joining the packet with measured hand poses requires the same explicit clock identity. No automatic installation, owner hook, recipe or launcher is added.

The caller must hold the existing owner's exclusive read phase. The API does not expose a mutex spanning its Python getters. Before/after clock equality detects drift but cannot make unrelated concurrent resets safe; this package does not advertise independent thread-safe atomicity.

## Remaining material-Slip gap

The output is a **completed-physics-endpoint** director snapshot. It does not recover the 12-substep director history, intermediate excursions, or an exact contact-kernel state. Raw contact pairs are produced before the per-thread position commit, whereas these arrays come from the completed gather. Their identity is not a proof of contact-marker correspondence.

The adapter can supply directors and nodes to the already existing analytic fixed-marker reference, and the CPU tests exercise that integration. It does not repeat its mathematics. A fixed marker's rest coordinate is not the current contact station, and a node ID is not a continuous material coordinate.

The output therefore retains `continuous_contact_s_m=None`, `r0_tangential_slip_m=None`, `physical_error_bound_m=None`, unavailable substep history and unverified pair/substep correspondence. Native/Teacher qualification remain false. No R0 coefficients, 0.5 mm gate, learned observation schema or original source admission is changed.

The previously documented bounded substep telemetry site remains `SimAccess::substep` after `cudaGraphLaunch` and before the next iteration, with a separate same-stream capture kernel. This endpoint adapter does not implement that ABI change or solve the continuous node-sphere/contact-material correspondence problem.

## CPU verification

Evidence directory:

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/material-director-readback-cpu-d1aqeb45`

Final command:

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_material_director_readback.py tests/test_cosserat_material_marker_reference.py tests/test_contact_material_observability.py tests/test_deformable_pair_source_basis.py tests/test_dh116_deformable_grasp_task.py`

**111 passed in 9.81 s**, exit 0: **30 new + 81 related**. There were no skipped tests. This used explicit CPU CUDA-view/stream protocol doubles and real source/binary byte inspection, not a CUDA context, extension import, native solver or new physical evidence.

Tests cover default-stream rejection, synchronization/copy order, real API names and source ordering, padding semantics, fixed-marker spin integration, owned output independence, exact backend/type/instance, reset/episode/run/env/rod continuity, skipped/repeated SDK steps, solver tick/time drift, clock changes during getters, position/rest/director mismatch, error state poisoning, reset recovery and weak-reference ownership. An independent import subprocess confirms no Isaac/USD/ropesim import and no initialized Torch CUDA context.

The initial missing-module run records 2 failures/24 setup errors; a subsequent import-path mistake was preserved and corrected. The first implementation passed 26 focused tests and then 109 related tests. Independent read-only review identified the failed-reset clock regression; two new negative tests reproduced it before the fix. Final 111-test verification includes that correction.

| Evidence | SHA256 |
| --- | --- |
| `final.log` | `9080fa012a0942590f7d864d8a882b0b909c2b64c76d5d21adcd615e14a06e8b` |
| `final-command.json` | `daf507eb868337d206f920532c47e7b954ba8bc32b34ca7ca1dd66f3cfab446a` |
| `review-red.log` | `65314a9fb8f4c1c20226be90480dd456b1d3dd79238013d228484d310d484c76` |
| `source-basis.json` | `1041a316417f7bb55de71f843e7ebef3940567e8370ae097be59f6baf789cbaf` |

The peer review was source-only and did not execute these tests or any native work. No native director sample has yet been collected by this adapter.
