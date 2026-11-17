# Fixed Cosserat material marker: CPU reference and exact telemetry boundary

This package supplies a narrow mathematical reference for fixed material-marker transport, plus exact source locations for the missing telemetry. It does not provide current native Slip qualification. No old API, task, runtime, configuration, compiler output, native process, training or checkpoint was changed.

Base: `a6bb0dcc292540572299bddd5df6ce675559797d`. Three new files only: `tabletop_cosserat_material_marker_reference.py`, its CPU tests and this report.

## What the reference computes

The original material coordinate is fixed at

`s = sum(rest_lengths[:i]) + alpha * rest_lengths[i]`.

For a caller-selected segment `i`, fixed fraction `alpha` and fixed cross-section material offset `rho=(0,rho_y,rho_z)`, the reference evaluates

`x_marker(t) = (1-alpha)*x_i(t) + alpha*x_(i+1)(t) + Q_i(t)*rho`.

The original solver uses `Q_i*(1,0,0)` as segment `i→i+1` material tangent. Thus local Y/Z span the director cross-section. The quaternion is the actual material director convention, **not** a frame reconstructed from the geometric centerline tangent. Shear and twist are retained. No interpolation of neighboring segment directors is silently introduced.

This is an explicit piecewise linear centerline plus rigid director cross-section reconstruction hypothesis. It is not the original node-sphere collision geometry, a calibrated physical cross-section, or a measured current contact station.

The marker and pad reference are frozen dataclasses containing owned scalar tuples. Original rest-length bytes, material UID, segment, fraction and offset are fixed. Changed rest coordinates, invalid quaternion/pose/normal, nonfinite values and accumulated-coordinate overflow are rejected.

At the initial reference, the caller supplies the same-contact normal expressed in the pad material frame. The reference freezes that normal and the initial local marker position. At later samples it transforms the same marker into the current pad frame and computes:

`delta = a(t) - a(0)`,
`delta_t = delta - n0*(delta dot n0)`.

This measures displacement in the fixed initial pad reference plane. It does not integrate friction force, replace the reference normal every frame, infer contact from a nearby point, or reselect a material point. It is displacement, not accumulated path length; an out-and-back trajectory can return to zero. A future verifier would retain all substep values and their maximum instead of examining only the final endpoint.

The result deliberately retains `current_contact_s_m=None`, `r0_tangential_slip_m=None`, `physical_error_bound_m=None`, `native_clock_alignment_verified=False` and `contact_marker_correspondence_verified=False`. Native/Teacher qualification remain false.

## Analytic evidence

Tests cover fixed rest coordinates, translation, common rigid rotation/translation, axial stretch, bending with a chosen segment director, geometric shear, quaternion sign equivalence and unused final padding.

A 0.2 rad material-director spin about the straight rod's local X leaves every centerline node unchanged but moves a 5 mm radius marker by `0.005*sin(0.2)=0.000993346654 m` in the fixed tangent plane. A centerline-only tracker would miss this approximately 0.993 mm displacement. This is analytic data, not a claim that native v4 experienced that twist.

Axial stretch changes world positions while the marker's original rest `s` stays fixed. Common rigid rod/pad motion cancels in the pad frame. A separate test shows that the same node station may belong to two adjacent segment frames with different offsets: node identity alone does not select a unique surface material marker.

## Source-pinned availability

Source root:

`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source`

The helper verifies the exact SHA256 of:

- `sim/cosserat.cu`: `7c9151c1c4df64c4735b7df69de605d030bbbb99e8b9638095ef99fb3dbddb03`. Lines 76–83 use local +X in stretching; 249–271 update the segment director.
- `YarnBall.cu`: `06af0339837c196cc8f716b6095b50e5409cbe7544c01d9da8a9067f0b801e68`. Lines 145–169 initialize segment frames from +X with twist transport.
- `csrc/bindings.cpp`: `89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5`. Lines 380–389 define segment quaternion XYZW and unused last padding; 856–860 expose the real `RopeBatch.quaternions()` view.

Additional instrumentation source hashes are recorded in the private `source-audit.json`; nothing was compiled or imported as an SDK.

| Needed quantity | Already available | Missing evidence |
|---|---|---|
| Original material rest lengths | Existing rest-length getter and actual reset readback | A physical continuous contact station cannot be inferred from these node labels |
| Final nodes and segment directors | `RopeBatch.positions()/quaternions()`; `gatherK` copies solver arrays | Current physics ledger stores positions but not directors |
| Actual substep node/director state | Device `d_verts[].pos` and `d_qs[]` | No bounded host-readable history |
| Actual substep mesh pose | Device `d_sdfPoses` written by `initItr` | No direct pose-history getter; submitted target is not a readback |
| Contact tuple | Existing substep/node/mesh/point/normal/force/torque capture | No continuous segment fraction, surface material marker or material uncertainty |
| Fixed reference normal | Can be bound from a genuine same-substep contact and mesh pose | Correspondence between that contact and the selected material marker must first be verified |

## Minimal genuine instrumentation path

An endpoint-only investigation can first clone the existing final `positions()` and `quaternions()` views together, without modifying the solver API. It must preserve the actual owner/stream/clock boundary, rest/reference UID, pad identity and fresh reset identity; current native code does not yet provide this binding. This would improve marker transport evidence but still miss excursions between physics boundaries and still not measure current continuous contact `s`.

For a complete substep history, the smallest defensible instrumentation site is `csrc/batch.cu:67–73`, inside `SimAccess::substep`, **after** `cudaGraphLaunch(s->stepGraph,s->stream)` and before the next `startIterate`. The graph contains all Cosserat sweeps and `endIterate` (`sim/step.cpp:19–22`). A bounded capture kernel queued on the same stream can then copy the completed node positions, segment directors and the actual `d_sdfPoses` associated with that substep. This ordering supplies a kernel boundary after the position commit.

Do not read neighboring `verts[i+1].pos` opportunistically from inside `endItr`: other threads may have committed that neighbor while the current thread has not. `iteration.cu:114–120` reads contact before per-thread `pos += dx`; a new neighbor read there risks mixing old and new states. A separate capture kernel after the existing graph avoids that race.

For one 41-node rod, 40 real segment directors, six meshes and 12 substeps, a simple complete snapshot is:

- 41×3 float32 node positions;
- 40×4 float32 segment directors;
- 6×7 float32 actual mesh poses;
- repeated for 12 substeps, plus explicit frame/substep/rod/reset identities and validity metadata.

The numeric payload is 15,600 bytes per physics step. This is a bounded baseline, not an implementation or a performance measurement. A smaller predeclared-marker capture can copy only each selected marker's two endpoints/director and named pad pose, but it must continue through no-contact intervals, include all referenced fields, and may not drop or reselect markers based on force. The existing force/pair buffer can remain unchanged; a separate opt-in telemetry buffer avoids altering the frozen pair ABI.

On every reset invalidate the history, bind actual reset node/director state and a fresh episode/material-reference UID, and publish validity only after the completed solver substep. Detect incomplete/overflowed buffers, mismatched SDK/solver clocks, missing mesh identities and partial solver failure; do not insert zero measurements or a fabricated rollback. The current owner/reset packages remain the necessary integration boundary.

`dxRel` at `contact.cuh:94–109` can optionally be logged for friction-law diagnosis. It is not necessary for the position-based material-marker reference and must not be integrated as a substitute for surface marker motion.

This proposed instrumentation requires a new source/binary identity and separately approved native validation. It was not implemented or rebuilt in this CPU package.

## Remaining condition that telemetry alone cannot fix

The current contact model is a node sphere against an SDF. It has no segment barycentric contact coordinate. Its reported surface point is `x_node - R*n`, independent of the segment director. At a node junction, the corresponding material director is not uniquely selected; a sphere point can also have a nonzero component along a chosen segment's local X. Dropping that component or choosing the nearest segment would silently change the physical meaning of the point.

A genuine Slip source therefore still needs a justified continuous material-contact correspondence or independently observed persistent surface marker, a defined relation between the marker reconstruction and the actual contact model, and validated error bounds. Introducing continuous segment-contact physics would be a physical-model change, not merely adding telemetry.

Original `FlexibleGraspEpisode._slip` also requires bilateral valid geometry and current contact stations/uncertainties. Freezing marker `s` is correct for tracking a material marker; feeding this constant value as the **current contact station** would manufacture zero material-station drift. This helper does not do so.

The unchanged 0.5 mm gate includes reference plus current material uncertainty and tangential uncertainty. No native director readback, marker correspondence, geometric-model discrepancy or uncertainty has been measured here. None can be filled with zero on the strength of analytic tests.

## Verification

Evidence directory:

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/material-marker-reference-cpu-ni7hzlng`

Final command:

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_cosserat_material_marker_reference.py tests/test_contact_material_observability.py tests/test_deformable_pair_source_basis.py tests/test_dh116_deformable_grasp_task.py`

**81 passed in 7.26 s**, exit 0: 20 new and 61 related regressions. All source-pinned tests ran on Ubuntu; no test was skipped. Earlier 80-pass evidence is preserved; the final run adds the coordinate-overflow guard.

- `final-cpu.log` SHA `c14701b527966af46c30b02319e1dd28547fcdb8e5b4babade6e25ef834ebdb3`
- `final-command.json` SHA `ade60bbfbf769d29d5d7a4226125221e0c66e14aa0f13da9827dae90e5a9a46a`
- `source-audit.json` SHA `93e2d9dc8a0b62ec8c59a6b71008e8175b5901398fcb6efb5839a20cf2173816`

No native experiment, source rebuild, old API change or training admission occurred. A peer review was requested but the peer was handling another urgent task; no completed independent review is claimed.
