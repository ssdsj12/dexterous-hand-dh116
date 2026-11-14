# Contact material observability after left Pinch v4

This independent CPU package inspects the completed v4 records and the exact existing pair-build CUDA sources. It does not implement an R0 slip provider: the available contact tuple cannot establish a continuous contact material coordinate, fixed surface-material correspondence, or an error bound adequate for the unchanged 0.5 mm gate. No native process, compiler, optimizer, physics/configuration change, or release was run by this package.

Base: `5c0b71fd410718d76890e0e6f597a59970be607b`. Only the new audit CLI, tests and this report are included. The frozen v4 entry, configuration, validator, runtime and original R0 reducer remain unchanged.

## Executable slice and source proof

`scripts/audit_dh116_contact_material_observability.py` verifies five exact source hashes: `contact.cuh`, `iteration.cu`, `YarnBall.h`, `csrc/sdf.cu` and `csrc/bindings.cpp` from `teacher-pair-build-zQEenV/source`. These are existing build-tree sources, not a fresh compilation or new binary/source attestation. The recorded backend remains `92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0`.

It independently replays the original precision validator, exact source-bound v4 actions/targets, and raw atomic pair reconciliation, then compares only solver substep 11 with its simultaneous actual final node and WXYZ pad-pose readbacks. The point convention `point = x - R*n` is verified against the stored node via `point + R*n`. The radius is the frozen constructor setter value; no unsupported native contact getter is invented. A 2 µm numerical consistency check is explicitly not a physical slip error bound.

The helper retains each original node identity and first pad-frame normal; it never reselects a closer node. Its reported fixed-node centerline displacement in the first pad tangent plane is a diagnostic, with `continuous_material_s_m=None` and `physical_slip_m=None`. A missing endpoint is not filled from aggregate force. Complete raw captures, exact clock/slot/body identity and copied actual arrays are prerequisites.

Actual source details:

- `contact.cuh:27–45` applies regularized Coulomb friction to the substep relative displacement.
- `contact.cuh:94–109` uses a node sphere, SDF normal and relative mesh motion. It does not receive a rod cross-section director.
- `contact.cuh:165–206` records only substep, node, mesh, contact point, normal, force and torque. It does not record segment/barycentric material contact coordinates or `dxRel`.
- `iteration.cu:17–32` interpolates the mesh pose at `(sub+1)/nSub`. `114–120` performs readout at the final iterate before committing the node position.
- `bindings.cpp:856–860` exposes final `RopeBatch.quaternions()`; the current ledger does not capture it, and pair records do not contain substep directors.

For continuously contacting nodes, the recorded `point+R*n` does allow denser node-center trajectories, and known mesh interpolation can supply a reconstructed pad frame. This is useful motion evidence, but it does not assign a persistent surface-material marker or a continuous contact station. This package deliberately uses the final-substep actual pad readback instead of calling reconstructed poses live readbacks.

## Completed v4 evidence

Read-only input:

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-left-pinch-v4-20261113-lvr575/native-runs/flex.f4-pinch-left-v4-native/attempt-wezd3fyw/native`

All 238 rows were replayed. Raw hand pair samples total 1165; 567 solver substeps have simultaneous positive thumb/index normal load. These counts are contact observations, not effective opposition or a grasp.

| Diagnostic | Thumb | Index |
|---|---:|---:|
| Actual material node identity | 35 | 34 |
| Consecutive final-substep samples | 50 | 48 |
| First/last measured phase | 189/238 | 191/238 |
| Maximum fixed-centerline tangent displacement | 0.0613856 mm | 0.0545986 mm |

These small centerline values **do not establish the 0.5 mm surface/material-slip condition**. Cross-section rotation, contact-station migration and material correspondence remain unmeasured. The same node labels are used for identity only.

The actual reset has 41 nodes and total rest length 0.5 m. Maximum station spacing is 0.012500002980232239 m. The conditional nearest-station bound is 0.006250001490116119 m; this theorem applies only if an arbitrary continuous coordinate is actually assigned to its nearest station. The captured physics does not provide that assignment, so this remains neither a physical contact-anchor error bound nor a slip bound.

## Why force cannot fill the missing coordinate

The source-mapped pad coefficients are `mu_s=1.0`, `mu_k=0.8`, `v_slip=0.001 m/s`, obtained from frozen constructor settings and the original hand material contract, not measured material calibration. Actual captured `|Ft|/N` ranges from 0.4251031283 to 0.9963305632.

The CPU scalar reference of the exact real-arithmetic source law gives:

- `|Ft|/N=0.9` for both 0.0009 and 0.0015 m/s.
- `|Ft|/N=0.8` at every speed above 0.002 m/s. The reference demonstrates equal force at displacements of 0.000347222 mm and 1 mm in one 1/5760 s substep.

The latter is a nonidentifiability counterexample, not a claim that the rod attained the example's large speed. Force alone cannot select the branch or supply an upper displacement bound. Actual motion samples can constrain centerline motion where consecutive contact data exist; they still do not recover the missing surface-material correspondence.

A second reference keeps the node-sphere geometry identical but rotates the unobserved cross-section material by 0.2 rad at 5 mm radius: surface arc travel is 1 mm. This is likewise an observability counterexample, not a claim of measured twist in v4.

## Current R0 requirements and next source boundary

The unchanged `FlexibleGraspEpisode._slip` requires bilateral valid contact geometry, fixed material-reference identity, distinct continuous hand contact stations, anchor/current material uncertainty and fixed-pad tangential displacement/uncertainty. On initial binding `2*max(material_uncertainty) <= 0.0005` is necessary. It then adds current and anchor uncertainties and accepts only when every drift plus uncertainty is within 0.0005 m.

A genuine successor source needs a defined and validated continuous contact correspondence or independently observed material marker, same-contact-substep rod state and directors, persistent pad/material identities across contact loss and node switching, and an independently supported combined uncertainty bound. Adding a geometric nearest-point projection alone cannot provide these. Because current contact physics acts on node spheres, introducing a continuous segment-contact interpretation also requires a justified model/calibration boundary; it is not a harmless relabeling of node IDs.

All R0 material station, tangential slip and slip-acceptance outputs remain `None`. Native/Teacher qualification remain false.

## Simulation cleanup and exact observed references

The v4 helper receipt records explicit ready-subscription unsubscribe, field clear and original clear returning, with zero extra GC and zero physics steps. Receipt SHA:

`ef29ddc105acce0fa738d25cc5e4c4beb39adb8bd95d21f74a7e3da23c5ba572`

Before and after app close, the owner JSON bytes have the same SHA:

`6daa22643429c25f9846d5c991b64a0b43cc9f48a249c08c3ef192fde8e7dac7`

Simulation `140719542273264` remains alive with exactly two recorded direct Python references:

1. IsaacLab `SimulationContext.__init__:290` frame `93825252662528`.
2. Frozen acquisition `<lambda>:525` frame `140719542053952`.

The selected constructor frame has one observed upstream frame: `torch.cuda._lazy_init:346`, object `140719542052416`. The installed Torch file SHA is `4bd2da05273edd47203133b45843ece0e59234af28204f2ed5cb10d6175a4c67`; line 346 is its initialization lock context. The trace did not follow the upstream owner of this Torch frame or the acquire lambda, so it does not identify the ultimate retaining object, an active lock wait, a native thread, or a new shutdown root cause. The ready bound method present in v3 is absent from the final v4 direct refs. No arbitrary frame was cleared or extra GC run here.

The parent handles the GPU table-filter warning and final lifecycle/force qualification. This package does not reinterpret missing supported table-contact measurement as zero.

## CPU verification and immutable evidence

Evidence directory:

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/contact-material-observability-cpu-p9x1tieo`

Verified command:

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_contact_material_observability.py tests/test_deformable_pair_source_basis.py tests/test_dh116_pinch_force_precision.py tests/test_dh116_deformable_grasp_task.py`

Result: **66 passed in 7.69 s**, exit 0: 11 new + 55 related regressions. The first regression invocation misspelled the historical task test filename and exited 4 with zero tests; its `final-cpu.log/json` are preserved. The corrected verified run is the result above.

- `verified-cpu.log` SHA `45b8df26b41fe2fbc1403e243a71bdba941dd3d8f8b3ec079e90e5c4974742b1`
- `verified-cpu.json` SHA `943bee121880175b81fbeeacd52e9aee0c3e2dadbd7c2bf3e651af1b1a5ce852`
- `v4-material-audit-final.json` SHA `d6b6f861df45bf078bee16dfeccd88765d064287eb2b5a2a2a3e833ce41c792c`
- `audit-final-command.json` SHA `f2e1002b03085e86b1490f3b236dab2fdbd4238c92478a975deb1c2cb31db7fc`

The audit output embeds SHA256 for every consumed original native artifact and the five inspected source files. CLI output uses exclusive creation and refuses existing paths before reading inputs. The tests exercise source changes, node/substep/radius/pose mismatch, rigid-frame invariance, force ambiguity, unobserved rotation and preservation of prior evidence. No original evidence was changed.
