# DeformX mesh observations: independent preparation and build

Date: 2026-11-14. Task base: `56a2028afeabb21b2885213eef5440e209d6b610`.
Branch: `codex/mesh-pose-instrumentation-prep-20261114`.
This package adds a source-locked preparation script, CPU source tests, and this report. It does not alter any old project runtime, recipe, source lock, SDK source, or installed backend. Root owns integration and any later native experiment.

## Result and limits

The isolated extension configured, compiled, and installed into its new local prefix once, successfully. After compilation, a separately authorized CPU import-only check verified the new binary path/SHA and four exported methods. No RopeBatch/RopeSim instance was constructed and no native method was called. No CUDA solver, Isaac, simulation step, PPO update, native calibration, or default-backend switch occurred.

New binary:
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114/build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so`

SHA256: `8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340`.

Build success is not API/GPU/Reset/contact/Teacher qualification. All those qualification fields remain false. In particular the old projected Pinch evidence does not prove a correct first-step mesh reset, or strict positive/negative contact classification. The new interfaces are intended to supply missing direct native evidence in a separately approved successor.

## Source evidence and minimal change

The read-only base is `parameter-getters-build-e96wtu/source`, at upstream commit `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` plus the previously locked local delta. The complete 72-file inventory digest is `3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472`. Its old binary SHA is `2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec`.

Exact original source locations:

- `third_party/yarnball/KittenEngine/YarnBall/YarnBall.h:176–179`: the original pair record has substep/node/mesh and point/normal/force/torque, but no mesh pose or gap.
- `sim/contact.cuh:94–109`: `sdfContactAt` evaluates the actual transformed local probe, assigns `gap = phi - contactRadius`, and returns only negative-gap, nonzero-gradient contacts.
- `sim/contact.cuh:27–43`: `surfaceContact` computes the penalty normal force `N = -k * gap` before friction.
- `sim/contact.cuh:189–202`: `surfaceReadout` holds the same actual `P`, gap, final displacement, and force evaluation used by the existing pair record.
- `sim/iteration.cu:10–34,114–120`: `initItr` assigns each substep mesh pose; `endItr` invokes the force/pair readout at the final substep iterate. Neither file nor its call order changes.
- `csrc/sdf.h:112–116`: actual device current/target/teleport/solver-pose buffers. `sdf.cu:169–175` stages targets, `55–62` consumes teleport at step start, and `67–86` commits a slot. The new getter calls none of these mutators.
- `csrc/sdf.cu:233–243`: the existing pair getter synchronizes the actual solver stream before owned copies.
- `csrc/bindings.cpp:1072–1112`: pair capture is bounded and one-environment only; the old serialization remains byte-for-byte unchanged.

Only five source files change in the separate tree: `YarnBall.h`, `sim/contact.cuh`, `csrc/sdf.h`, `csrc/sdf.cu`, and `csrc/bindings.cpp`. The patch adds no equation, material change, root constraint, solve iteration, or step. The existing arithmetic writes `N` to an optional output for the diagnostic record; it does not feed that output back into force or Hessian calculation. A larger record and extra stores can affect performance/compiler behavior; this package does not claim GPU bitwise equivalence or zero overhead.

## New API contracts

`RopeBatch.contact_pair_samples_v2()` preserves the old `indices[K,3]` and `vectors[K,12]` fields and adds owned float32:

- `mesh_positions_world_m[K,3]`, `mesh_quaternions_xyzw[K,4]`: the actual copied `P.pos/P.q` used for that readout.
- `surface_gap_m[K]`: the native `phi - R` result, not depth inferred from a later node endpoint.
- `normal_force_n[K]`: the unaveraged penalty normal magnitude from that same native evaluation, before friction.
- Schema `dh116.sdf_pair_samples.v2`, original capacity/attempted/overflow/substep fields and aggregate readbacks.

The record still contains only accepted contact readout events. It is not a dense all-node/all-mesh gap sampler. Empty output alone does not establish contact absence; overflow remains incomplete. New API qualification and consistency with old aggregate/force laws need actual bounded validation.

`RopeBatch.mesh_state_snapshot()` takes the owner mutex and synchronizes its solver stream, then copies `d_cur/d_tgt/d_tele/d_poses`. It returns owned current/target/solver position and XYZW quaternion arrays `[N,M,3/4]`, and uint8 teleport flags `[N,M]`, compacted using the original stride `e*MAX+m`. Only active slots are exported. Host `pending_set/moving/pair_capacity/pair_ready` are separately identified as native host state. The snapshot performs no commit, upload, target staging, step, or implicit reset.

Both new bindings record actual `b.raw()->meta.time` before/after the read under the mutex and the actual last-error code. These are **owner metadata times at readback**, not per-substep measured timestamps. Pair `substep_timestamp_s` is explicitly null; the original integer substep index remains available. No `n/480` or interpolated time is presented as a measurement.

Future Reset validation can compare actual current/target/tele flags immediately after construction, after deferred teleport, and after same-slot reconstruction without consuming a solver step. Future moving-contact validation can inspect actual recorded substep poses and gaps. Neither is executed or qualified here. Existing source-qualified readers will reject this new binary until an independent successor explicitly binds its full source/binary identities.

## Preparation and actual build

Preparation validates all original bytes and exact anchors before creating a new destination; it refuses existing or protected destinations. It copies all 72 sources, applies only reviewed replacements, preserves the Python facade, and writes explicit commands. It never invokes a subprocess.

New root:
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114`.

Reviewable delta:
`build/observation-only.patch`, SHA `1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41`.
The `build/preparation.json` is the pre-build snapshot and deliberately retains `compiled=false`; the separate actual build result is authoritative.

After root and physics source review, root authorized the isolated build under the existing environment-preparation authorization. Each phase ran once; no retry:

| Phase | Limit | Actual duration | Exit |
| --- | ---: | ---: | ---: |
| configure | 60 s | 6.330554 s | 0 |
| build | 180 s | 35.281586 s | 0 |
| install | 30 s | 0.064633 s | 0 |

`CUDA_VISIBLE_DEVICES=''`, two build jobs, `OMP_NUM_THREADS=2`; Release, CUDA 12.8.61, architecture 89, GCC 12.3.0, Python 3.10.12, pybind11 2.13.6. The exact command vectors are in the evidence `isolated-build/commands.json`; CMakeCache toolchain entries and hash are in `toolchain-cache.json`. Install prefix is only the new `build/python`, with no pip/environment installation.

The old source inventory, new source inventory, and old binary were hashed before and after all phases. They are identical. The complete snapshots share SHA `d9ad3b9757f19248ff7fdacaf048101c168a0af41df60108f095e3e8691097f9`.

The extra pair fields comprise 36 scalar bytes per record before any implementation padding. At the task's 41×11×12 capacity this is 194,832 additional scalar bytes; this estimate is not a measured runtime allocation/performance result.

## CPU import-only ABI check

After the build, root authorized one additional import-only process, with a 30-second cap, GPU visibility empty and PYTHONPATH pointing only at the new build prefix. It exited 0 in **0.516222 s** on Python 3.10.12. The actual module path and SHA matched the new binary above. Both new methods and old `contact_pair_samples`/`parameter_snapshot` were callable exported attributes with nonempty documentation. No owner was constructed, no method was invoked, and torch was not imported.

Evidence is `isolated-import/{command.json,stdout.txt,stderr.txt,result.json}`. Result SHA: `7fcee5aecaa1fdd9d6faa42f3c88cffbe9e5520ef8df9568110d03f266ce6bdd`; stdout SHA: `1ca0d592d78f22251b256398d36ffc89bbeef111b310cab5809164cc99381d27`. The earlier build result's `extension_imported=false` remains an accurate snapshot of that earlier phase. This separate check proves import/export compatibility only, not GPU/API numerical or Reset behavior.

## CPU verification and evidence

Evidence root:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-observation-prep-cpu-cca16or5`.

TDD started with an actual missing-module collection error in `red-missing-module.txt`; after implementation 17 source tests passed. The final timestamp-contract test brings the new tests to 18. Final relevant regression: **67 passed in 6.97 s (18 new + 49 existing)**, command:

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short tests/test_deformx_mesh_observations_preparation.py tests/test_dh116_deformx_parameter_getters.py tests/test_dh116_deformx_pair_capture.py tests/test_dh116_deformx_source_lock.py tests/test_dh116_deformx_cuda_source.py
```

Tests cover exact reverse patches, only five changed files, untouched solver order/source, old pair serializer bytes, native sample provenance, owned getter layout and stream synchronization, rejection of changed originals before mkdir, protected/existing destinations, copy-only preparation and explicit bounded commands, and actual versus unavailable timestamp semantics. They are CPU source/preparation tests, not GPU numerical assertions.

`Repository.verify_protection()` exited 0. Final hashes:

| Artifact | SHA256 |
| --- | --- |
| `related-regression-final.txt` | `b9af37e8d5e92a240d312ebc564687cc3c07101a15093c1f343029345432b45c` |
| `red-missing-module.txt` | `b75e8f16a85bdfb0535206ce75a3dbb0be45fbffdcbff6c8e015407e20d09d22` |
| `protection.txt` | `af5d8a21858f42808c9bcc17012b77bcc2fd6ab3b071dbd4a05f13f410470541` |
| `isolated-build/result.json` | `e04d1f55493bbb1dd21309cc1e25232980866146ca42fe9811e57e7171fd4038` |
| `isolated-build/toolchain-cache.json` | `fcd7bf8d39ecf43856d02e49c0a64f2a5ccb17b66c160b86e7548ae7a0dd2f25` |

Root and physics independently reviewed the five-file delta before compilation. Physics reviewed the preparation source SHA `213115e0ea79195769a386d94306e1e30dac7a08c8a2317199e67bf927939d45`, with no additional shape/ownership/timing blocker identified. That review did not run GPU code or inspect a native result. No native recipe or registration is added in this package.
