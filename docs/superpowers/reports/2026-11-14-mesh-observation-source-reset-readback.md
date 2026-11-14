# Explicit mesh-observation source profile and direct reset readback preparation

This independent CPU package binds the newly compiled observation backend and prepares direct reset readback. It does not run Isaac, instantiate RopeBatch, call a native getter, execute CUDA kernels, build anything, publish, create a native recipe, or qualify Reset/contact/Slip/Teacher. The original 2fea backend, default source functions, frozen recipes, physics, source grids, and historical evidence remain unchanged.

Base: 659d495141e33b61765ac5ab82f33f4af6d53ece. Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observation-profile-20261114. Branch: codex/mesh-observation-profile-20261114.

## Source evidence

New binary: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/mesh-observations-build-20261114/build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so.

- Binary SHA256: 8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340.
- Canonical 72-file inventory SHA256: 34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64.
- Preparation JSON SHA256: 612696a63262571ab9910ea88421f532f583e7bdb4174851f21a46de662537b9.
- Actual observation-only patch SHA256: 1918ebebfde3a00a9c0c3bf979313674d8015f9f5fa42220767b562d169bde41.
- Patcher SHA256: 213115e0ea79195769a386d94306e1e30dac7a08c8a2317199e67bf927939d45.
- Separate authorized build receipt SHA256: e04d1f55493bbb1dd21309cc1e25232980866146ca42fe9811e57e7171fd4038.
- Separate authorized CPU import-only receipt SHA256: 7fcee5aecaa1fdd9d6faa42f3c88cffbe9e5520ef8df9568110d03f266ce6bdd. Its command/stdout/stderr bytes are also pinned. No API numeric/GPU/reset claim is inherited.

source_basis() rereads all 72 new files and all 72 original getter files, the old and new binaries, exact preparation/patch/build/import evidence, original package/SDF bytes, local patcher, source-lock JSON, and unchanged material configuration. Exactly five changed paths are permitted: csrc/bindings.cpp, csrc/sdf.cu, csrc/sdf.h, YarnBall.h, and sim/contact.cuh. The 67 remaining source files must match the original inventory. It does not validate a replacement binary by changing an expected SHA alone.

The relevant compiled source facts are sdf.cu:140-161 (add synchronously commits its slot; L156 saves pendingSet and L160 restores it), sdf.cu:195-202 (clear invalidates pair capture, clears grids, sets pending/moving false), sdf.cu:329-352 (solver-stream synchronization and owned D2H current/target/teleport/solver copies), and bindings.cpp:1131-1173 (batch mutex, compact N×M arrays, host metadata time/error). The snapshot never stages, commits, resets, or steps. The source change can affect instrumentation cost/register layout; GPU bitwise equivalence is not asserted.

## APIs and integration boundary

- tabletop_mesh_observation_source: VERSION, PARAMETER_VERSION, BUILD, BACKEND, BACKEND_SHA, source_basis(), verify_runtime(rod), source_scope().
- tabletop_mesh_reset_readback: capture_mesh_state(sim, rod), validate_capture(packet), validate_reset_readback(packet, *, reset_receipt, mesh_plan), MeasuredMeshReset(seed, *, episode_uid, mesh_plan).run(sim, rod, bindings, *, owner_resets, record_readback=None).

source_scope must be the outermost explicit scope around new preflight/live execution and disk post-exit validation. It binds only parameter reader VERSION/BUILD/BACKEND/BACKEND_SHA/source_basis, builder BACKEND/BACKEND_SHA, and health's copied BACKEND/BACKEND_SHA. Existing parameter live and audit modules alias the same reader; health obtains parameter VERSION at function execution. Nested same-profile scopes are permitted on the same thread. Foreign partial bindings and other threads are rejected. Defaults restore on success or exception. This is process-local opt-in preparation, not a global backend migration.

The existing parameter endpoint class keeps its original implementation and same-instance/clocks/free-end/material checks. It receives the new explicitly validated profile inside the scope. The existing health getter and disk validator were exercised together under this scope with CPU objects: new backend/version agreement passes; the same packet outside the scope is rejected. A zero last_error does not become Hold stability.

The source basis exposes external sources and local_sources separately. A new recipe must add local_sources to its project-relative fingerprint and all external sources to its external fingerprint, alongside the new modules, original owner chain, reset/health modules and the independent v2 pair consumer. The old parameter preflight already normalizes its candidate-config path. No recipe is produced here.

The new reset delegates the original full hand/rod Reset and committed eleven-slot rebuild. Direct snapshots bracket that operation, preserve the same actual owner, actual SDK timestamp and native host solver time, and add no SDK or solver steps. The after snapshot must have 11 slots, capacity 5412, pendingSet/moving/pairReady false, every teleport flag zero, and exactly equal current/target/solver pose arrays. Each hand slot matches the original binding's body index/path and actual post-reset get_link_transforms XYZW; table uses the source-verified original table pose. Position comparisons use the exact FP32 input. Quaternion normalization uses an explicit 8×FP32 epsilon component comparison, not a physical clearance/error qualification.

The true constructor state has no registered rod physics_step/timestamp. Both remain None with clock_valid=False, reset_id=rod_tick=0 and actual solver time 0. They are not replaced with SDK0. A completed reset must have a valid fresh identity. The original raw mutation clock_valid=False is retained separately. SDK time remains actual, with the existing bounded ≤240-step callback error envelope; this is not arbitrary-duration PPO reset support.

On failure, raw acquired snapshots and a pure error string remain on readback_receipt. A successfully completed original reset is retained separately in completed_reset_receipt even if the new getter later fails. The optional callback receives owned evidence and that completed receipt, enabling a successor owner to durably retain the real failure prefix. The rod is poisoned, and no later step is allowed. Reusing an already completed transaction rejects without erasing its successful receipt. Borrowed solver/SDK/binding/callback aliases are cleared from exception frames.

The caller still must wire the new reset factory into an isolated episode scope, write durable evidence, and run an independent post-exit audit. The earlier reset receipt's native-current-pose NA is not rewritten; the distinct direct-readback packet is reconciled separately. A measured mesh pose does not establish an initially safe rod/hand placement, external-force readback, physical contact, continuous material station, Slip, or training eligibility.

## CPU verification

Exact command, from this worktree:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_mesh_observation_profile.py tests/test_mesh_reset_order.py tests/test_dual_hand_reset.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_dual_hand_episode.py tests/test_flexible_health_native.py tests/test_r0_numerical_health_source.py tests/test_flexible_parameter_readback.py

Actual result: 180 passed in 34.97 seconds, including 35 new tests. The original reset/runtime implementations are used with explicitly simulated SDK/solver objects; returned mesh-state arrays in these tests are CPU fixtures, not native measurements. Real pinned file/NPZ/table sampler bytes are read on CPU.

Coverage includes exact full source proof and byte tampering, same-profile nested/error restoration and foreign binding rejection, original first and second episode resets, all eleven meshes and actual named pose mapping, strict JSON codec replay, source/clock/instance/pose tampering, current/target/solver divergence, tele/pending/moving/pair capacity/readiness faults, native-shaped error code and getter failures, owned failure-prefix/exception-frame cleanup, repeated transaction evidence preservation, and actual original health capture/validation with the new profile.

Evidence directory: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-observation-profile-cpu-zk405hoi.

| Evidence | Result | SHA256 |
| --- | --- | --- |
| red.log | 30 failed before modules existed | 8855a7ea2bdcd4b7829487f9541a0ea468911afe50589a8898555dbc6af9ef17 |
| repeat-red.log | 1 failed / 1 passed before repeat-evidence fix | fe9f55812f094298de5f7009d033b0a689129c0dcb5f3ac9f43d463af6f18e55 |
| focused.log | 35 passed / 12.84 s | f40d40089c9f9ee471b8f08c2b63bce3dad1d74c350a603ff2bac8c48142a720 |
| regression.log | 180 passed / 34.97 s | 739e57a5a0710e75bf76335cb88eb4fa9255c9104398a8a5cdd7f4b210db4335 |
| protection-summary.json | 528 protected files + 2 Checkpoints passed | ffc85f5b338f9f442eb9bad6465029bfce8eeecc1323d57416b99747df1e60ee |

Intermediate first/second/third diagnostic failures remain in the same directory. They exposed JSON array decoding, exception type expectations, and the real unregistered constructor clock (including a logical env_id=7 CPU fixture); the final run above passed. No failed native attempt or native evidence is claimed in this package.

Independent camera-agent read-only review found no remaining identified blocker at profile SHA 2c1eedb5787055ba5d2bb6615611260cd9aca831e2fe347844164b57fd253388 and readback SHA 53c362e365df241280caa8d89ebd821a8a541d7c62c845d4956a6c12530ca0cc. The reviewer did not run these CPU tests or native. Its initial pendingSet concern was withdrawn after checking add's L160 restoration. A separate peer torque concern was also withdrawn after reading the actual compiled contact.cuh: r already includes -R*n and captured point=P.pos+r.
