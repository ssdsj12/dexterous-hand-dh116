# Single-left Pinch v2: corrected reset, measured quiet window, pair capture and release

Prepared on 2026-11-13 from integration fb204d11c38f667a8a59fca3f40d3e346f58c8b2.
This package was prepared and tested on CPU only. It launches no Isaac instance, CUDA solver,
native Pinch attempt, optimizer or PPO update. Existing Pinch v1, lifetime v1/v2, pair v1/v2,
SDK sources and physics coefficients are unchanged.

Dependency: the independent single-hand geometry helper from source commit
71095aca7a4e2848f39b5f58cba1ed94344110f8 (locally cherry-picked as c8bd226).
Root integration must include that package before this package.
Its module SHA256 is 3c272069c43eaed71f11c17d73af28f810f39762cea792771a83647c04924bd9.

## Concrete scope

The new entry is scripts/verify_dh116_pinch_candidate_v2.py; use the dedicated
scripts/run_dh116_pinch_work_packages_v2.py wrapper and
configs/dh116_execution/deformx_pinch_left_candidate_v2.json recipe.

One original left DH116 hand remains at a fixed root. The initial root and six active joint
values come directly from configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json, with its
source identities checked. The original free 41-node rod, five separate original pad SDF slots
0–4, separate finite table slot 5, gravity, material/actuator/mimic parameters, action kernel,
velocity limits and delayed mesh-origin-to-body-COM wrench routing are retained.

The builder is a readable versioned copy of the successful frame-v2 builder. The additions are
the corrected pose, original PhysX table, no-step actual kinematics refresh, source geometry
capture, pair instrumentation and a hand-table contact view. CPU AST checks confirm the original
Articulation, FreeRodRuntime, material and mimic calls remain identical. No pinned rod node,
extra constraint, attachment, added solver step, source SDK patch or runtime AST rewrite exists.

After the existing two SDK initialization steps, explicit joint reset is followed by
update_articulations_kinematic and asset-cache refresh. Actual SDK time/step must remain unchanged.
At that clock, the geometry helper copies actual body names/paths/transforms and checks all
17 original source collider identities and five locked historical F4 pad hulls. Its independently
replayed numerical geometry gate requires nonnegative hand-table and hand-rod gaps.
The caller-supplied initial rod geometry is explicitly distinct from a solver measurement.
Existing cooking getters may be queried; unavailable records stay NA. No cooking request is made,
and stored cooking representations do not prove live PxShape attachment.

The original 238 measured steps remain: 48 zero-action steps, 160 closing-action steps, then
30 zero-action steps. Before the first closing command, all 48 quiet rows must contain no
reported hand-table contact points and no DeformX hand-pad contact samples. Rod-table settling
contacts remain allowed. Any hand-table contact at a later row also terminates the candidate.
Failures preserve available rows and go through release/close; they do not become qualifying
receipts or trigger a retry.

Hand-table evidence uses installed RigidContactView.get_contact_data: counts, indices, contact
points, normals, separations and normal forces. It does not infer contact absence from zero net
force. Exact original body paths, 17 sensors, one table filter, finite active data and nonsaturated
buffers are required. Only PhysxContactReportAPI with threshold zero is authored. The IsaacLab
activate_contact_sensors helper is deliberately not invoked because it also changes sleep
thresholds; authored sleep values are read before/after and must remain unchanged.

Each measured row stores the atomic six-mesh DeformX pair packet. Its explicit array shape
metadata preserves empty packets through JSON. Independent replay reconstructs the raw packet,
reconciles per-node and per-mesh totals and binds it to the same episode/reset/SDK/rod clock.
The parsed normals/forces remain measured pair evidence; the old net-wrench metrics do not
become a grasp or full-state admission claim.

## Actual pair prerequisite

This preparation reopens the already successful pair-v2 calibration:
evidence/native-attempts/flex.f7-deformx-pair-capture-v2-native/
94d855ce6be8700816a85a8e84d41a87fe74b52454bf4f8591661e932c6d08c6/attempt-nobp3d5x/receipt.json
under /home/xk/.local/share/dh116-autonomous-3c80166.

Receipt SHA256: 7af3e7d7358ea58c2d4be656c889d162835830ccbe1db20f36c457a5cfe51ba2.
Historical result: outer exit 0, observed shared lock, 2.803089093999006 s, three verified pairs,
one CUDA step, twelve substeps, solver last_error=0. Its raw artifacts and independent outer/log
identities are revalidated. This package did not rerun that experiment.

The exact loaded extension remains:
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so`
with SHA256 92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0.
The native entry checks the actual loaded module path/hash. The recipe also freezes its binding
files, libcudart, the prerequisite receipt/artifacts, SDK tensor code/libraries, schema/cooking
bindings and GDB. Original backend hashes remain provenance; they are not relabeled as the
newly loaded pair extension.

## Independent lifetime and receipt gates

Both build_native and measure_owned explicitly clear native aliases and capturing closures in
finally, including success when a completed frame is retained. Exceptions become text after
completed frames are cleared before STOP/close. The existing owned-call synchronization,
reference release, SimulationContext clear, frozen Kit clock, STOP, one frozen app update and
actual App close ordering is retained.

The new contact view joins the hand/root-simulation/view/composer/provider/runtime/rope weakref
set. Full process maps and scalar referrer/liveness evidence are recorded around the same
boundaries. Before App close all owned resources must be dead; unknown/live resources fail.
A detachment marker is only evidence that Python assignments ran, not proof of native release.

The wrapper reenters itself in the real scheduler child, scopes/restores its worker marker and
uses the already verified all-stop debugger source:
scripts/dh116_physx_lifetime_debugger_stopped.py
SHA256 2ee4da78216db91ac4d0c3f417285eaeb989f0a608dfcd0c916eba3255affc35.
Normal finalize and fatal paths retain stacks, thread stacks, registers, instructions, libraries,
full /proc maps and GDB mappings. Outer exit, inferior status, stopped-thread count, actual stack
frames and mapped program counter are checked independently; marker-only captures cannot pass.

The post-exit audit reopens geometry, pair/contact rows, action/SDK targets, COM reaction, actual
clocks, initial free-rod acquisition, frame detachment, weakrefs, STOP, App close and GDB artifacts.
A generic passing result is revoked before this audit and restored only when it passes.
Previously qualified prerequisite receipts are handled by their original validators.
The old frozen Pinch/lifetime/pair entries and shared scheduler were not edited.

## CPU validation

Final Ubuntu command used DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python,
DH116_NUM_THREADS=2, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1 and
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

```sh
bash scripts/python.sh -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_dh116_pinch_v2.py tests/test_pinch_initial_geometry_v2.py \
  tests/test_deformable_grasp_runtime.py tests/test_dh116_pinch_scheduler.py \
  tests/test_dh116_physx_lifetime_frame_v2.py tests/test_dh116_physx_lifetime_probe.py \
  tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_scheduler.py \
  tests/test_dh116_physx_lifetime_debugger_stopped.py tests/test_dh116_deformx_pair_capture.py
```

Result: 169 passed in 113.40 s, exit 0: 42 new Pinch tests, 24 geometry-helper tests,
103 existing regressions. The CPU GDB fixture is a small ordinary process, not an Isaac run.
The actual 238-iteration Python measurement loop was exercised with CPU SDK/solver doubles.
Quiet-contact and solver exceptions stop it early; retained frames and real subprocess worker
reentry are tested. Synthetic packets/receipts remain CPU fixtures, not native measurements.

Peer review found a JSONL newline-escaping defect. A complete independent disk success fixture
first reproduced its rejection, then passed after correction; damaged clock/contact/liveness/
cleanup/geometry/GDB/action/target/reaction fixtures fail. Peer review found no further blocking
issue in the reviewed ownership, quiet-source or wrapper paths.

Evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v2-preparation-cpu-20261113-0jzvmfv7`.

| Evidence | SHA256 |
|---|---|
| pytest.log | b5721777812524f5bbdb9a25ca3c456f6423d1431533ea338842407d14c67c19 |
| verification.json | 27001a7288b39b9216b995020357fec2b028fda919c47595dac4fcf5a3f26dbb |
| fingerprints.json | 4dee222f8692c77e5851fb2e1f5798c5d6ab0dbc3295bd094ce88bf596250a2e |

528 protected source files and two historical Checkpoints passed verify_protection.

## Exact recipe and limits

| Identity | SHA256 |
|---|---|
| command_spec_sha256 | 6d8e139b171ebd4e363fa34ccb1e2991c734cb3be800365960727ecf281dc755 |
| code_config_sha256 | 35cefcf7a7b328303dfc45794cd310d7ef0ad62de3f8b9ac77ec44e281e838e5 |
| inner candidate code_config_sha256 | 9fa9b54dbb58eaa2132215b3c3bf29e266cc80bb213a35a56609cf4656a8fccc |
| preserved Pinch v1 code_config_sha256 | d0ec5e2f6146c82e637331acd8dac173fd46cc80425e769578d36e7ce98273de |
| preserved frame-v2 code_config_sha256 | 5c5654de8d08a9a96a989d0a811f4411db8ded8c99b4ac40e5f7ee04c4e599c6 |
| preserved pair-v2 code_config_sha256 | 94d855ce6be8700816a85a8e84d41a87fe74b52454bf4f8591661e932c6d08c6 |

GPU 3; one environment; exactly 240 total SDK steps (2 setup + 238 measured);
238 rod steps of 12 substeps; zero PPO; one attempt per frozen code/config; 180 s qualification
limit; at least 4096 MiB free GPU memory and 1 GiB free disk. The existing scheduler records
overdue work and rejects qualification after 180 s; it does not hard-kill at that threshold.
SDK internal reset/render work is not claimed to be absent. No camera/render product is created.

Suggested independent registration: flex.f4-pinch-left-v2-native with evidence scope
single_left_free_rod_lifecycle_readback_v2_only, requiring the geometry CPU
package, passed flex.f4-physx-lifetime-frame-v2-native and passed
flex.f7-deformx-pair-capture-v2-native with their exact native scopes.
Root handles registration, integration and execution under the user's existing bounded
authorization. This source package performs no native launch or retry.

Actual reset tracking, combined Isaac/pair-extension behavior, new contact-view liveness and
measured contacts remain pending this specific native attempt. Only five hand pads are DeformX
rod colliders; twelve other hand links have initial source geometry checks, not new rod-contact
solvers. Quiet evidence covers the configured five-pad rod contacts and seventeen-body table
reporters. It is not a proof of all possible unmodeled contacts. Stored cooking data remains
distinct from live attachment. Grasp, 371D policy binding and Teacher admission stay false.

A complete run can pass lifecycle/readback with zero hand-rod contacts; it must never be used as
a qualified hand-contact dependency. After exit, pairs.jsonl gives directly reproducible counts:
total contact samples = sum(len(record["pairs"])) across all 238 rows (each item is one
substep/node/mesh sample); actual hand pair rows = rows containing any mesh_index < 5;
loaded required-digit rows = rows in which both finger14_link and finger23_link have samples
with normal_force_on_rod_n > 1e-5. Report these separately from table samples, lifecycle
qualification and the existing net-wrench opposition row count. No nonzero count is asserted by
this preparation. Zero touches do not qualify a dual-hand contact experiment.
