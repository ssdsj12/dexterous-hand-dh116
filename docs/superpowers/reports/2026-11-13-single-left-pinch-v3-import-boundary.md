# Single-left Pinch v3 import boundary repair

Base: 3f41d9fc409a0cbe66e3d3679aa3ffdfedd48239. Preparation only:
no Isaac instance, native attempt, CUDA solver, reservation or publication was
started by this package. The consumed v2 files, failed attempt, old recipe,
SDK, original data and physical configuration remain unchanged.

## Actual failure and bounded causal conclusion

The real v2 attempt was
native-attempts/flex.f4-pinch-left-v2-native/35cefcf7a7b328303dfc45794cd310d7ef0ad62de3f8b9ac77ec44e281e838e5/attempt-5bk553__
under /home/xk/.local/share/dh116-autonomous-3c80166/evidence.
Its receipt SHA256 is
75413332580ddb95b4bc60a8ebabaf7309d8bccc3bbf5d60da1e371112f4a582;
native.log SHA256 is
59fd7a9b2239e2f6860a2bea4fe35513a202563d7c02ae58f2cadc5d72c2e61c.
Outer and GDB inferior exited 1, no signal, in 47.48027761897538 seconds.
App close returned. There was no successful SimulationContext construction or
acquisition; initialization/measurement ledgers were not reached. The result
clock/measurement/rod counters are null, not fabricated zero observations.

The first failure is PhysicsContext.set_physics_dt writing
/physicsScene.physxScene:timeStepsPerSecond, with Empty typeName.
Native log lines 32-33 already warn that pxr, Plug, Usd and UsdPhysics were loaded
before SimulationApp. The exact source route is v2 main -> preflight ->
load_geometry -> from pxr import Usd, UsdPhysics; the CPU import guard reproduces
this route without launching Kit. No evidence establishes an early explicit
PhysxSchema import or an SDK/environment upgrade. Both this attempt and the
successful frame-v2 control use scripts/python.sh and the same installed SDK
family. Frame-v2 successful receipt SHA256 is
f60996c5bd42e87825fa56e89d36a0fc97141eb9984c62ac3a9e350012beba86
(attempt-nmh6rqlf, outer 0).

Premature USD/Plug loading is a confirmed startup-order defect and a candidate
explanation for the schema error. Whether this repair resolves the actual
Empty typeName failure still needs the newly approved native observation.
The secondary error was snapshot_weakrefs rejecting zero registered owners at
both close checkpoints; it added unrelated diagnostic stacks.

## Narrow successor

The new entry keeps the old preflight byte/provenance checks and reconstructs
exactly the old geometry source SHA map using file bytes. CPU regression compares
it to real load_geometry output, including the original USD, reference, candidate,
geometry code and all five locked pad capture files. It no longer opens USD before
AppLauncher. Full source17/locked5 actual-link geometry remains in the unchanged
v2 builder after App launch and before measured action; full geometry is also
recomputed by the unchanged independent post-exit validator.

The entry rejects preloaded pxr/PhysxSchema before output creation. The startup
adapter repeats that check immediately before the old execute path imports
isaaclab.app, and records before_app_launcher.json. StartupTrace emits an empty
resource map with observation=no_registered_owners if construction never acquired
an owner. It preserves the primary failure, runs actual close, leaves clocks
unavailable and release false; it does not synthesize dead owners or weaken
nonempty weakref bounds. Bindings restore VERSION/trace/snapshot even on exception.

The dedicated v3 wrapper re-enters itself in the real child worker and freezes
its entry, source, marker, output, all-stop GDB and independent audit identities.
The post-exit v3 audit first runs every old outer/GDB/clock/geometry/quiet/action/
pair/ownership check, then requires its clean import artifact. Old v2 result
identity, missing boundary, preloaded USD, empty owners, bad outer, live owners or
clock drift remain failures. A generic receipt cannot bypass the dedicated audit.

The original acquisition/measurement functions, 17 sensor sources, immutable
physical parameters, corrected pose, free rod, source mesh identity, explicit
ownership cleanup, STOP/frozen update/close sequence and GDB implementation are
reused directly; no new native setup copy or AST-generated runtime was introduced.
Only the versioned entry copies the previous small byte-preflight/CLI flow.

## CPU evidence

Evidence:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v3-import-cpu-20261113-h2b8wdgi.

The first test-first run reproduced the old early USD import and failed the two
new v3 tests because their modules did not yet exist. Intermediate 14 boundary/
lifecycle/transport tests and 9 independent full-disk gate tests passed. The first
full regression had 189 passed and one fixture error: cleanup_case was called
outside the wrapper binding context. That test now uses the real binding context
and additionally exercises native_recipe. A preparation-only recipe-generation
error (USD asset accidentally added to source_files) was rejected by release_path;
the final recipe keeps immutable assets in the existing inputs instead.

Final regression: **190 passed (24 new + 166 existing), 0 failed**, pytest 136.69s.

Exact command and environment: DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python,
DH116_NUM_THREADS=2, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1;

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_pinch_v3.py tests/test_dh116_pinch_v2.py tests/test_pinch_initial_geometry_v2.py tests/test_dh116_physx_lifetime_frame_v2.py tests/test_dh116_physx_lifetime_probe.py tests/test_dh116_physx_lifetime_scheduler.py tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_debugger_stopped.py tests/test_dh116_pinch_scheduler.py tests/test_deformable_grasp_runtime.py tests/test_dh116_contact_material.py

| CPU evidence file | SHA256 |
|---|---|
| pytest.log | 37c71f5442c06499a4b90180a2d93112ad556ff3f6097265b178eaf3358d9cb0 |
| verification.json | 6b708dea473b73ef4f288f45276b2b07e9f9fccd7460f83b9b77db9c3bd43b15 |
| fingerprints.json | 694d37053f21430bdd7585cf7286bb9c8832138b543a5226ec06b36384cc9758 |
| protection.json | c0eabcf07563281e0ebb1335a87b697507fc46cceee6fa6941426317b8c40146 |
| source_sha256.json | 940cd3c560581f257c75b08e32921235e2ce50ed726574df6d18ab13c94aea3d |
| first-regression-terminal-chunk.json | a68653c490196c6d927a2642fe6efa4533ae9c636bfd7108ba61e319e69bfe32 |


The real-process tests execute the actual v3 CLI, full byte preflight, worker
identity gate and unchanged execute function up to an injected AppLauncher
constructor sentinel. Only lock, ropesim module and launcher are CPU substitutes;
no SDK app/solver is constructed. Separate real parent/child scheduler fixtures
test v3 reentry and reject a failing post-exit audit. Their logs/receipts are CPU
fixtures and do not assert native success. Source equality is also checked against
a separate full CPU USD geometry load.

Independent read-only peer review of the four new runtime files found no blocking
identity, binding-restoration or qualification issue. It did not run native or
claim additional tests. verify_protection passed all 528 protected source files
and two historical Checkpoints. Git tracked-file diff against base is empty;
all package files are new. No old C0-C5, Pinch v1/v2, SDK or data was edited.

## Frozen successor and limits

Suggested native scope: single_left_free_rod_lifecycle_readback_v3_only.
It must depend on the v3 CPU package plus the existing passed frame-v2 lifetime
and pair-v2 calibration scopes, not on the failed Pinch v2 receipt.

| Item | Value |
|---|---|
| entry | scripts/verify_dh116_pinch_candidate_v3.py |
| dedicated scheduler | scripts/run_dh116_pinch_work_packages_v3.py |
| recipe | configs/dh116_execution/deformx_pinch_left_candidate_v3.json |
| command_spec_sha256 | a7e2bedc60969037198db8fe2b5f40214de516f9734d1a5dda2f81390599788a |
| code_config_sha256 | 113882bbab640846e6b3d3d5042bdfa6487ddbffbd497869d71acd8e11eb8362 |
| inner candidate SHA | b622afecfd2586a8f95fc43a97bbf1747242b7ac0acd91f8e93b1927e835fe45 |
| budget | GPU3, env1, at most 180s, 240 total SDK = 2 setup + 238 measured, 0 PPO |
| attempts | one attempt for the new exact code/config; no automatic retry |
| tests for registry | tests/test_dh116_pinch_v3.py, tests/test_dh116_pinch_v2.py, tests/test_pinch_initial_geometry_v2.py |

Recipe freezes 74 source files and 66 external inputs; the scheduler additionally
includes its own source and scripts/python.sh. The qualified pair extension and
receipt are the same exact v2 prerequisite. All-stop GDB source remains
2ee4da78216db91ac4d0c3f417285eaeb989f0a608dfcd0c916eba3255affc35.

Passing lifecycle/readback still does not require effective hand contact. Actual
hand-pair rows, required-digit load and total/table contact counts must be reported
from raw ledgers separately, as in the v2 report. Source/cooked attachment,
successful grasp, bimanual same-rod admission, full371 and Teacher admission
remain unproven. This package does not restart the deferred dual-hand builder.
Root owns registry integration, publication and native scheduling.
