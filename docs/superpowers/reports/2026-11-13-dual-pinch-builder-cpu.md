# Same-rod dual Pinch builder: CPU preparation only

Base: 9e3f24b20076010ce13ecc70ad8d050c2278bab0.
This package adds one builder module and its SDK CPU fixtures. No existing
frozen source, configuration, physics coefficient, original asset, data or model
is changed. It adds no launcher, reservation, native recipe or native budget.
No Isaac/CUDA solver instance or native attempt was launched.

## Exact paired input and unchanged scope

The builder consumes the existing private CPU proposal:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json
SHA256:
996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37.

It checks both that full file and the authored reset seed
647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d.
The exact side-specific roots, full11 q/qd, zero root offset and one proposed
41-node rod are consumed. The rod's nominal axis height is 0.305 m, radius
0.005 m; all nodes remain free and root orientation is not fixed. The old
0.308 m reference nodes are not substituted. Original unknown task material/
rod fields remain unknown; the existing P0-derived candidate coefficients
are carried through without pretending to be measured task parameters.

The loaded extension path and bytes must match the qualified pair-v2 extension:
 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so
SHA256: 92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0.
This is extension identity, not a new bimanual contact calibration.

## Public integration boundary

Module:
src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py.

load_pair_proposal() performs byte/JSON checks only. A real CPU child process
blocks pxr, PhysxSchema, isaaclab, isaacsim and ropesim imports while importing
the module and reading the proposal. build_reset_owned requires a fresh existing
App-owned SimulationContext; its SDK/geometry imports are lazy and occur only
after that context/zero-clock guard. No offline schema loader runs in the
production builder before App launch.

build_reset_owned(sim, owners, project=..., prepared=..., ownership=...,
record=..., owner_resets=..., run_uuid=..., episode_uid=...,
physics_budget_end=...) stores the borrowed live episode at
owners.resources['dual_episode'] and returns an owned scalar/list summary.
Call it within the existing run_owned_call scope. The caller may borrow the
episode there for separately authorized control; it must not return the live
episode out of that owner scope.

run_preparation(...) is the preparation-only adapter around run_owned_call.
It performs one observed SDK reset (0 -> 2 setup ticks) and one dual-hand/rod
reset with no added SDK or solver tick. It never calls episode.control,
SimulationContext.step or RopeBatch.step_free. physics_budget_end is forwarded
to the published episode owner as a caller-provided future interval cap; this
package does not approve, freeze or consume that future control budget.

record receives:
- kind=setup, before_reset/after_reset with the actual SDK clock;
- the original episode kind=reset evidence, including owned CPU arrays/tensors;
- kind=initial_geometry for each side, containing full source17/locked5 data.

Reset contacts are unavailable and can include NaN with validity masks in the
original owned rod packet. A journal must preserve that unavailable meaning,
not convert it to a measured zero contact. The recorder itself must not retain
native handles. Current controller/filter/contact/camera-history reset owners
are explicit caller callbacks; the builder does not invent successful resets.

## Reuse and physical identity

The production chain is the existing ScenePreparation/assemble_prepared_scene
and original spawn_scene(external_wire=True), current material/mimic/mounted-wrist
configuration, counted SDK reset, actual gravity/dt readback, actual 23-link
view mapping, one FreeRodRuntime plus DualHandRodRuntime, and
DualHandPhysicalEpisode.reset.

The original source hand assets and all ten locked SDF identities are retained.
Mesh slots remain left five 0..4, table 5, right five 6..10. Pair capture capacity
is 41*11*12=5412, enabled before any possible measured step. Initial pad poses
come from actual get_link_transforms XYZW buffers; pre-rod bootstrap does not
fabricate a valid rod clock. After reset, bind_actual_geometry and the existing
LiveGeometryReader use the actual synchronized rod/SDK identity.

The original 2.1 mm policy floor is unchanged. Full source geometry and each
locked five-pad set are checked at that same floor with the existing
capture_pinch_initial_geometry helper, and against the same grounded rod with
minimum rod surface gap zero. Stored cooking is explicitly unavailable in this
minimal helper; no cooking request or live PxShape attachment claim is made.
Geometry separation is not proof of contact-free native physics.

The exact paired file's path.initial_all_shape_table_clearance_m values are
2.1108968102464543 mm left and 2.116394609043226 mm right. The CPU SDK-shaped
FP32 pose/joint readback, built from the real source USD/FK, yields
2.110871670052472 mm and 2.1163905693060148 mm. Those observations match the
original static values within 0.0001 mm. Locked-pad minima are separately
3.097800597465228 mm and 4.0828137261403885 mm in that CPU surrogate.
These are separate fields, not native contacts or a reason to lower the floor.

## Initialization and ownership findings

Installed Articulation._create_buffers creates both wrench composers only at
backend initialization (articulation.py lines 1601-1602 in the pinned SDK).
A lazy SDK fixture reproduced premature composer access before sim.reset;
the builder now obtains composers only after the counted setup boundary.

load_sdf_arrays returns dictionaries. A first broad Grid double did not catch
positional dictionary expansion; replacing it with the actual installed pure
NumPy ropesim.sdf.SdfGrid reproduced the failure. The final builder passes
SdfGrid(**fields). This test does not load the CUDA extension.

Both permanent and instantaneous composers from both hands are grouped under
the existing NativeOwners 'composer' cleanup hook. Every reset is attempted,
then existing run_owned_call synchronizes CUDA, clears failed traceback frames,
drops all owned handles and performs its existing collection. No new blind
collection or SDK modification is added. Builder locals and capturing closures
are explicitly cleared in finally on success and failure.

The retained-frame regression deliberately holds build_reset_owned,
run_preparation and initial mapping returned frames. It verifies the CPU hand,
view and rod objects can die after the external fixture releases its own handles.
An earlier broader fixture retained every nested clock callback frame, which
also retained unchanged dependency caller frames through f_back; it is not a
claim that arbitrary SDK-held callback frames are solved. Actual SDK callback,
SimulationContext, STOP and App-close liveness remain future native evidence.
The caller owns STOP, singleton cleanup, App close and outer/GDB validation.

## CPU evidence and boundaries

Evidence:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-builder-cpu-20261113-7v_71_ot

18 new tests passed in 133.33s. They use real original USD/schema authoring,
geometry/FK, locked SDF bytes, actual installed CPU SdfGrid, and the published
real runtime/reset/episode code. The solver and SDK execution are CPU doubles;
their two simulated setup ticks are not actual Isaac steps.

Negative coverage includes changed paired bytes, wrong loaded backend path,
unexpected setup count, bad native link path, solver reset failure, reset-owner
failure, recorder failure, sub-floor clearance, clock drift, and retained
builder frames. No solver step is retried. All four composers are attempted even
if one reset raises. The module also compiles under the actual Python 3.10 runtime.

Related regression: **135 passed, 0 failed**, 174.76s; combined with the
18 new tests, **153 CPU tests passed**. Environment: DH116_NUM_THREADS=2,
CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1 and
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python.

Focused command:

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_builder.py -x

Related command:

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_hand_scene_preparation.py tests/test_dh116_dual_hand_episode.py tests/test_dual_hand_reset.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_deformable_bimanual_step.py tests/test_dh116_flexible_live_geometry.py tests/test_dh116_flexible_physical_exchange.py tests/test_dh116_flexible_rod_boundary.py tests/test_dh116_paired_pinch_reset_preflight.py tests/test_pinch_initial_geometry_v2.py

| Evidence | SHA256 |
|---|---|
| focused-terminal-chunk.json | 48d5c81047cd8bbf5820b0c56a9f14fc18fe7fbce31cc6cff85c08f36e81c838 |
| focused-command.json | d373bc4479fb5cadde29ba411bfc5da6dd509aa390c771e4516e6dd6c14ef36f |
| regression.log | 406466a94eae466b5d8f7a61ce7a65ea4bc60ded76d25ad020eabb85b81564c9 |
| verification.json | 9637094ec50042c7d3780d800618b66446e98aeaf17ee0b2ad566140aa8f07a2 |
| source-verification.json | 466168483977658ab4907b6c692890df0192c2b137ef2806df7486388d12c0c6 |


verify_protection passed 528 protected source files and two historical
Checkpoints. The tracked-file diff against base is empty; all three package
files are new. Registry/readiness/README/CHANGELOG integration and publication
belong to root.

| Source | SHA256 |
|---|---|
| tabletop_dual_pinch_builder.py | 5b190d1896647773c2f8895e73224f387d1dd17a565a44c4e2212db8f477fe17 |
| test_dh116_dual_pinch_builder.py | c0e2c0e6e7fd26ebe4d528ad60f170c3aacc2a0b91c05b5bd98fac6d4d3d7d28 |

This package does not inherit single-left native success as bimanual admission.
Native table support, actual two-hand contact, grasp, complete371 and Teacher
qualification remain false/unproven. A separately frozen native entry, journal,
same-clock contact/force validation, and lifecycle/outer gate are still required.
