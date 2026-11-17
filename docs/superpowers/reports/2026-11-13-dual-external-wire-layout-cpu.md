# Dual-hand external-wire layout compatibility: CPU preparation

Date: 2026-11-13. Base: 2044ba909566bc612619aa4bbf96ad478c9d4b2f.
This package changes no consumed v1 file, physical parameter, solver, rod seed,
native budget, registry, or qualification. No Isaac process, GPU operation,
native attempt, or training was started.

## Observed failure and narrow cause

The consumed native attempt is:
evidence/native-attempts/flex.f7-dual-pinch-zero-interval-native/50a33500aaad8a825a78a9c6891070149ba38560a23c6a05b74ce2bb72309358/attempt-d3ypguyp
under /home/xk/.local/share/dh116-autonomous-3c80166/.
Its receipt records exit code 1 and duration 55.352164524025284 s.
The unchanged native.log contains KeyError: 'wire_center'; SHA256:
b64b30cfcfcb415706ee1110baab1af4ec8a84c880e8f2d4d06d78c216801bdf.

Original spawn_scene first translates wire_center and table_center by the
environment origin. This happens before any object is created, even with
external_wire=True. Original scene preparation supplies authored wire_points,
table_center, table_size, hand_roots and active_joints, but not wire_center.
The external-wire branch returns with wire=None and sensors=[] before the later
RigidObject path can consume wire_center. Therefore this missing key is legacy
layout bookkeeping, not a request to create another physical wire.

## Independent successor adapter

tabletop_dual_hand_scene_preparation_v2.external_wire_layout() is an explicit,
process-local context manager. After App initialization, enter it around the
existing builder operation. It temporarily binds the original scene module's
spawn_parameters, calls its original implementation, validates finite authored
(N>=2,3) wire_points, then returns a shallow layout copy with their finite
arithmetic mean as wire_center. An existing inconsistent wire_center is rejected.
Args, source identities, table dimensions, roots, joints, materials and
solver configuration remain those returned by the original function.
The predecessor binding is restored in finally, including exceptional exits.

All project/NumPy imports inside this adapter are late. Importing its module does
not import pxr, Isaac, omni or ropesim. This small adapter owns no native handles.
The existing builder/lifecycle remains responsible for native ownership cleanup.
Do not use the authored point mean as the paired solver's actual construction
seed: the paired free rod remains the original explicitly supplied seed.

## CPU evidence

TDD first reproduced the concrete original KeyError before any fake SDK
construction (one passing regression), then failed because the new module did
not exist. After implementation the seven new tests passed in 7.31 s.
The tests compile and execute the unmodified AST of original spawn_scene;
they do not replace that function or generate runtime implementation code.
Only SDK constructors/configuration endpoints are CPU doubles. Actual source
USD references, original mount/mimic/material authoring and the complete
external-wire branch run against a real in-memory USD stage.
RigidObject and ContactSensor constructors are forbidden; neither is reached.

The SDK-shaped articulation fixture updates existing referenced transform attrs
instead of trying to add duplicate USD xform ops. It checks both original
assets, two hands, one table, two materials, activated reporting configuration,
64/16 iterations, 0.15 Nm effort, original joint values and all four actuator
groups. It also checks binding restoration, malformed/nonfinite point rejection
and a real subprocess import guard. These are CPU authoring checks, not evidence
of native contact reporting capability or physical behavior.

Combined regression command (DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python,
DH116_NUM_THREADS=2, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1):

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_external_wire_layout.py tests/test_dh116_dual_hand_scene_preparation.py

Combined result: 26 passed in 100.68 s (7 new + 19 existing), exit 0.
Evidence directory:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-external-layout-cpu-h7he34lm.

Repository.verify_protection() passes for the existing protected set (528+2).
Byte comparison against the base confirms original scene preparation, original
spawn source, v1 recipe, original preparation entry and native worker unchanged.

## Frozen new files and limitations

Adapter SHA256:
ae24abf3fe1dd67f15f89086c921df1e07e0d90f3f301fb6f9b4da94505cb0e9
Test SHA256:
1729034e9f0755d3483f34e6ce4fe6218a3c9b6e4adf79f81bac8dda180dbc3a

The external branch's required layout accesses are hand_roots, wire_center,
table_center, table_size and active_joints. Its direct required args are
position_iterations, control_mode and mimic_frequency; existing optional
velocity_iterations, hand_material, load_parameters, passive velocity limit,
palm_transport and free_wrist_grasp are already supplied by spawn_parameters.
The full original-function CPU execution finds no additional missing field on
this exact configured external-wire branch. It cannot establish native ABI,
actual resource initialization, reset success, geometry readback or execution.

No successor recipe or authorization is created here. Root integration must
freeze the adapter plus unchanged original entry/worker inputs for any separately
scheduled successor. The consumed v1 failure remains failed and immutable.
Same-rod contact, grasp and Teacher qualification remain false; static GPU
hand-table availability remains NA. No change to the 2.1 mm guard is made.
