# Bimanual original hand-table contact source: CPU preparation

Date: 2026-11-13
Base: c8f7ecebd6034a270c952fa66c54ff614ffbb97a
Branch: codex/dual-hand-table-contact-c8f7ece

## Scope and installed-source findings

This is an independent, opt-in source reader, not a native launcher or gate
qualification. No existing builder, frozen source, physics parameter, journal,
registry, recipe or authorization was changed. No Isaac App, GPU solver,
native attempt, lock, training or PPO was run.

The original dual builder delegates to tabletop_vec_isaac.spawn_scene, whose
hand UsdFileCfg already sets activate_contact_sensors=True for both hands
(line 88). external_wire=True removes the old Wire ContactSensor objects but
does not disable this USD contact-report authoring. Therefore no reporting
mutation wrapper is needed. The new module checks the actual authored APIs
after reset; a configured boolean alone cannot produce available data.

Installed schemas.activate_contact_sensors adds PhysxContactReportAPI and a
threshold defaulting to zero. It also contains an existing sleep-threshold
write. This module neither calls that function nor changes/reapplies sleep,
material, geometry, actuator or physics settings. Both the existing spawn
source and installed schema source are byte-pinned and recorded.

Installed omni.physics.tensors.impl.api.RigidContactView.get_contact_data(dt)
returns six buffers: force, point, normal, separation, count and start. Its
docstring says five in one sentence, but implementation and shape descriptions
contain six. The backend receives dt to convert impulses to forces. We use
1/480 seconds and safely reuse the frozen single-left six-buffer parser.
Installed sensor_paths and filter_paths are read from the actual backend,
rather than inferred from configured filter expressions.

Pinned installed API:
- /mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py
- SHA256 bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b

The module also checks the getter's actual Python code filename against this
installed API. These source checks are not native execution attestation.

## Composition contract

After the existing App-owned builder has validated prepared geometry and
completed its counted setup/reset, call:

    source = owners.keep("bimanual_table_contacts",
        open_bimanual_table_contacts(sim.stage, scene["hands"], prepared,
                                     read_clock=rod.clock))

The caller supplies the existing verified prepared source object and actual
hands; this reader does not independently reload or qualify USD geometry.
It checks the complete original 17DOF names and each hand's 23 link-path
mapping, then selects exactly the 17 source collider bodies per side.

Both hands must share the same existing root SimulationView and retain
distinct articulation views. The finite table must exist as an active enabled
collider at prepared["prefix"] + "/Table". All selected bodies must be active,
enabled rigid bodies with contact-report threshold zero. Missing reporters
produce an unavailable source without constructing either contact view.

The reader constructs two independent RigidContactViews, each with 17 exact
body paths, one exact table filter per body and capacity 4096. It checks actual
sensor membership and resolved filter paths; sensor reordering is handled by
the actual getter order. Supported filter layouts are a single-table broadcast
or 17 explicit single-table entries. Unknown layouts and wrong partners fail.

After a separately owned real SDK/rod step, call:

    packet = read_bimanual_table_contacts(source, read_clock=rod.clock)

Clock capture brackets both hands and checks actual 480 Hz timestamp plus
run/env/episode/reset/rod identity. A completed SDK step newer than creation
or the previous successful read is required. Report/body/table availability
is checked again. The output contains owned JSON values only, exact body and
table paths for each recorded contact, copied count/start metadata and dt.

The source intentionally owns its two views and stage. Keep it inside the
existing NativeOwners scope. source.release() clears those references;
existing run_owned_call remains responsible for exception-frame cleanup,
synchronization and full teardown before STOP/App close. No hidden callback,
global view cache, destructor stepping or new gc.collect is introduced.

## Failure and qualification boundaries

Absent source, absent reporting, released views or inactive/disabled source
prims yield available=False and total_contacts=None. They never become zero.
Malformed identities, changed clock, stale frame, reset mismatch, malformed
buffers, nonfinite values, overlapping ranges and capacity saturation raise.
The inherited parser conservatively refuses ranges ending at capacity as well
as full capacity. This detects suspicious local buffers; the installed getter
does not expose a separate upstream-overflow flag, so that guarantee remains
false. There is no contact-cancellation shortcut using net force.

Raw source normals/forces are preserved with their existing field labels.
No new calibrated force sign/frame, actor/collider numeric-ID, grasp, native
or Teacher qualification is issued. Rod-table pairs are never read or used
as hand-table evidence. A valid zero count is only a measured empty packet
from these explicit fresh SDK-shaped buffers, not geometry-based contact proof.
The original 0.0021 m guard, 17DOF per hand and incomplete 66/371 state contract
remain unchanged.

## CPU validation and limits

The new tests execute the installed RigidContactView class AST against a CPU
frontend/backend, exercising its actual allocation, six-buffer return and dt
forwarding without importing Kit. Explicit synthetic USD prims provide report
API/type/activity checks; these are not claimed to be native original hands.
The already published c8f7ece builder/source-geometry interval regression is
not rerun or counted as new native evidence here.

A real CPU subprocess rejects pxr/Isaac/omni/ropesim imports while importing
the module, proving the entry remains safe before AppLauncher. Other cases
cover actual sensor reordering, table partner identity, two filter layouts,
NA behavior, fresh-zero behavior, saturation/overlap/NaN, frame/reset mismatch,
report removal, deactivated body/table and explicit view release.

Command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_bimanual_table_contacts.py tests/test_dh116_pinch_v2.py

Result: 62 passed in 29.14 seconds: 20 new source-reader tests and 42 existing
single-left v2 regressions. Initial missing-module failure and the two inactive
prim negative failures were observed before their implementation. Final 0 failed.
Python compile and Repository.verify_protection (528+2 entries) passed.

Evidence:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/bimanual-table-contact-cpu-f0linnte

Source SHA256: adcb88c15ce10f53cbc58e6940ba9d4eb00e31c1f949f057be0cb5c8494e185b

Native double-hand availability, actual resolved filter layout, count coverage,
force semantics and outer-exit/lifetime remain unexecuted. A future native
package must pin this module and its listed dependencies, explicitly keep the
source in its owned scope and consume NA as unavailable. This package creates
no recipe or native permission.
