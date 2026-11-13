# Dual-hand free-rod runtime composition — CPU preparation

Base commit: 210d865fbe55e1d838b80d8913e154f4fb482439.
Worktree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/dual-hand-rod-runtime-20261113.
No native launch, Isaac instance, CUDA solver experiment, training, scene activation or execution
recipe is introduced or run. This package adds one isolated library module, its tests and this report.

## Existing paths and chosen scope

The existing FlexibleObject and WireRuntime were read first. They use BatchedCosserat /
TorchCudaCosseratRod, with their own material and impulse-based coupling paths. Reusing their scene
activation would restore a different solver path. Refactoring the current FreeRodRuntime would
change a frozen native source fingerprint. The chosen composition therefore consumes one exclusively
owned, fresh FreeRodRuntime, leaving all old source files unchanged.

The existing base constructor still owns RopeBatch creation, gravity, free node masses, rod material,
damping, five original left-pad SDF slots and the finite table SDF. The new layer adds only the right
hand's five SDFs using the same effective hand/rod friction calculation. The actual slot order is
left 0–4, table 5, right 6–10. It does not reorder solver data or mutate base.names.

Bindings include side, original distal link name, original prim path, actual body index and supplied
source USD / cooked geometry / SDF SHA256. Guards reject missing pads, duplicate body indices within
a hand, crossed side/name/path identities, shared hand roots and invalid hashes. Hash syntax and
mapping checks do not verify the supplied bytes against a live SDK scene: binding_provenance remains
caller_supplied_unverified and source_qualification=false. Source geometry verification remains an
independent prerequisite for any future native integration.

An instance is single environment / single free rod. Attempting a second composition around the same
base is rejected. A failed native append consumes that owned scope, so its caller must dispose it
instead of appending again. There are no new pinning, root drives, attachments or table force routes
to either hand.

## Data, wrench and clock boundary

DualHandRodRuntime.reset reuses the original base reset and host state capture, teleports the right
five meshes too, and requires a fresh episode UID with the next reset ID. It preserves the solver's
actual global time, sets rod_tick to zero and invalidates all previous measured contact. Old episode
UIDs cannot be reused.

step accepts the caller's next actual SDK physics_step, ten world link poses (xyzw quaternions) and
ten world COM positions. It changes ten mesh poses and calls the same RopeBatch.step_free exactly
once for 1/480 second and the existing 12 solver substeps. It does not step Isaac. The original
FreeRodRuntime.capture and read_rod_snapshot implement state copying and the existing node schema.

Each genuine mesh wrench is force BY rod ON mesh in world axes, with torque about that mesh's link
origin. Routing reuses scripts/verify_dh116_deformx_rigid_bridge.py:wrench_at_com to shift the torque
to the corresponding body's COM. Left/right forces and torques retain explicit distinct side/body
maps even when both hands use identical body index numbers. The table wrench stays separate, with
world-axis torque still about the table mesh origin.

write_composer_reactions validates and converts both hand packets before any write, then performs
one exclusive reset + set_forces_and_torques call per hand with is_global=True and the five original
body IDs. It does not advance physics. Its caller supplies the tensor conversion/device and owns
exclusive composer access; a native composer failure can still produce a partial write and must be
handled by the caller's existing owned teardown. This is not a new generic force accumulation layer.

The actual aggregate node forces and eleven mesh wrenches must balance. If the solver exposes the
genuine contact_pair_samples API, one atomic capture must match this step's actual aggregate arrays,
mesh count and substep count; the existing read_pair_capture validates pair uniqueness, bounds,
normal direction, finite values, overflow and reconciliation. No pair is manufactured from a net
wrench. Missing capture remains None/unavailable; require_pairs=True rejects a missing API before
the solver advances. Even validated solver pair samples retain contact_pair_qualification=false.

clock returns run_uuid, env_id, rod_uid, episode_uid, reset_id, physics_step, rod_tick, timestamp_s,
solver_time_s and clock_valid. physics_step is the caller-supplied SDK boundary; this module does
not independently query the SDK. policy_step/substep belong to the external action transaction and
are not invented here. Per-step and cumulative solver time must agree with 480 Hz.

Any failed mutation or readback requires fresh reset. If set_mesh_poses fails after the SDK has
already advanced, the supplied SDK step is retained while the rod tick stays unchanged. If the
solver call throws with unknown partial progress, rod_tick becomes None; actual solver time remains
readable and clock_valid=false. No rollback or completed step is fabricated. If a completed solver
step subsequently fails pair/force validation, its actual SDK/rod counters remain recorded but the
invalid clock cannot be admitted by the action adapter.

The peer implementing BimanualStepConnection confirmed the bridge:
begin → submit existing 24D targets → caller SDK step → this rod step → hand.update → finish.
That adapter requires clock_valid=True and checks SDK +1, rod_tick +12, solver time +1/480.
Full state371, source, native, contact-pair, grasp and Teacher qualification remain false in this
module's measured records and provider packets.

## CPU verification

A real unchanged FreeRodRuntime is instantiated around a CPU fake RopeBatch API. Thus tests exercise
the actual base mass/material/reset/capture/provider code while starting no native solver. The fake
supports distinct left/right/table slots and real copied arrays; its synthetic contact records are
test data, never physical evidence.

Test-first evidence: 20 expected missing-module failures preceded implementation. Four failing
clock/fault tests then caught partial solver progress and absent clock validity. One failing provider
test added explicit unqualified packet flags. Peer review identified SDK step accounting on a mesh
pose write exception; a dedicated failing test preceded the one-line ordering correction.

Final command:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dual_hand_rod_runtime.py tests/test_deformable_grasp_runtime.py tests/test_deformx_wire_runtime.py tests/test_deformx_batched_adapter.py tests/test_dh116_deformx_pair_capture.py

66 passed, 1 skipped, 0 failed in 3.74 seconds: 29 new and 37 prior tests.
The unchanged real-CUDA WireRuntime contact test is skipped because CUDA_VISIBLE_DEVICES is empty.
Baseline before new code: 37 passed, 1 skipped. An initial baseline command used the wrong pair-test
filename and collected no tests; the corrected command above established the actual baseline.

Private CPU log:
 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-hand-rod-runtime-cpu-20261113/pytest.log
SHA256: b3f7592740f7fec0ce4bd4d60a1ad098df5053dc263fce30cc8e26fea3892428.

Independent peer review checked slot/world-COM accounting and the shared clock boundary. The
subsequent pose-write exception regression closes its specific finding. CPU success does not grant
native activation; the recorded PhysX shutdown problem, genuine pair-source qualification, scene
identity verification and full Teacher state/task admission remain separate blockers.

Protection verification passed for 528 original source files and two historical Checkpoints.
All previously tracked files and frozen native fingerprints remain unchanged.
