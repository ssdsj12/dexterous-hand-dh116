# Dual Pinch one-interval diagnostic entry: CPU integration

Date: 2026-11-13
Base integration: 5c0b71fd410718d76890e0e6f597a59970be607b
Dependencies: contact package 5a6223c (local equivalent b9c17ab), static
capability correction b6cc0d0ec0790a4d47040948ad2982fd3bcda2a6.
The published physical journal is integration 1e32e81.

## Delivered scope

A 142-line composable operation integrates the actual dual builder/reset,
DualHandPhysicalEpisode, existing run_owned_call, published
DualHandEvidenceJournal, hand-table source and original safety projection.
It executes exactly one zero 24D control interval: two SDK setup steps plus
eight measured SDK/rod steps, 12 solver substeps each, no lift, no optimizer.

This package does not create an AppLauncher, resource lock, scheduler worker,
native recipe, registration, approval, readiness change or native execution.
The CLI describes the intended scope and rejects --execute-native before any
output directory or SDK import. A fresh CPU subprocess guarded imports of
pxr/Isaac/omni/ropesim and verified both paths.

The library adapter run_existing_app accepts an already owned app and explicit
caller adapters (make_sim, synchronize, clear_sim, STOP, freeze and manager
readback). The existing lifecycle invokes make_sim; a future native caller must
gate those resource-bearing adapters. The adapter reuses the frozen v2
run_lifecycle through scoped VERSION/OWNED_NAMES/StartupTrace substitutions
and restores all bindings on exit. No giant lifecycle wrapper was copied.

## Actual composition and durable evidence

The operation calls build_reset_owned with budget end 10 and the original
verified prepared source/paired reset. It preserves original 17DOF per hand,
one free rod, the 0.0021 m safety floor and measured 66/371 observation subset.

One existing journal records setup, reset, both geometries, begin(pre, zero
action), eight physical callbacks and end(result). Each physical callback
appends hand_table_contacts to that same original row after the actual SDK
and rod advance. No second journal or parallel physical log is introduced.
Existing encode_owned preserves arrays and unknown NaN state correctly.

inspect_interval first runs the published independent journal replay, then
requires one decision/eight settled rows, final SDK 10, exact zero action and
eight same-boundary static hand-table NA records. It grants neither contact
absence, native qualification, grasp qualification nor Teacher admission.

The operation's capturing record closure and native locals are cleared in
finally. NativeOwners keeps the contact source through composer reset and
CUDA synchronization; the inner operation does not release native views early.
The reused lifecycle clears exception frames before freeze/clear/STOP and
app.close. All builder hand/view/runtime/rope/episode aliases plus simulation
and the contact source are included in the weakref release check.

The CPU test deliberately retains its simulation in the test scope, so
lifetime_released_before_close remains conservative; no native SimulationContext
release is inferred. The future clear_sim adapter must call the already
published source-verified clear_known_simulation_instance helper. This package
does not silently substitute an unverified default cleanup implementation.

## Static table capability gap and corrective ordering

Real v4 evidence reports the GPU filter for static /World/PinchTable unsupported
(see the separate b6cc0d0 report). The original dual Table is also static.
This diagnostic therefore creates no unsupported hand-table contact views and
records available=False, total_contacts=None, hands=None at every boundary.
No dynamic-table conversion or alternative full-report readback is enabled.

The first composed CPU test exposed an unnecessary reporter-schema query before
the already-known static capability rejection. The offline source fixture's USD
registry rejected that API type. The static branch now exits before reporter
inspection and records missing_reporting=None (not inspected). This is a narrow
contact-source ordering correction, not a SDK/registry/physics modification.
The separate active dynamic CPU fixture still exercises the installed getter.

The earlier 5a6223c CPU empty-buffer behavior is not promoted to native contact
absence. Its successor and this integrated entry explicitly retain the static NA.

## Validation

New integration:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_entry.py -x

3 passed in 69.72 seconds. The positive case uses actual original builder/source
geometry/episode/journal with SDK/solver/App doubles. A duplicate solver pair
fails after SDK 3 and one solver call, persists a valid unsettled journal prefix,
performs clear/STOP/frozen update/close and releases fixture hand/view/rope refs.
The positive case settles exactly eight rows and independently replays disk.

Immediate contact and published-journal regressions:

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_bimanual_table_contacts.py tests/test_dh116_dual_hand_evidence_journal.py

30 passed in 7.92 seconds (21 contact + 9 journal). Final total: 33 passed,
0 failed. Compile and 528+2 protected-file verification passed. The earlier
static correction's separate 21-pass run is not counted again.

Evidence:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-entry-cpu-xth6emme

The evidence includes subprocess test output, exact commands, source hashes
and copies of the successful/failed-prefix CPU fixture journal/result files.
Every copied artifact is labeled CPU, not Isaac/native evidence.

## Remaining gate work

Native resource launcher, exact frozen sources/budget, approved worker
reentry, verified all-stop GDB/outer capture and source-verified clear_sim
binding remain the responsibility of a future separately assembled package.
Hand-table absence/contact qualification is blocked by the static GPU source
capability gap; the diagnostic can only preserve NA. Full 371D state and
Teacher admission remain false. No new native run or authorization was made.
