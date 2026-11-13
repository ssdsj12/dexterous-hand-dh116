# Opt-in post-step source-clearance guard: CPU preparation

Date: 2026-11-13. Base 43abe38b555fe6063cbdf2fac3ad87907c4a5686.
Only a new module, tests and this report are added. No frozen exchange, original
controller, zero-action native entry, recipe, material or geometry is changed.
No Isaac/native attempt, GPU solver, training or publishing was performed.

## Concrete gap and scoped correction

The frozen FlexiblePhysicalExchange reads current source-envelope clearance
before wrist projection and again after SDK advancement. The latter is used
for pad poses/COM but not recorded in the returned physical row or included in
the existing terminal guard. A projection floor value is not a measurement of
post-step clearance. This package prepares a later opt-in diagnostic and does
not retroactively change any earlier native receipt.

New API:
- ClearanceGuardedExchange, a subclass of the existing FlexiblePhysicalExchange.
- validate_post_step_clearance(exchange_row), for an owned decoded physical row.

At future episode construction, a separately reviewed caller may explicitly
select the subclass (for example a scoped binding of the episode module's
FlexiblePhysicalExchange symbol). Importing the module does not install it.
The existing zero-action operation remains unchanged and does not use it.

The subclass requires the original 0.0021 m floor and an explicit after-SDK
geometry callback. It intercepts the frozen exchange's existing
sdk_complete_rod_pending read at the actual SDK boundary; no extra step,
reset, render, actuator write, physics setting or geometry calculation is added.
The source label must identify the existing actual PhysX link-transform/local
COM reader with original USD collision envelopes. It copies the two measured
clearances, global lowest source-shape index, actual SDK time and boundary,
then completes the matching rod advance and action readback via the original
_step. The same physical row receives the clearance packet and original
post-action identity.

A value below 0.0021 m adds post_step_source_clearance:left/right as the terminal
reason. An existing joint, mimic or 30 N guard keeps priority; the failed
clearance remains recorded simultaneously. Original control delivers the row
to its observer before breaking and permanently closes the exchange, so there
is no next target submission or SDK step. Finite/shape/source/phase failures
after SDK advancement raise immediately; the original control closes and
cannot replay. Such an incomplete SDK/rod boundary is a failure, not a fabricated
complete physical row.

The row validator independently checks exact floor/source hashes, pre/post
clock identity and SDK timestamp, recomputes failed sides and the minimum,
and recomputes original joint/mimic/force guard priority from the saved row.
It preserves native_clearance_qualified=false, hand_table_contacts=None and
Teacher admission=false.

## Evidence limits

The clearance values come from the original live reader, which copies current
SDK link transforms and calculates clearance from all original source collision
envelopes. The new record is an owned measurement and threshold audit; it does
not save every link transform/world vertex needed for a separate disk geometry
reconstruction. Its source hashes bind the reader and formula implementation.
Historical pad cooked hulls, current native PxShape attachment, complete actual
cooked meshes and hand-table contact remain separate unresolved evidence.
A source-envelope pass must not be presented as known native contact absence.

The extra state contains only copied lists/scalars/identity hashes; _step clears
the pending packet on success and error. Local self/callback arguments are
cleared in finally. The preexisting owned lifecycle still owns the hands,
views, rod, callbacks and native traceback cleanup.

## CPU verification

TDD first failed on the missing new module. Initial implementation then passed
16 tests, including actual builder/reset/LiveGeometryReader/eight-tick control.
After adding exact threshold and durable terminal-journal tests, the final
command with DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python,
DH116_NUM_THREADS=2, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1 was:

    scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_poststep_clearance_guard.py tests/test_dh116_flexible_physical_exchange.py tests/test_dh116_dual_hand_evidence_journal.py

34 passed in 38.69 s (19 new + 15 existing), exit 0.
Evidence directory:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/poststep-clearance-cpu-q4g0r4_0.

Tests cover safe full intervals, breach at first/last substep with no subsequent
action, existing guard precedence, malformed/nonfinite/stale/mislabelled geometry,
tampered stored floor/identity/time/flags/source/reason, exact-floor acceptance
and one-representable-value-below rejection. The existing durable journal
persists and fsyncs the terminal row before interval settlement; independent
journal replay and new row validation pass for that terminated prefix.
The builder integration uses original source geometry and SDK-shaped CPU
doubles; it grants no native physics or full 371D state qualification.

Repository.verify_protection() passes (528+2). All preexisting tracked files are byte-unchanged.
Independent read-only peer review checked this module plus the original
_geometry/_step/control ordering and reported no added blocker. That review
did not run native, validate actual geometric measurements or re-certify the
preexisting lifecycle's traceback cleanup.

New module SHA256: 98d95efd26b1cb7fb833649e897c30d361bde78807b28671f7132f118ceec9dc
New tests SHA256: 62ae0fba703ab9062a82bfac09a9cced3985cef38684a440f4275d016157549c

No recipe, registration, activation, authorization or native budget is created.
A later bounded nonzero diagnostic still requires explicit frozen action values,
its own audit of this added field, actual identity/clock/geometry checks and
normal ownership teardown. Original physical parameters and the 2.1 mm guard
remain unchanged; grasp, same-rod contact and Teacher qualification stay false.
