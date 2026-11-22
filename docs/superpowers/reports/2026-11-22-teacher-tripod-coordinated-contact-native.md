# Tripod coordinated target ray A: actual native rejection
## Result
The supported flexible rod was not lifted. At SDK1801–2056 (256 zero-action hold rows),
right-hand loaded evidence passed256/256, left8/256; both hands were continuously loaded for
at most1SDK (2.083ms). External supports stayed loaded for all256rows.
Peak median rod displacement from SDK1704 was0.07936mm, not30mm off-support Lift.
No learned Teacher, PPO update, successful trajectory or Student expert qualification is claimed.

## Execution and immutable evidence
Execution HEAD765f875d2bf056e8eeab42e2427e551ff36f4734; source a95c1764fcbcdd580316f8cf337713a7b96caed5.
Frozen SHAefc57d90e16d2cd6c22451efeeaad55dd896c0a8b732abbfa58957ea98dd9c88.
GPU2, single instance/environment,2056SDK steps,227control steps,24648rod substeps;
12four-joint dose windows and32zero-action hold windows, no model/Teacher/PPO.
Exact SDK1704 replay boundary and actual24D target readback passed.
Full private records remain outside GitHub under
ref902/teacher-tripod-coordinated-contact-v1/native-run-v1; readiness stores hashes.
Raw arrays and original diagnostic versions are retained unchanged.

## Target delivery versus real movement
Left finger21 target advanced0.0005893707rad, measured q changed-0.0000262260rad.
Its target-minus-q grew0.00131255→0.00192815rad. Right targets remained unchanged.
The dose reached the backend; target-space reachability does not predict loaded joint motion.
This alone does not prove an actuator bug: PD tracking can differ under load.

Using actual same-SDK link poses and locked SDF at24endpoints, left index alternated
negative gaps/impacts with positive gaps/no contact. SDK2050:gap-10.566um and peak1.138934N;
SDK2052:gap+23.163um and zero contact. The approximate four-SDK cycle warrants coupling/contact
response analysis. CPU endpoint arithmetic is not GPU-bitwise or twelve-substep geometry proof.
No material, damping, frequency, physical limits, action/observation contract or threshold was changed.

## Exit and budget failure
App.close returned and the GDB inferior footer records normal exit0, noSIGSEGV.
However the600s soft limit was exceeded (last-log-write elapsed approx615.779s).
The private observer returned without waiting for the final child status, so an independent outer
exit code is unknown. The observer tool session failed1. No forced termination was used.
The post-timeout observation is explicitly not an outer-wait receipt; it cannot qualify natural
exit or the budget. Both recorded PIDs are gone. Future bounded supervisors must preserve the
soft-deadline failure and wait for actual child status without inventing0 or killing.
This observer defect is acknowledged separately from the physical grasp failure.

## Comparison limits and next work
Original SDK1513–1704 had24/192bilateral rows; the new hold had8/256.
Different time cohorts and denominators prohibit a paired improvement claim.
The original v1 reduction mislabeled the SDK1704 anchor as SDK1296; corrected v2 is hashed
separately and preserves the original file. No raw timestamp or geometry was changed.
Reject ray A and do not rerun this fingerprint or add dose along it.
Next: use actual contact/pose/force histories and source coupling order to distinguish
under-load oscillation from incorrect reaction timing or geometry. Stable opposing contact,
then off-support rod Lift, remains the mainline; existing protection stays in place.

## Validation
Source package:119CPU regressions passed, including31owner/hook cases; separate60Registry
tests passed before native execution. Publication reruns are recorded in readiness.
528protected sources and3historical checkpoints unchanged.
