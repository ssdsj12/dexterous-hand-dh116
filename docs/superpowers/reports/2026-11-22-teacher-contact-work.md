# Actual reaction timing and grasp-contact response
The mainline is opposing contact, stable carrying and real off-support rod Lift.
This package adds a reusable read-only calculation on the failed Tripod run to choose a
physical fix from evidence rather than increasing closure again.

## Actual code
scripts/audit_dh116_grasp_contact_work.py: endpoint_work computes World-frame force times
COM displacement plus World torque times relative rotation (XYZW quaternion convention).
audit_pair binds adjacent actual SDK records, unchanged run/episode/Reset/rod/body identity,
twelve rod substeps, and the previous reaction really issued to PhysX.
main reads the requested bounded archive interval and its preceding sample, hashes each input,
writes an exclusive fsync-ed result, and never advances simulation or changes forces.
No new controller, simulation setting, qualification exemption or safety subsystem was added.

## Real archived result
257 inputs from execution765f875d produce256SDK intervals (1801–2056), both hands and14links.
Left index issued work estimate:+1.3584878mJ; current rod reaction work estimate:-0.3433047mJ;
difference:+1.7017925mJ. Right index corresponding estimates:-0.000041812mJ,
-0.000129819mJ and+0.000088007mJ.
Left thumb difference:+0.3107987mJ; right thumb:+0.000268463mJ.
These are endpoint quadrature estimates, not a complete measured energy balance.
They are consistent with a much larger lag/work mismatch on the intermittently loaded left
index; they do not alone prove a unique numerical instability mechanism.
Original authored left/right distal index masses are nearly equal (3.65404/3.65406grams);
no asset or inertia was modified. Effective articulated inertia and contact conditioning can
still differ with pose, and authored mass is not a new native mass measurement.

## Source-backed interpretation
FlexiblePhysicalExchange._step writes the completed previous rod reaction, advances PhysX,
then solves the rod through the actual measured hand motion and queues the next reaction.
This is explicit partitioned coupling; its order alone is not evidence of a routing bug.
Locked contact.cuh computes normal penalty k=m*(kappa/h)^2 and N=-k*gap, with no normal dashpot.
Do not estimate this k from a maximum substep force divided by a different endpoint gap.
The actual target readback is correct; loaded PD tracking differs from target-space FK.

Antigravity actually read the current coupling sources in CoordinationR4. Codex accepted its
adjacent-record work comparison and rejected several overclaims: SDK2049 was not zero force
(index peak0.15234N); a stable right hand does not logically exclude every common coupling
defect; positive endpoint difference is not strict proof of energy injection; thumb causality
is still a hypothesis. Original UI reply is retained privately with a SHA in readiness.
The four-round coordination review is complete.

## Validation and next step
Ten new cases first failed for the missing implementation, then passed. The focused suite
passes38cases total, including World rotation/quat-sign and joule-scale checks and actual
reaction identity rejection. The real archive CLI completed and produced a parseable result.
No new Isaac run, physical step, Teacher query or PPO update was made in this package.
The preceding actual run still has left8/256loaded rows, bilateral longest1SDK and no Lift.
Next work must isolate the one-SDK reaction/contact response before selecting a changed
coupling or contact-acquisition candidate; do not rerun rejected rayA or add its dose.
Retain the original controller, material, clock, limits, fixed stereo and success criteria.
