# Tripod radius-aware distal contact and wrist relief

The opt-in diagnostic uses the measured 9 mm diameter, 450 mm free rod, both real DH116 hands and the completed supported Reset at SDK step 10. Each action checks all 34 USD source convex envelopes and 10 archived PhysX cooked finger pads against the matching wire radius. The selected three left distal pads may approach to −0.15 mm and the remaining shapes retain the original +0.25 mm clearance floor. A conservative bound falls back to exact finite segment-to-hull distance when necessary.

The left path holds the targets at the first valid distal pair with a measured left-side reaction. The following SDK step must confirm application of the matching wrench in PhysX. A separate final hold window covers contact at the final substep. On the right, CPU screening found a 3.0 mm transverse approach with +0.02 rad intrinsic-Z rotation and 0.361 mm minimum rod clearance across 13 samples. This path keeps the left hand targets constant. Both paths retain table and support checks.

A longer 4.3 mm pure upward left approach failed the exact geometric bound. The thumb and middle source envelopes reached −0.434 mm and −0.312 mm. This path was excluded before native execution.

## Verification

The CPU regression passed 87 tests, including physical spec split identity, contact transport timing, fail-closed wrench checks and exact radius-dependent clearance. The right approach freeze verified the source and raw receipt identities. The first native approach is recorded below.

The reference catalog holds out wire dimensions and keeps force observations in newtons and relative to measured weight. The diagnostic checks a single training specification. Holding out size, stiffness and physical initial conditions in a full grasp evaluation is required for generalization. Published grasp gates remain 30 mm off-support Lift, 5 s Hold and 0.5 mm Slip.

Next record actual wrist travel and the first correctly delivered distal reaction. Then validate sustained opposing reactions before the two support loads drop to zero.


## First live right-wrist attempt and hold-audit correction

The single Isaac instance reached SDK 11 in 74.10 s. After both setup steps and eight support-settling steps, the first right-wrist action completed one SDK exchange. The minimum rod clearance was 1.320 mm across 44 hulls. The rod remained on both supports and produced no hand pair. The hold audit stopped the run when the inactive left mount's translated targets moved from zero by at most 17.70 nm in local coordinates. App.close returned and GDB observed ordinary process exit 1, with no signal, timeout or forced ending.

The fixed hold audit now checks no inactive-side action and the exact projected world Reset pose and angles. It independently computes the expected SDK prismatic targets using the original mount conversion and the observed, fixed mount anchor. It checks all non-prismatic joint targets exactly and compares the entire backend target to the expected result. The archived SDK 11 row and original frozen Reset passed the new validation without changing any geometry threshold. A left tangential wrench without a valid normal pair now stops the unilateral probe, as does any hand wrench in the right pre-contact probe.

The regression passed 92 tests. The updated native path still requires its own run and result.

Antigravity independently matched the corrected hold auditor and the material catalog source. Only the selected left-pad SDFs are sampled in the contact probe. The right pre-contact probe checks the 44 hulls and measured wrench but still uses its single frozen positive-Z recipe.
