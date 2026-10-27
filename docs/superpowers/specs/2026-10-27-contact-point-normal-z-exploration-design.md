# Contact-Point and Normal-Based Z Exploration

## Problem

The current teacher probe controls a scalar hand load and moves a palm proxy in
the Z direction. `pad_relative` is a calibrated pad-position proxy, not a
measured contact point. As a result, a 10 N total normal-load target can still
produce a small signed vertical force when the real contact geometry remains
near the cylinder centerline.

The probe must explore the actual contact geometry: PhysX contact locations,
the corresponding normal-force directions, and the resulting vertical force.
It must not inject an external force or treat palm/pad proxy motion as proof of
contact-point motion.

## Confirmed requirements

- Use Isaac Lab/PhysX contact reports as the contact-point source.
- Keep the six target Z locations relative to the cylinder center:
  `-4.0, -3.0, -2.0, -1.5, -1.0, -0.5 mm`.
- Use the existing 10 N total normal-load target: 5 N per hand.
- Keep the authored XY pinch geometry and finger closure behavior.
- Explore only along world Z.
- For each hand, every effective contact point must be below the object
  centerline. The force-weighted contact center is used for target tracking.
- Reject a candidate when contact data are missing or insufficient; do not
  fall back to `pad_relative`.

## Data and geometry

Each sensor is configured to report contacts between one hand link and the
filtered cylinder. `contact_pos_w` is the PhysX contact-position aggregate for
that sensor body/filter pair, while `force_matrix_w` is the matching normal
force vector. The sensor keeps up to 16 contact samples per prim in the
tensor pipeline.

For hand `h`, valid contacts are those with finite contact position and a
positive normal-force magnitude. The measured contact center and normal are:

```text
p_h = sum_i(||F_i|| * p_i) / sum_i(||F_i||)
n_h = sum_i(F_i) / ||sum_i(F_i)||
```

The point and force are expressed in the world frame. The probe reports the
point relative to the cylinder center and the signed world-Z components of the
normal and friction forces. The normal-load target remains a magnitude target;
the force direction comes from the measured contact geometry.

## Control flow

1. Enable contact-point tracking on the filtered hand sensors.
2. At each physics update, aggregate valid PhysX contact points and normal
   forces per hand.
3. During close, compute the target contact-center Z error for the selected
   candidate and apply only a bounded world-Z palm correction. XY and the
   authored pinch/finger commands remain unchanged.
4. Hold the candidate until both hands have valid contacts, all effective
   contacts are below the centerline, the weighted centers are within the
   target tolerance, and the normal load is stable at 5 N per hand.
5. Record the measured point, normal, normal load, normal-Z component,
   friction-Z component, signed vertical sum, and lift gates.
6. Continue to lift only when the existing vertical-force admission and safety
   guards pass. Otherwise mark the candidate invalid or failed and reset.

The existing force-direction release/reclose state machine remains the safety
mechanism for changing a loaded grasp. It receives the measured contact-center
error and never teleports a contact point or applies a synthetic force.

## Interfaces

- Add a pure contact-geometry helper that validates sensor shapes, filters
  invalid contacts, computes force-weighted centers/normals, and reports
  per-hand validity and the all-contacts-below-centerline condition.
- Extend the tensor-world measurement contract with the measured contact
  center, normal, validity, and contact-count fields.
- Add the target-Z controller input to the teacher probe. The existing probe
  output gains explicit measured-contact fields; `pad_relative` remains
  telemetry only.
- Version the probe contract so old checkpoints cannot silently claim the new
  contact-point semantics.

## Failure handling

- No finite contact point, zero normal load, or invalid normal: candidate is
  invalid and cannot enter lift.
- Any loaded contact at or above the centerline: candidate remains in close or
  resets according to the existing close timeout/vertical-force guards.
- Contact-center error outside the configured tolerance after the bounded
  correction timeout: candidate fails; no fallback to a proxy point.
- Existing excessive-force, table-penetration, mimic, velocity, slip, and
  close-timeout guards remain unchanged.

## Tests and verification

Test-first coverage will include:

- weighted contact-center and normal aggregation with multiple contacts;
- invalid/empty contact reports and centerline gating;
- Z-only bounded correction and target tolerance;
- contract/version fields proving measured-contact semantics;
- probe telemetry containing measured contact fields and excluding proxy data
  from the control decision.

Verification will run the focused unit tests, Python compilation, and one
headless Isaac teacher probe over all six target Z values. The probe output
will include a cross-section figure with measured contact points, measured
normal vectors, and the 10 N load split.

## Scope boundary

This change does not alter object radius, density, material, lift height,
reward weights, or the trained policy checkpoint. It changes only how the
teacher position probe observes and reaches candidate contact geometry.
