# PPO synchronized-Z force-phase training design

## Goal

Train the two hands as one synchronized grasp search. During close, both palms
must explore the same world-Z height while the task evaluates only the signed
vertical component of hand-to-object normal force. After lift admission, the
task may use the signed vertical sum of normal and friction contact forces for
lift stability and success. Each hand's policy force target is bounded to
2--12 N. The existing close reward, downward penalty, and four-second close
reset behavior remain unchanged.

## Current context

The tensor action has 20 entries: 12 joint targets, six palm translations, and
two per-hand force targets. The current palm controller can restrict close
exploration to `z_down`, but the policy path must apply one shared Z command to
both hands while permitting only a low-weight ±0.5 mm XY refinement. PhysX
reports normal and friction contact components separately;
the close task already consumes the normal-only signed vertical sum. The old
force target range is 0.1--12 N per hand and must be raised at the lower bound.

## Design

### Synchronized Z exploration

For every live close control tick:

1. Restrict each policy palm action to a small XY search window of ±0.5 mm
   and a non-positive Z search window of up to 4 mm. XY action magnitude and
   its exploration regularizer are weighted at 25% of the corresponding Z
   terms, so XY can refine a grasp without competing with the vertical search.
2. Compute one policy Z value as the mean of the two hand Z actions.
3. Compute one contact-point correction from all valid PhysX contact centers,
   using the force-weighted center and the shared target Z.
4. Apply the same bounded correction to both hands and write the same final
   world-Z target to both palms. The XY policy offsets remain small and are
   applied symmetrically with the shared Z command.

Invalid contact centers are masked before aggregation. If neither hand has a
valid contact, the correction is zero and the shared policy Z value is kept.
When the phase reaches lift, palm exploration is zeroed; the task's common
lift offset moves both hands together.

The synchronized controller does not use `pad_relative` or kinematic pad
centers for contact decisions. A candidate is eligible for direct lift only
when all valid hands have contacts below the object centerline and the shared
close force gate is satisfied. XY exploration is a refinement around the
shared grasp pose; it does not create independent per-hand Z heights.

### Phase-specific force semantics

Let `N_z` be the signed world-Z component of the object-filtered normal force
from each hand, and `T_z` the signed world-Z component of the corresponding
object-filtered friction force.

- Close force: `F_close_z = sum(N_z_left, N_z_right)`.
- Lift stability force: `F_lift_z = sum(N_z_left + T_z_left,
  N_z_right + T_z_right)`.

Close readiness, the close upward/downward scoring, the close downward event,
and the four-second close timeout continue to use `F_close_z`. Friction remains
telemetry during close. After the phase changes to lift, `F_lift_z` is used only
by the lift admission/stability gate and the physical lift success check; it is
not added back into the close reward or penalty contracts.

Any non-finite normal/friction component invalidates that force sample. Missing
valid contacts cannot admit lift. Existing physical safety guards and reset
handling remain authoritative.

### Per-hand force target range

The two force action entries continue to be independent per-hand targets, with
the mapping changed to:

```text
target_n = 2 + 0.5 * (action + 1) * (12 - 2)
```

The policy therefore explores 2--12 N for each hand, or 4--24 N nominally in
total. The measured load remains a separate observation and is not treated as
the signed vertical support force. A fresh policy's force-action bias is set so
its initial request is just above the 2 N floor. Checkpoints carrying the old
0.1--12 N force contract are rejected for ordinary resume; any explicit
weights-only migration must record the changed action mapping in the contract.

### Contract and compatibility

The implementation will add a versioned phase-force contract recording:

- synchronized world-Z palm control;
- close normal-only force source;
- lift normal-plus-friction gate source;
- per-hand force bounds `[2.0, 12.0]`;
- unchanged close reward/penalty versions.

The previous probe artifacts remain historical evidence. New training and
probe outputs will write the new contract and will not silently relabel old
normal-plus-friction reward data as normal-only data.

## Data flow

The policy emits joint, palm, and force actions. The coordinated controller
normalizes the palm Z actions, shares one Z value, and applies the measured
contact correction. The force controller maps the last two actions into the
2--12 N target range and regulates measured per-hand load. The world adapter
publishes normal and friction vertical components separately. The task selects
the normal-only close channel before lift and the combined lift channel after
lift admission. The recorder stores both channels, the phase, contact validity,
and the final close sample so a candidate cannot be called successful from an
episode average.

## Testing and validation

Unit tests will cover:

- identical left/right final Z targets for valid, partial, and no-contact
  reports;
- NaN masking for invalid contact centers and force components;
- exact 2--12 N force-action round trips and rejection outside the bounds;
- close gates ignoring friction;
- lift gates using normal plus friction only after phase 3;
- unchanged close reward and downward penalty contract versions.

The validation run will use the latest teacher checkpoint only as an inference
policy, write a new output directory, and report for every Z candidate:
close-end normal force, lift-phase combined force, per-hand loads, contact
centers, contact validity, failure code, and actual lift success. A candidate
is successful only if the task reaches lift and satisfies the existing success
dwell and slip/height checks.
