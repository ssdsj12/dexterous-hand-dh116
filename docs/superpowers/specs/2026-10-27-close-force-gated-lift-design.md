# Close force gated lift design

## Goal

Make close finish with a nonnegative signed hand-to-object vertical force before
admitting lift. During contact, downward force is penalized continuously. The
existing downward grasp-point repositioner gets a short correction window; a
failed correction resets the episode.

## Behavior

- Use the existing aggregate signed force convention: positive is world-up.
- During close contact, accumulate downward impulse
  `max(-vertical_force_n, 0) * dt` and add the existing continuous downward
  reward cost.
- Use the existing `-0.15 N` per-contact trigger for downward repositioning.
  Preserve its release, 1 mm correction, 2 mm maximum correction, and bounded
  reclose flow.
- Set the close downward impulse reset limit to `0.20 N·s`. Reposition timeout,
  exhausted attempts, or this impulse limit marks the episode failed so the
  normal environment reset path runs.
- Require aggregate `vertical_force_n >= 0` in close readiness and in the
  close-to-lift transition condition. Downward force cannot enter lift.
- Increase the coordinated close command slew from `0.05 rad/s` to `0.20
  rad/s`, while retaining joint envelopes, mimic coupling, pressure relief,
  and physical transport guards.

## Validation

Add focused CPU tests for continuous close impulse accumulation, reset at the
limit, reposition failure handling, nonnegative-force lift admission, and the
faster command slew. Run the existing DH116 task and controller tests after
the focused tests pass.
