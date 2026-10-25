# Controlled mimic-boundary episodes

The user requests sustained PPO learning and intervention when control fails.
V30 and the nf400 evaluation both stop globally on finite, bounded mimic
deflection near 0.02 rad. The callback returns before SB3 adds that failed
transition to the rollout, so the policy cannot learn from those failures.
Changing palm dynamics and increasing mimic stiffness did not resolve this.

Investigate a different lifecycle, initially in a throwaway diagnostic:

- Keep the original 0.02 rad episode termination, failure label and penalty.
- Only a finite mimic-only terminal state below 0.03 rad is eligible for reset;
  joint-limit, force, geometry, object-drop and nonfinite guards still stop.
  The additional bound limits reset eligibility, not permitted episode motion.
- Immediately reset the failed slot before advancing the shared world again.
  Preserve terminal observations, diagnostics and reward before reset.
- Refresh and validate reset slots at every remaining physical substep;
  any failed reset or repeated physical fault during that interval stops all.
- Validate repeated real fault/reset cycles and other-slot isolation in Isaac
  before changing the training lifecycle. No additional Isaac instance.
- Only explicitly validated recovered transitions may enter PPO training;
  evaluation continues to count the original fault as unsafe and unsuccessful.
- Production use requires opt-in contract metadata and strict weights-only
  migration, unchanged actions/reward/physics and fresh optimizer/counters.

No successful grasp or stable long-run training is claimed by these probes.
The nf400 variant is not qualified for training; retain the v30 nf200 weights.
