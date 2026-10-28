# Approved reward design

User approved improving force direction first, then actual lift milestones; retain additive close-downward -3 and ongoing negative force cost. Preserve current policy and training progress.

Force potential: normalize measured signed hand-to-object Fz by runtime mg; low-pass normalized force with tau=0.05 seconds, Phi=2*tanh(filtered_ratio). Exactly once per PPO control transition, add gamma*Phi(next)-Phi(previous), gamma=0.995 matching PPO. Terminal next potential=0 (success and all episode failures); rollout end is not terminal. Reset force ratio=0. This is potential shaping, including terminal cancellation, not a standalone positive-only progress bonus. Direction deterioration can subtract shaping; this does not change the previously exempt continuous negative-force cost during close.

Physical lift milestones: bottom at least table+3 mm and +10 mm, valid phase lift, speed<0.03 m/s, slip<=0.02 m, measured-load readiness, sustained 0.1 seconds. Award +2 and +3 once each per episode. Loss of stability resets dwell, paid milestones stay paid until episode reset. Existing 30 mm/0.5 second success +25 and all existing rewards/guards stay unchanged.

Append five observations: tanh(filtered force/weight), two milestone dwell fractions, two paid bits. 151->156 exact migration: retain old policy columns and Adam moments, zero new columns/moments. Additional explicit reward contract; no unrelated checkpoint drift accepted.

Read-only lift gate telemetry: fraction of lift substeps blocked by mimic, tracking, measured load, slip, pressure; command advance and height-cap fractions. Gates overlap; do not relax any thresholds. Keep event force displays and all-step displays.
