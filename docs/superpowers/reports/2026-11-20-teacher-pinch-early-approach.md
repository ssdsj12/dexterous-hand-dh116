# Pinch early approach on owned SDK boundaries

This offline tool screens a combined wrist and thumb/index target from a frozen native Pinch journal. It does not submit commands, produce a native recipe, or grant Teacher admission.

## Method and source review

Antigravity reviewed the exact Ubuntu HEAD `1de5eff838ba013bf7556a293d6e354e89e51850`. Its useful recommendation was to maximize over all four pad gaps in each *single* SDK snapshot before comparing times. The audit independently enforces frame, run, reset, rod and clock identity on the original link and rod snapshots, and binds each SDF to its side, live link and cooked capture digest. Each hand's normal dot product is evaluated at its chosen nodes from the same snapshot. The original negative SDF margin of 0.100 mm and opposition criterion of cos ≤ -0.25 are used as geometric screening conditions.

The 44-envelope table proof includes the entire linearized wrist/finger command path and original 2.100 mm floor. A separate bounded support-plane calculation checks all non-target envelopes against the fixed archived rod, and the complete hand sweeps must separate along world Y. The initial target pads must have nonnegative signed gaps. A native run would still require original command tracking, contact reactions, timing, support and force receipts. The displayed SDK step requirement is a lower bound from the declared rate and the measured-state displacement alone. It omits controller tracking and is not an actual action sequence.

## Results

The candidate uses bounded translations, intrinsic XYZ turn and the original thumb/index channels. The left fit used SDK 16/32/48. The right numerical search reached its iteration limit, so its last candidate was retained to identify why it fails. Neither side's independent fit is a successful bilateral candidate. Screens also use SDK 64/80/96/112 on the same original rollout as a later-state check, not a holdout episode or a prediction of how the rod would move.

- Candidate requires at least 43 SDK steps left and 144 right under the original slew caps. A 96-step diagnostic input rejected it before geometry screening. All seven geometry rows use the same longer hypothetical bound of 144 with zero SDK steps executed.
- At SDK 16, both table paths cleared the original floor in the kinematic calculation. The four predicted gaps were left thumb/index -0.226/-0.100 mm and right thumb/index -0.269/-0.183 mm. The left index result was 0.0044 micrometre outside the exact negative-margin gate. The two predicted opposition cosines were -0.826 and -0.736. All initial pad gaps were positive.
- The right `source:finger13_link` could not be separated from the rod at the midpoint or endpoint of that candidate. An independent finite-segment/convex-geometry calculation on the endpoint verified the source hull to capsule surface gap of -0.278 mm at SDK 16, -0.187 mm at SDK 32 and -0.020 mm at SDK 48. These are calculated signed gaps to the fixed rod using the source convex envelope. They do not give penetration depth, contact load or impulse.
- At SDK 64 the right envelope can be separated from the fixed rod in this model, but the right table sweep no longer passes and the paired pad margin is missing. No one frame passed all bilateral screening gates. No recipe was generated.

Next, test a right thumb orientation or an initial staged path that clears `finger13_link` and `finger22_link` before opposing the wire. Tie the stage exit to measured same-substep opposition and wrench on both pads. A larger wrist slew or combining independent gap minima cannot qualify the geometry.

## Reproduction and regression

Run in the Ubuntu project environment with the exact archived v13 journal and the checked candidate.

```sh
python -m scripts.audit_dh116_pinch_early_approach \\
  --project "$PROJECT" --journal "$V13_JOURNAL" \\
  --journal-sha256 3b46bc46f0b615eca40e26c8bafec9c5df9b873cb89ddaf45b171889f5cf2b7b \\
  --candidate "$PROJECT/configs/dh116_execution/pinch-early-approach-archived-screen-v1.json" \\
  --output "$UNUSED_OUTPUT"
```

Final retained replay SHA-256 `2edf3ef620146c81b38e55d7f0625ce1829d0d7fcaea5e12a096333dc72748a6`. Additional capsule witness SHA-256 `b8518e51856f155a87c5380d1bdf445a381a224bc17090870f883b5d219cc386`. Ubuntu CPU regression, 116 passed in 20.76 s, exit 0. Isaac, solver and PPO steps added were all zero.
