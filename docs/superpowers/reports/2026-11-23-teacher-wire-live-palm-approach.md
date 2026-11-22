# Native wire cross-section approach from measured palm pose

The opt-in Tripod probe reuses the published measured 9 mm × 450 mm Reset and the original DH116 articulations, supports, DeformX rod and 24D exchange. After the parked SDK10 endpoint, the decision binds both PhysX palm poses, the current material station and tangent, the live CUDA rod endpoint and the 12-substep reaction packet. It computes the transverse direction in World XY from each palm's actual orientation. The 60 Hz position increment is limited to 50 µm per hand. Z, finger and wrist rotation action channels remain zero. A hand reaction or invalid pair provenance blocks the pose pulse.

The original action plumbing verifies the submitted targets and uses the previous completed rod reaction at each SDK write. The non-contact rod envelope uses the actual verified radius. The existing table clearance and both support guards are applied during each substep. Any contact, support transition or active guard stops the diagnostic with its trace intact.

## CPU checks

- 94 passed. The new tests cover different radii, rotated palm frames, same-tick binding and contact-gated holding.
- Replaying the preceding sealed native SDK10 sample selected X = −0.3 for each hand on the original normalized translation channels. This is only an offline action calculation.
- The independent CPU process froze one Reset, 18 maximum SDK ticks, 192 maximum DeformX substeps, one control window and zero PPO. Native performance has not been verified for this profile.

This probe tests the causal approach entry. It does not exercise pressure adjustment or perform a grab or lift. After actual pose and action readback, the next stage must demonstrate stable opposing reactions. Keep the original Lift, Hold and Slip gates, and measure all three grasp styles independently.

Antigravity read the driver, calculation and tests. Its two runtime findings were reproduced in source: the existing wrist projection requires float64 bridge inputs, and terminal clearance must be recomputed with the rod at SDK18. Both were fixed. The float32 readback regression exercises the final conversion.
