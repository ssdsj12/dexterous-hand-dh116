# Native wire cross-section approach from measured palm pose

The opt-in Tripod probe reuses the published measured 9 mm × 450 mm Reset and the original DH116 articulations, supports, DeformX rod and 24D exchange. After the parked SDK10 endpoint, the decision binds both PhysX palm poses, the current material station and tangent, the live CUDA rod endpoint and the 12-substep reaction packet. It computes the transverse direction in World XY from each palm's actual orientation. The 60 Hz position increment is limited to 50 µm per hand. Z, finger and wrist rotation action channels remain zero. A hand reaction or invalid pair provenance blocks the pose pulse.

The original action plumbing verifies the submitted targets and uses the previous completed rod reaction at each SDK write. The non-contact rod envelope uses the actual verified radius. The existing table clearance and both support guards are applied during each substep. Any contact, support transition or active guard stops the diagnostic with its trace intact.

## CPU checks

- 94 passed. The new tests cover different radii, rotated palm frames, same-tick binding and contact-gated holding.
- Replaying the preceding sealed native SDK10 sample selected X = −0.3 for each hand on the original normalized translation channels. This is only an offline action calculation.
- The independent CPU process froze one Reset, 18 maximum SDK ticks, 192 maximum DeformX substeps, one control window and zero PPO. The native result is recorded below.

This probe tests the causal approach entry. It does not exercise pressure adjustment or perform a grab or lift. After actual pose and action readback, the next stage must demonstrate stable opposing reactions. Keep the original Lift, Hold and Slip gates, and measure all three grasp styles independently.

Antigravity read the driver, calculation and tests. Its two runtime findings were reproduced in source: the existing wrist projection requires float64 bridge inputs, and terminal clearance must be recomputed with the rod at SDK18. Both were fixed. The float32 readback regression exercises the final conversion.

## Bounded native result

GPU2 executed the published source `27117dc6fb1b0eca29bf7482dad6da901e3b69a7` with one Reset, 18 SDK ticks, 192 DeformX substeps and one actual control window. No model queries or PPO updates occurred. Isaac and the debugger exited naturally with code 0 in 66.462 s. An independent CPU audit verified all 47 artifacts, frozen source bytes, each submitted target, each PhysX backend position target and the same-clock terminal rod clearance.

| Measured | Left | Right |
| --- | ---: | ---: |
| Requested transverse translation | 50.000 µm | 50.000 µm |
| Actual palm travel along the request | 26.047 µm | 25.988 µm |
| Transverse distance to the selected wire station at SDK10 | 102.886 mm | 53.458 mm |
| Transverse distance to the selected wire station at SDK18 | 102.860 mm | 53.432 mm |

The minimum measured hand-to-wire clearance in the pre-contact guard was 1.868 mm. The end supports retained the wire and all eight action ticks had zero hand pairs and zero table pairs. The palm geometry reader, target routing and action verification are connected to real physics. This result does not qualify contact, Lift, Teacher training, or generalization.

The next controller needs per-pad targets in addition to palm geometry. The two isolated nominal bilateral ticks (2066 and 2094 in the prior 10 mm native trajectory) both placed the left three distal contacts on nodes 33/35 and the right three on nodes 2/3/4. The nearest material nodes to the palms were 31 and 9 respectively. These observations can seed per-pad targets and compare the left/right reaction under small corrections. They do not provide a stable grasp reference.

Frozen SHA256 `217fd946b2b17f64169598ab9eed4ec21728d1bd358d7ec92740a88f7fdeaed0`. Outer receipt SHA256 `b9dabfd4749575c23b0ea92777ddddfada6ac88c9cd71a097e03a91aa3bd35c4`.

The CPU geometry screen used 44 hulls for material anchors (35,5), (35,3), and (34,3). Moving only the right anchor to node 3 increased the minimum hand–support gap from 2.513 mm to 3.688 mm while retaining the 1.747 mm minimum hand–rod gap. The next diagnostic candidate uses (35,3) and still requires native verification.
