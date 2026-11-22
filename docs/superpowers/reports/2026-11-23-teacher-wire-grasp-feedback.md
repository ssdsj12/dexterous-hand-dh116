# Causal pose and reaction observations for wire grasping

The supported-wire variant verifier now records a prospective policy observation from its last completed physical interval. Both hands use the same ordering for thumb, index, middle, ring, little and palm. Each side provides the measured wire station and unit tangent in the palm frame. Each available contact row provides minimum and mean normal load along with the XYZ reaction on the hand. Forces use newtons with the same scale for every specimen. The mask distinguishes a measured zero, a channel unavailable for this grasp and force evidence invalidated by the contact audit. Matching SDK and solver clocks, reset, episode and rod identity are required.

The source spec, exact material coefficients, weight and clock evidence stay outside of the proposed actor observation. The original 24 action coordinates and grasp acceptance remain fixed. This release records the observation in the resized-wire verifier. Existing Teacher models and their checkpoints do not consume it yet.

Offline packing of the sealed 9 mm Tripod endpoint at SDK 1426 reproduced six measured zero-force rows and six unavailable rows. That endpoint was on the supports, before hand contact. CPU tests check changing pose and force, shared distal/Power ordering, invalid channels, nonfinite or missing trusted hand data and cross-reset or stale input. This release starts no new Isaac run, and it provides no held-out success result.

## Tripod next physical change

The native SDK 3298 result came from one 9 mm wire. An offline geometric decomposition of that record transports the last loaded thumb contact point at SDK 3194 with the measured SDK palm and separately with the measured thumb link. At first loss (SDK 3204) palm motion contributed −21.43 µm along the old contact normal, while motion of the contact point relative to the palm contributed +2.09 µm. The node moved −18.93 µm. By first index load (SDK 3292), these components were −175.73 µm, +3.21 µm and −80.75 µm, respectively. This describes drift along the original normal, not a new SDF evaluation or evidence of which actuator caused it. The reported indexed contact and clean exit remain valid.

Before requesting further finger motion, evaluate a measured wrist settling and contact re-seating step. A correction must use the current wire geometry and a valid pad load, recheck all existing guards, and compare retained thumb load against the previous native run. A loaded index by itself does not unlock Lift.

## Generalization evidence still needed

Keep train, validation and test grouped by actual wire specification and independently sampled initial conditions. Evaluate the same frozen policy separately for Pinch, Tripod and full Power on held-out diameters, lengths and elastic stiffnesses. Use paired material variants to isolate stiffness at one size. Report relative pose and force corrections per grasp, support separation, net Lift ≥30 mm, Hold ≥5 s and Slip ≤0.5 mm. Report the lowest success rate among tested specimens alongside the aggregate.
