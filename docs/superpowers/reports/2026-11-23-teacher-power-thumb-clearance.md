# Power thumb closure on a measured wire

The archived dual-palm boundary provides the measured DH116 wrist, joint and rod state at SDK 6504 and rod tick 78024. The wire diameter is 10 mm. This audit resolves the original left and right registered SDFs, verifies the two predecessor audits and retains the source/cooked colliders, calibrated finger offsets, support and table geometry. The search samples the original coupled thumb controls without changing the mimic equation or limits. For each fixed wrist and partner finger it checks same or adjacent interior material nodes, opposing normals, thumb SDF proximity and clearance of the full rod against the thumb middle link.

| Hand | Screened wrist/partner candidates | Thumb angle pairs per candidate | Middle link clear | Surviving closure | Middle-link lower bound |
| --- | ---: | ---: | ---: | --- | ---: |
| Left | 38 | 2,604 | 49 | Thumb and index, nodes 37 and 36 | 0.691 mm |
| Right | 23 | 2,604 | 7 | Thumb and ring, nodes 4 and 3 | 3.311 mm |

The remaining middle-link rejections were certified below the 0.25 mm margin for 615 left and 542 right joint proposals that reached the initial witness screen. The selected left and right pad normal cosines are −0.7863 and −0.4329. These are static geometry witnesses. The loaded rod shape and tracking error under movement require fresh physical measurements.

## Sampled motion

The left standalone abduction ended with the distal thumb already touching the rod. The thumb-first sequence explicitly scopes the existing target-pad penetration allowance to that pad. During straight wrist transit, six of 17 samples failed, with the first invalid palm position at 37.5% of the transit and a signed gap of −0.695 mm. A sinusoidal retraction of up to 1.5 mm along the archived palm normal keeps the same starting and ending poses. The exploratory 0.5 mm and 1.0 mm retractions still failed.

The right isolated abduction had two rejected samples. The first support clearance was 1.017 mm against the original 2.1 mm minimum. The linear thumb-first variant had one rejection at 1.995 mm. Advancing flexion with a quadratic progress curve kept the same joint endpoints and cleared the sampled support obstruction.

The fixed detours passed all 33 samples in each of wrist transit, thumb-first and opposing-pad closure on both sides. The offline motion screen uses the existing workspace and hand separation bounds and checks the wire and static scene at each sample. This is a sampled route on a fixed snapshot.

The sampled routes keep the opposite hand at its archived pose. They still require a continuous swept-geometry check and a staged live verification with each hand's independently measured position, pad load, opposing reaction and applied wrench on the next tick. Two-pad opposition here is an intermediate step toward Power. Full Power requires palm support and all five fingers closing around the wire. A restart must preserve the original support, reset, motion and success contracts.

## Generalization

One geometry at one archived tick cannot establish held-out performance. The existing factorial catalog separates training from eight held-out combinations of 9.2 and 10.5 mm diameter, 475 and 525 mm length, and 0.7× and 1.3× elastic modulus. The existing 104-start offline cohort cleared 24 Pinch, 72 Tripod and 0 Power starts. These are geometry screen results. They give no evidence of a successful physical grasp.

The existing causal feedback pack supplies palm-relative wire position and tangent, per-pad normal load, reaction vector and channel validity. It can support adjustments to wrist pose and finger closure from the last complete physical interval. The current Teacher checkpoints do not yet consume that feedback. Physical acceptance requires the same frozen policy per grasp across the held-out specs, actual off-support Lift of at least 30 mm, Hold of at least 5 s and Slip at most 0.5 mm. Report each spec and the worst spec alongside aggregate results.

Antigravity reviewed actual server source and identified the coupled thumb bounds and same-material-node geometric checks. Codex reproduced them from the pinned native ledger and validated the two local changes on CPU. This package uses no new Isaac or PPO steps. The readiness file records exact inputs, invocation, hashes and regression results.
