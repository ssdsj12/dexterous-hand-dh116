# Pinch wrist sweep at the SDK submission boundary

Archived-v13 replay and CPU exchange tests only. No native physics or PPO steps were added.

## What changed

- Build a continuous kinematic table-clearance check from the measured link frames, all 11 finger coordinates per hand, the original active target and mimic map, and the intrinsic XYZ wrist convention. Each hand uses its 17 authored collision envelopes and five locked cooked pad envelopes.
- Preserve the original per-step translation and rotation caps, workspace, tracking guard, and 2.100 mm floor. Check the whole path, including its measured start. Any optional extra clearance reserve applies only to the endpoint.
- Add an explicit, isolated exchange binding. It requires both hands to pass before composer writes or SDK submission. An exercised rejection leaves the SDK, solver and writer untouched and restores the original bindings. A later recipe must bind the live geometry checker and recorder explicitly.
- Antigravity's read-only source review identified that applying a Z minimum after the existing rate clip can exceed the translation rate, and that the sum of absolute intrinsic-axis changes is needed for a swept-path angular bound. Both were verified against the source. The existing exchange and historical frozen recipes retain their original bytes.

## Archived results

Journal SHA-256 `3b46bc46f0b615eca40e26c8bafec9c5df9b873cb89ddaf45b171889f5cf2b7b`. The final audit reproduced SHA-256 `fa8d2d378d9e52ef04558f07ea257eddb96fb4a2dc1e113634f1a4783445a744` from 116 measured SDK boundaries. Each row is evaluated independently against its recorded previous target and measured state. Earlier counterfactual targets could change both.

| Result | Left | Right |
| --- | ---: | ---: |
| Original candidate certified | 34 | 78 |
| Alternative candidate certified | 40 | 7 |
| No enumerated candidate certified | 42 | 31 |
| Best rate-limited endpoint upper bound below 2.100 mm | 12 | 24 |
| First uncertified candidate set, SDK step | 73 | 88 |
| First upper bound below floor, SDK step | 93 | 95 |
| Upper bound at that first step | 2.091084 mm | 2.090396 mm |

The finite candidate search can reject feasible motions. The endpoint upper bound proves a stronger result for those 12 and 24 rows under the kinematic model and archived finger targets, even allowing the full original wrist rate balls. These are conditional geometric results. They do not predict the actual PhysX trajectory or force. The minimum original-command endpoint in this replay was 1.940884 mm left and 1.752982 mm right.

A 50 micrometre endpoint reserve was screened independently. The first candidate failures moved to SDK steps 56 and 84, with 63 and 35 failures respectively. A second archived screen shifted measured pad positions upward by 0.4 mm against the unchanged rod. The minimum thumb/index SDF distances increased from 1.651/0.604 to 1.737/0.750 mm on the left and from 1.270/2.390 to 1.287/2.580 mm on the right. An earlier wrist/path correction must preserve rod reach. No revised pose or reserve was promoted.

## Verification and next gate

Final targeted regression on Ubuntu, 40 passed in 168.01 s, exit 0. The replay tool independently reproduced the digest above.

A broader CPU run had 42 passes and two retained-receipt failures. The old checker requires its literal receipt path to equal the resolved path, but the Ubuntu home storage resolves under a different root. The receipt SHA-256 still exactly matches `97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746`. The historical script, tests and receipt were retained.

First design the earlier bilateral approach around these limits. Re-audit the same 44 envelopes and the rod-to-pad reach, then bind the live checker in a separate frozen recipe with actual tracking, reaction-force and exit receipts. The prior short PPO update established collector/optimizer wiring. Pinch contact, and full Pinch, Tripod and Power qualification, remain pending. The original Lift, Hold and Slip criteria apply.
