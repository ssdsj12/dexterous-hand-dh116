# Native Reset for the staged Pinch candidate

The source-locked Reset owner read the fixed candidate published in `5e45992f675f3d27c287a33c63ffc00da4543040`. The run used the published manifest commit `bce9590b905090b15bf2472675418085b675466c`, one environment, GPU 2, one Reset and two setup SDK steps. It ran no control steps, rod evolution, Teacher queries or PPO updates. The outer observer recorded a natural exit 0 after 46.08 seconds. The existing Reset seal validated the complete native capture and the content hashes of its outer receipt.

## Measured state

| Measurement | Left | Right |
| --- | ---: | ---: |
| Lowest table clearance at the actual link poses | 2.365216 mm | 2.326099 mm |
| Smallest convex-envelope to rod-capsule gap | 0.537286 mm | 0.863737 mm |
| Maximum active angle error against the candidate | 1.58e-8 rad | 2.23e-8 rad |
| Maximum measured joint speed | 0 | 0 |
| Measured finger12 position | 0.454557 rad | 0.008949 rad |

Each hand used 17 source envelopes and 5 locked cooked pad envelopes at current SDK link poses. `finger13_link` controlled the table clearance on both hands and `finger14_link` controlled the rod gap. The geometry reader did not recook hulls or prove live PxShape attachments. The two numeric Reset samples were complete and byte-equal and remained at rod tick zero. The pre-action readback agrees with the prescribed wrist poses and active angles. Right `finger12` is only 0.001036 rad above its active lower limit, so its actual position and speed during settling and closure still need readback.

The original guard judged positive initial geometry at the same Reset clock. No settling contact, loaded opposition, Lift, Hold or trained performance was measured. The candidate stays separate from existing trained policies. The previous 30 mm Lift, 5 s Hold and 0.5 mm Slip criteria still apply.

## Evidence and next gate

- Frozen recipe fingerprint `7b2a3db8d294ec5fad892b630070f03fab9c652264a535693bbdd181a5d95877`.
- Outer receipt SHA-256 `5234d161cc6b99af59c4c59c477a8f4e0e4cc552269b4b564f497c95ac5204b8`.
- Resolved native Reset manifest SHA-256 `2f8ec96ba01922b81502b4575a7b9657bbae0bc8c928c14698b3811936e2e9a8`.
- Independently checked measurements SHA-256 `1765728a3dcccdcc32594ef1158c92bf8d974c7a7caf8efb0f3d4c1e006c7caa`.

The next separately bounded experiment uses the sealed Reset, verifies the unloaded settling window, and executes the published 64-step close while recording the original measured guards and same-substep opposing reaction forces. It must stop before Lift if either hand loses clearance, exceeds tracking limits, or fails to establish opposing contact.

The focused CPU regression completed 142 tests with exit 0. The expanded CPU run including the older full builder fixture was stopped by its 170-second timeout before pytest produced a summary. That incomplete run contributes no passing count.
