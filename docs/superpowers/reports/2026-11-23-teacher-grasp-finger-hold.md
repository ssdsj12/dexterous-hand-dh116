# Fixed nonparticipating finger targets for native Pinch and Tripod diagnostics

The opt-in finger hold keeps the original 24D action and both wrist groups. Pinch moves the two thumb joints and index per hand. Tripod also moves the middle finger. The hold retains the backend targets of all other fingers at phase entry.

| Grasp | Active fingers per hand | Held 24D joint action channels |
| --- | --- | --- |
| Pinch | Thumb and index | 3, 4, 5, 9, 10, 11 |
| Tripod | Thumb, index and middle | 4, 5, 10, 11 |
| Power | Original channels | None |

## Controller boundary

For an authored action, call `verify_action` before submission. It requires exactly zero in the held channels. A candidate generator can use `project_action` before evaluating a new NumPy or Torch action. Projection preserves the original shape, dtype, participating channels and wrist commands. Torch gradients on projected channels are zero. Retest any newly projected candidate against the original motion and contact gates before use.

Create a new `GraspFingerHold` at the first opted-in phase, proposed after Pinch SDK 432 or Tripod SDK 568 for these archived lineages. Call `observe` on each consecutive native readback in that phase. It binds the first pre-action direct PhysX backend target and checks the same binary32 value for each held joint in pre-action, submitted and post-action buffers, including across policy windows. It requires the original named active-index bindings, two distinct side mappings, and a contiguous SDK/solver clock in the same Reset. Recreate the instance or call `clear` at a new phase or Reset. Existing controller guards still own target submission and may report a correction. The hold flags any correction that changes a fixed target. Actual measured joint positions may move through compliance.

## Read-only verification

The sealed live Tripod 902 archive accepts all eight historical controls at SDK 553–560 with bit-exact held targets. Its next historical window is rejected on SDK 561, where the original authored action puts normalized residuals of about `−2.48e−6, −1.99e−6, −1.04e−5, −1.19e−5` into the held channels. The sealed Pinch right-first-contact archive rejects the historical SDK 433 action for residuals whose magnitudes range from `2.48e−7` to `3.97e−6`. These negative controls preserve the documented scope of the opt-in. Keep the historical prefix byte-for-byte and apply the hold to the newly authored phase.

At the achieved Tripod SDK 568 cutoff the original direct backend targets are 0.700000, 0.950000, 0.250014, and 0.250016 rad in held-channel order. At the archived Pinch SDK 432 cutoff they are 1.150349, 1.149912, 1.151035, 0.275317, 0.564244, and 0.251243 rad. The replay evidence preserves unrounded target and measured values. The helper takes its baseline from the live pre-action buffer.

Verified on CPU with 111 tests, including the four sealed Tripod interlock tests. The isolated work covers the observer and projector and includes no new native execution, policy integration, PPO admission, or grasp success claim. The archived replay program, input receipt hashes, result and CPU log are recorded in [the finger-hold evidence contract](../../contracts/dh116_teacher_grasp_finger_hold_v1.json).
