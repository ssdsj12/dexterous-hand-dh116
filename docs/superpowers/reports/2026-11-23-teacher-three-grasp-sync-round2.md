# Pinch, Tripod and Power parallel checkpoint, round 2

Three Ubuntu worktrees ran independent CPU packages using authenticated Isaac evidence. The shared target is the supported flexible DeformX wire, with the published 24D action map, left and right hand identities, and both fixed RGB cameras preserved. This round ran zero new Isaac steps and zero PPO updates.

| Grasp | Measured archive or CPU result | Next test |
| --- | --- | --- |
| Pinch | On the SDK 424–432 reference, the right thumb/index gaps changed from 0.645/0.024 mm to 0.431/0.215 mm at material station 4. The best warm candidate uses −0.06 rad of tangent roll, with normal dot −0.989, proximal hull clearance 0.279 mm and support clearance 4.263 mm. The two gaps are positive. All nine cold seeds and the warm candidate fail the opposed-pad endpoint gate. | Search wrist translation and thumb/index flexion together around the measured wire while retaining non-pad clearances. Only a qualifying endpoint should advance to continuous route and opposing-force tests. |
| Tripod | 69 measured first-loss events yielded 64 cases with left thumb, middle finger, right-hand and support load. No single 480 Hz step had enough authority. At the existing control-window rate, 35 of 64 endpoint checks recovered index-pad geometry in CPU replay. The SDK 1404 candidate used 23.283 µm of left-palm translation and a 1.5625 mrad world rotation vector. Its index gap moved from +10.398 to −1.212 µm, with 0.644 mm minimum non-contact rod clearance across 17 sampled poses and 22 shapes. | Compile the candidate into canonical intrinsic XYZ commands, prove continuous clearance, then measure the signed left index tracking response and force over a full controller window. |
| Power | From the SDK 6504 archived wire and 963-window bilateral target, both existing partial contacts passed. Fixed-wrist single-finger closure left all 18 tested roots ineligible. The left middle, left ring and right little cooking checks exceeded the penetration cap by at least 0.398, 0.521 and 0.425 mm. The left little and right index remained separated throughout the tested flexion. | Preserve the pending live partial-contact recipe. Refit wrist and wire-relative pose while retaining palm and thumb anchors. Verify palm plus five loaded fingers on both hands before testing release from the initial supports. |

## Lift and Hold

The tail audit uses the final 64 SDK records from each selected run (Pinch 369–432, Tripod 2129–2192 and Power 6441–6504), with heights measured from each same-Reset rod. Every inspected tail has zero simultaneous qualifying bilateral rows and zero off-support Lift rows. Both supports still carried load at each tail endpoint. Maximum off-support Lift is therefore null for these windows.

Antigravity re-read the installed-path sources and verified that the project already archived 17 synchronized native segment-quaternion endpoints at SDK 2–18 with no hand contact. Hold qualification still needs active-contact substep material position, a fixed pad-to-rod anchor association and a calibrated physical slip error bound. Human pose references provide none of these measurements.

The frozen Power launch passed read-only verification of all 2,207 inputs. No attempt was started. Applicable physical qualification requires real opposing reactions, stable holding with fixed wrists, clearance of both supports, at least 30 mm net rod Lift, a continuous 5 s Hold, and at most 0.5 mm measured slip.

## Wire generalization and OakInk

Use the measured wire axis and material stations to adapt the approach, then adjust closure from authenticated contact reaction and joint tracking. Keep the eight held-out wire specifications and 104 layouts per grasp separate from training and candidate selection. Report native outcomes and failure causes separately per grasp and per wire specification. A successful reference-wire replay alone does not establish generalization.

The fixed OakInk metadata and shape archives matched both official SHA-256 hashes on Ubuntu. A ZIP central-directory inventory identified 653 pen, 50 screwdriver and 43 toothbrush pose records, each from one object in that category, and 8,554 cylinder-bottle records from four objects. The archives can guide pose candidate selection. This batch did not decode pose payloads or train/retarget from them. [OakInk data documentation](https://github.com/oakink/OakInk/blob/main/docs/datasets.md)

The per-grasp reports and the round 2 readiness contract record the replay recipes, artifact hashes, CPU checks, and the remaining native gates.

Targeted integration verification passed 171 CPU tests with the real archived Pinch fixture enabled.

The post-registration scheduler suite passed 60 CPU tests. The expanded suite produced 863 passed, 22 failed and 1 setup error. A clean checkout of the starting commit reproduced exactly the same 23 affected cases. The old Tripod Reset profile seals the original worktree absolute paths, and an older parent receipt has matching bytes but its stored path resolves through a symlink rejected by the strict reader. These validations remain open. Both logs are identified by SHA-256 in readiness.
