# 902 contact source binding after Antigravity review

Baseline: cbaafb3cdb7c9e26814ddaafcf910729ec99f732. This is an independent-entry hardening patch, not a new native run or a grasp result.

## Independently confirmed changes
In scripts/screen_dh116_902_command_contact.py:
- locked_geometry authenticates the existing candidate SHA, its F4 lock and pose-reference SHA, and original parity/outer-exit/capture packets. The current left/right USD hashes must equal the hashes in the authenticated historical GPU capture result. Replacing both a manifest and USD cannot silently change the contact geometry.
- validate_locked_geometry checks both model-source maps and exact original F4 pad vertices. Matching forged copies inside a proof and contact object are insufficient.
- rod_snapshot_digest requires all nine finite 41-node snapshots. load_stage checks their complete digest, not just the last archive row.
- load_stage includes 21 original geometry/asset input files in its freeze inputs. dependency_files includes the pure-JSON readers, parity implementation and evidence lock. No USD/Kit import occurs before the native launcher owns it.
- Existing v1, controller, joint/mimic guards, threshold values and camera configurations remain unchanged.

The formal native launcher already reran prepare in a child and compared the entire proof before Kit. Thus this patch closes standalone loader/runtime gaps; it does not imply that the failed SDK476 experiment used a fake or unbound hull. Its 240 s soft wall was already inherited from the mandatory precontact branch. No budget was increased.

## Failure-first evidence and validation
Three initial regressions reproduced acceptance of a changed full-envelope digest, two matching forged hull copies, and changed model-source identities. A fourth check demonstrated the missing explicit captured-USD comparison. These red logs remain in private evidence.

Five new cases were added to the existing suite (25 command-contact tests total), including earlier-frame mutation, missing snapshots, NaN, matching forged hulls and manifest/USD replacement. The final related CPU regression passed 202 tests in 42.95 s, with zero failures:
8e7d8f6eda6cf4a3ae2a98b768a6a598d5258cddd84443f9a1050dbbe3004d43.

The Registry regression separately passed 60 tests in 16.04 s. Total for this package: 262 CPU tests passed, zero failures. Registry log SHA db5d6451fe75fd2ff7a0292ef35303e7c492097556ac6661b64276b6dbcea33f.

A fresh CPU process read actual server F4 packets: original vertices/model sources matched, a four-point replacement was rejected, and the clean import boundary passed. A full 528-substep ideal-command proof was recomputed in 84.899 s (exit 0), SHA c507b510c56c876231e01f5e27b0ea06ec777bea3169b4a7b1e0dbcf22b075aa. Another fresh process loaded this proof, produced a 66×24 action array, verified exact F4 identity and retained the clean import boundary. Its returned freeze inputs contain 66 entries, including geometry and source dependencies.

This CPU proof retains the explicit ideal-tracking scope. The actual native contact failure remains blocked; no new solver steps, Teacher queries, PPO updates or successful grips were produced by this patch.

## Antigravity discussion and independent decisions
The antigravity-desktop-review workflow used the visible codex-dh116-ssh-reviewer and repo_info/read_source on the live Ubuntu worktree.
- R1 identified the two standalone binding gaps and agreed with the measured source/cooked distinction and wrist-tracking diagnosis.
- Codex corrected the formal-launcher and wall-budget overstatements; R2 accepted both corrections.
- R3 read final source SHA 7f906c763685b8a2f24a750fc33f7394f10e29bc914ecea877a9a85bade40bda and tests SHA 88ea33585bd1a57ccea17c3dc29d195e53bbb16d2bc9c88a05150532a7e5a74c, matching independent SSH reads. It found no blocking defect.
- Two optional cleanups were noted: prepare's CPU-only returned file list is shorter than load_stage's authoritative native freeze inputs; load_stage reads the locked geometry twice. These do not change qualification and are retained in this bounded patch.

The suggestion to move the wrist first was also checked instead of applied blindly. Two additional static endpoint checks against the same archived envelope found:
1. Wrist at final pose while all fingers retain boundary commands: rejected (noncontact lower bound -1.737882 mm, selected-contact lower bound -4.032303 mm).
2. Wrist at final pose after opening the middle finger to its final target: static endpoint passes (noncontact +1.176584 mm, contact -0.130609 mm, support +1.894254 mm).

These are only endpoint checks, not a path certificate or new physics evidence. A staged route must include a separately checked middle-finger opening phase, bounded measured wrist settling and then fixed-wrist closure. It must not skip guards or assume a zero command means a stationary wrist.

## Preservation and next gate
528 protected files, three historical checkpoints and both fixed RGB camera configurations are checked unchanged. Native guard failure, exact receipt identity and model qualification remain recorded separately. Raw assets, trajectories and logs are excluded from Git; publication is through the Windows relay.

Next work: certify every command substep of a staged route, include measured wrist pose/velocity convergence and rod motion, then freeze a distinct bounded native attempt. Opposed real hand/rod contact and subsequent PhysX reaction transfer remain prerequisites to loaded hold and lift. Pinch, Tripod and Power remain independently unqualified under the unchanged 30 mm / 5 s / 0.5 mm standards.

## Follow-up: full staged command path, CPU only
A private bounded harness reused the current source-bound setup, original command_step and _measure_hand implementations. It independently changed only the diagnostic command schedule: 9 windows opening the right middle finger, 66 moving the right wrist, and 54 closing with fixed wrist targets. Original action sums and the 902 terminal target were preserved; every float32 waypoint correction remained within 1 microradian. The other hand's commands stayed zero.

All 1032 ideal command substeps (both hands, nine original rod snapshots, original source/cooked geometry, table and support guards) passed in 130.511 s. Final errors were 1.142e-8 m position, 7.208e-8 rad rotation and 2.682e-7 rad active joints. All three terminal SDF pad gaps were negative, with normal opposition -0.972/-0.951. These are geometry estimates, not measured contact forces.

This is NOT a dynamically robust route or a native execution recipe. The smallest contact reserve above the unchanged -0.15 mm limit is only 0.122272 micrometres; the smallest support reserve above the unchanged 1 mm floor is 0.245109 micrometres. Those margins are far below the actual wrist/link tracking error previously observed. Thus a passing ideal path cannot justify another native attempt as-is. No settling steps were simulated; no frozen v3 recipe or expanded native budget was created.

Evidence is retained privately under ref902/command-v3-staged-path-screen.{py,json,log}. JSON SHA c2c24c228d1ba6e5956d0408b11a31ab95aadd6b6fd7811028485d7fc4b790a2; harness SHA ca1fd2976f8120ed1cac5466849853e025cc2875f7c891d2abc0125e16605ddb; log SHA 51611a4473aab0e8a6f6a41cdd00566a79f0cf8d5569d66b2bb9cddd3388591b. Run command used the existing Ubuntu CPU environment and /mnt/mydisk/xk/envs/dh116/bin/python with that private harness path. This diagnostic is separate from the 262 regression tests.

Next: increase intermediate path reserve without altering the final 902 grip, qualify measured wrist position/velocity convergence before closure, and bind every new phase to the existing guard and action receipts. Do not run the current thin-margin path merely because ideal geometry passed.

## Subsequent bound refinement
The 0.122/0.245 micrometre numbers above are coarse lower-certificate reserves, not exact measured clearances. The follow-up [tracking-reserve report](2026-11-21-teacher-902-tracking-reserve.md) tightens seven archived states while preserving the original proof and failure evidence. Support853 refines to3.760575 mm; selected rod reserves remain18–68 micrometres. Neither the old proof nor this selected refinement qualifies native tracking.
