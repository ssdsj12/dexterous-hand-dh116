# Power supported partial contact handoff

The new opt-in worker starts from the original supported Isaac Reset, replays the existing Power opening, bypass and palm witness, and compares its live SDK 6504 exchange to the sealed field values. It checks actual positions, velocities, targets, link poses, contact pairs, World reactions and wrist projection. The archive is an equality reference. It is never loaded into the running simulator.

The worker binds the three fixed-rod artifacts, rechecks exactly joined waypoints, derives the normalized 24D increments and rejects original joint or wrist rate violations. Four of these stages eventually admit thumb or pad contact, while the first continuation stage admits only palm contacts. The left palm can move through nodes 31, 30 and 29 during the first stage. Later stages use only node 29 on that palm, with the right palm assigned to node 9. The final left thumb and index use 37 and 36, and the right thumb and ring use 4 and 3. A contact on an unexpected link or node stops the run and is persisted.

| Stage after SDK 6504 | Windows | Final SDK step |
| --- | ---: | ---: |
| Wrist reseat | 284 | 8776 |
| Thumb first | 202 | 10392 |
| Two-pad closure | 88 | 11096 |
| Right approach | 82 | 11752 |
| Left approach | 307 | 14208 |
| Fixed-target observation | 1 | 14216 |

The physics owner checks both the SDK-to-rod handoff geometry and the just-completed same-clock rod. For each continuation step it uses the measured link poses and checks the 34 original source envelopes, the ten locked cooked pad envelopes and the selected registered SDFs. The original table and hand-side guards apply. The continuation checks the same support clearance used in the fixed-rod certificate. The completed-rod geometry evidence can be consumed once. Each readback also checks that the composer received the prior solver row's World force and torque for the same owned body IDs. Source, cooked and SDF guard failures have dedicated evidence.

The terminal check uses the final two consecutive SDK exchanges. It requires opposing thumb and distal normals, all original minimum reaction loads on the selected links for the complete two-step sample, and actual body-side routing. Support loads are required because this is a *supported partial-contact diagnostic*. Whole-hand enclosure, off-support Lift, Hold, Slip, Teacher admission and transfer to different wires still require their own native results.

## CPU evidence

The offline freeze command used the original near-open, proximal, opening, bypass and palm-witness source arguments, replaced the entry point with `python -B -m scripts.verify_dh116_power_bilateral_handoff`, and added `--staged-receipt`, `--continuous-audit`, `--right-route-audit`, `--bilateral-detour-audit` and `--soft-wall-seconds 3600`. The complete command and hashes are retained in the work evidence directory. With the approval argument absent, the command wrote a plan and stopped before starting Isaac.

The CPU replay checked the actual historical SDK 6504 anchor and loaded all ten cooked envelopes and 12 signed-distance grids. It also ran the first continuation stage's cooked and SDF guards against that same archived pose and rod. This is a geometry replay. The median cost for those guards over 20 repeat queries was about 3.1 ms. It excludes the 34 source hull checks, SDK and GPU calls, file I/O, and tighter geometry encountered later in the route. The old palm-witness run took about 1163 seconds. The new 3600-second soft limit remains a proposed resource bound.

The draft execution budget is GPU 2, one instance, one environment, two initialization SDK ticks plus at most 14,214 measured ticks, 1747 control windows and 170,568 solver substeps. There are no model loads, policy queries, PPO updates, automatic retries, or automatic kills. A naturally closed, hashed outer receipt for the new worker remains required. Antigravity verified the live-clock, archive and failure handling, and Codex repeated the source SHA and CPU checks independently.

The coupled CPU suite had 263 passing tests before the final rejection-ledger refinement. The final focused Power suite passed 72 tests after that refinement. The first freeze rejected a continuous-ownership requirement that exceeded the fixed-rod artifact’s sampled SDF claim. The corrected freeze checks the sampled evidence and still enforces material ownership on every completed live exchange.

After publication, rebuilding the same plan initially changed the fingerprint because the shared freezer included the Registry, Release Manifest and the handoff’s own readiness record. The Power entry now excludes precisely those three output records. The archived proof contract, scene and runtime file hashes remain pinned. The corrected candidate SHA is `6e3c66dcb2fecb69e5c5397865cb6f9c2874846e76e26fc1276a4e96edb7c333`. The final focused regression passed 73 tests.
