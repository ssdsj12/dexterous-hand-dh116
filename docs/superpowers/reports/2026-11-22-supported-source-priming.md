# Supported source priming before contact PPO

## Problem and implementation

The no-step Episode Reset remains SDK2/rod_tick0 and has no measured contact buffer. Neither a commanded zero wrench nor a fake on-policy token can provide contact evidence. Directly stepping the exchange would leave action history at Reset2 and reject the first policy's SDK3 boundary.

The new PrimedNativeActionHistory preserves the actual Reset and S0, validates one separately recorded zero-command physical exchange through the original _Sequence and _physical_source validators, and advances only its physical clock to SDK3/rod_tick12. The original _Sequence first boundary defaults to reset; only this new owner selects post_physics after verified priming. The first actual sampled policy window then runs3→11. Priming produces zero PPO transitions, no sampled decision and no reward window.

prime_supported_sources checks the existing17-mesh physical episode, same scene/budget identity, and enough budget for priming plus a full first8-step policy window. It charges the actual exchange before recording/validation, fsyncs a separate journal row, and updates the episode packet only on full completion. Any step/record/write/check failure poisons both episode and history. If an SDK operation advances then raises, observable SDK/rod progress is charged and solver completion remains unverified. Failed transactions cannot replay.

The current actual session still uses the original owner. This component is deliberately not selected by a default/native config before contact context, reward and Collector are coherently connected. No live integration or grasp success is claimed.

## Independent review and verification

The antigravity-desktop-review skill was used in the visible SSH reviewer conversation. R2 confirmed the Reset/history/context mismatch and the versioned solution after Codex challenged direct-step suggestions. R3 honestly reported that new untracked files were unreadable; Codex added intent-to-add for final R4 source review. R4 read the whole implementation and accepted its boundaries. Its additional partial-solver-progress suggestion was independently reproduced by a new failing CPU case and fixed: all observed ticks are charged and a partially consumed call counts against the call budget, while completion stays unverified. Review evidence is stored privately. No chat or raw experiment logs are uploaded.

TDD:5 initial missing-module failures, then5 passed. Additional failure tests exposed a real history-poison gap when journal writing fails before dispatch; fixed. An intermediate exception-expectation mismatch was corrected in the test. A command containing a nonexistent test path collected zero tests and is not counted.

Final related regression:136 passed,0 failed in84.44s across7 files, including12 new priming cases. Tests use the actual existing episode/controller/history/PPO classes with CPU SDK/solver doubles. They cover durable failure, progressed SDK exceptions, receipt mutation, no duplicate priming, original total budget and first on-policy transition. They are not native contact evidence.

Two initial regression failures were caused by an old Tripod source test fixture pinned to the physical exchange before the canonical-route repair. The fixture now selects the sealed, real fresh17 Reset v3. Its original strict manifest loader still re-derives and verifies source bytes; no receipt/SHA is repinned and no qualification is raised.

Metadata/Registry regression:73 passed,0 failed, with13 tests overlapping the primary136. Exact commands, log hashes and failed attempts are retained in readiness. New native runs0; Teacher updates0.

## Next executable integration

1. Add a pending Reset2→primed SDK3 contact context; retain S0 and unknown/NA fields.
2. Feed intact17-slot per-substep contact data through the existing reducer. Coverage must survive all12 substeps; persistence must use the original bilateral load/opposition predicate.
3. Add independent approach_contact_prelift_v1 reward/config/model namespace with only distance, contact, opposition and contact persistence. Lift/Hold/slip/success excluded; original FULL remains strict.
4. Wire the actual C04 Collector/session and CPU-test sampled actions, real input packing and exactly-once reward/update.
5. Freeze new source closure and run a distinct bounded native diagnostic. Historical Reset evidence cannot certify the changed contact-learning chain.

Pinch, Tripod and five-finger/palm Power remain separate qualifications. Fixed stereo, physical parameters, original24D/371D,528 protected files and3 old models are retained.30mm/5s/0.5mm remains the final success contract.
