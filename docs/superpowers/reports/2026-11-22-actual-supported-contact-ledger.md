# Actual Teacher contact ledger: evidence before phase integration

Baseline:f49488c3c119db40b7359461c621bcc639a732e7. Antigravity R3/R4 read the current relevant implementation through the readonly SSH role;Codex independently matched its source SHAs and corrected the baseline/qualification assumptions. Antigravity explicitly did not read external raw files;the measurements below were performed by Codex.

## Executable change
scripts/audit_dh116_teacher_contact_ledger.py consumes an existing raw physics JSONL exactly once while hashing the consumed bytes. It checks strictly increasing raw IDs,consecutive SDK/12-substep rod ticks and same-episode action pre/post identities. It calls the existing complete17-slot measure_supported_distal_contact reducer without filtering pairs or weakening its load/opposition/source/reconciliation guards.

Every output row retains per-hand/per-pad/per-solver-substep contact evidence,table/two-support force norms,41 world positions/velocities,all measured q/qd with original per-joint units,and actual submitted24D command and source contract. Missing joint/velocity sources remain unknown,never zeros. Submitted normalized commands are explicitly distinguished from actor latent or measured tracking. CLI creates exclusive output and flush/fsync;it never overwrites raw evidence or opens Isaac.

## Real offline replay
Input:the64 original physics rows of the v11 bounded native A,SDK3..66. Input SHAaf8780644423d297a8e5a1b43d54045ec0688276a78f180d825625c3a2efc688. This is replay of existing native data,not new collection or training.

| Measurement | Actual result |
|---|---:|
| PhysX/rod action-clock rows validated |64|
| Bilateral required3-pad contact rows |0|
| Left required3-pad maximum summed normal load |0 N|
| Right required3-pad maximum summed normal load |0 N|
| Completely support-free rows |0|
| Table contact rows |0|
| support_0 maximum per-substep force norm |0.09229257033610057 N|
| support_1 maximum per-substep force norm |0.09187661241154617 N|
| Maximum recorded node speed |0.07183308228640962 m/s|
| Measured q/qd complete for both hands |64/64|
| First-to-last median height difference |-0.005425065755844116 m|

No off-support baseline exists in this data. The node speed is an observed maximum,not a newly chosen task threshold. These observations establish no grasp;they do not establish a causal policy failure from8 controls alone. Original A remains an unqualified wiring diagnostic. The actor did issue commands and the measured joint positions changed,but required pad contact never formed. Forced interpolation or lifting before contact is not a repair.

Output ledger SHA:d8d3837344afc7d6937ed2d9f7628c4b1adf9ec9bfc176951ac56c178b1962fa. Raw/output files remain private on the data disk. Native runs0,new PPO updates0.

## Verification and failure evidence
Five new tests failed before the auditor existed,then passed. Selected old/new CPU regression:130 passed,0 failed,19.51s. Tests cover full real reducer routing,unhidden support,unknown q/qd,wrong raw SHA,cross-clock action,overflow,Power alias refusal and repeated physical ticks. One first command used a nonexistent test filename,exit4 with0 tests;the corrected command is preserved with the successful log. Exact SHA and command inventory are in readiness.

## Accepted plan and boundaries
R4 re-read the published route repair and found no defect. Codex rejected the R3 proposal to freeze height at entry to Lift:net Lift must start only after same-clock force AND geometric clearance confirm departure from both stands and table,with real contact still valid. Preserve initial support load records. Future recontact must remain visible and disqualify support-free holding.

Reuse existing contact reducers,support-segment geometry,baseline field format and v2 phase/slip owners. PhaseOwnerV2 remains cpu_math_fixture;the old hook's2mm/s and1mm/s values are diagnostic examples,not an authorized new physical contract. Fixed anchors,source-validity and original30mm/5s/.5mm remain necessary. R2,ForceProxy,full reward and formal successful BC/DAgger stay closed. Power requires five digits plus palm;it cannot borrow Tripod qualification.

Next:freeze and verify a fresh17 Reset under repaired source,then connect measured supported contact and phase fields without modifying old R0 or pretending an interface is full qualification. Fixed two RGB,528 protected files,3 original checkpoints and user changes are preserved.
