# Pinch native fixed-wrist force feedback

One new native trial completed all 55 control windows with 442 SDK steps and 5,280 DeformX substeps. GDB and the outer process exited zero naturally in 401.387 seconds. The first 45 actions and the SDK 362 state exactly matched the archived zero-action comparator in actual joint q, qd, backend targets and 41-node positions and velocities.

The live force feedback selected only left finger21 twice, 0.00005 rad each, at SDK 362 and 370. Both interval geometry checks passed. It issued zero increments for the remaining eight windows. Seven of those decisions had insufficient continuous opposition, and one had sufficient left normal load. The native control ledger replay regenerated all ten decisions. The capture was run without PPO and used no wrist lift actions.

| SDK 379–442 metric | Matched zero action | Measured force feedback |
| --- | ---: | ---: |
| Simultaneous signed opposition | 733/768 | 729/768 |
| All original load and opposition gates | 197/768 | 293/768 |
| Longest consecutive passing sequence | 14 | 17 |
| Table contact pairs | 29,613 | 29,483 |

This matches the earlier fixed two-increment trajectory. The measured selector reproduced that intervention and held its action when opposition lapsed. It has yet to establish a stable bilateral grasp. The longest passing section is 2.951 ms and table load remains. Before lift, the next controller needs a measured way to recover contact direction or reposition, while retaining the original support, slip and hold gates.

Frozen recipe SHA-256 `1f0579c935766da86f3e42ebbb9827aaa74c79dd3065a58485c9ee1ffd982a3b`, outer receipt SHA-256 `79498873c2211b8722338bc1be20292c74a496dd6918ea593bdea8530ef0d63d`. The initial post-publication CPU launch rejected an independent Tripod file changing during the broad snapshot and never entered Isaac. The accepted run was frozen from the subsequent clean published HEAD and used the same requested budget.
