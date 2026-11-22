# Conservative supported-contact source for Teacher

## Actual implementation

tabletop_native_teacher_contact_measurement.capture_supported_contact consumes the complete original17-slot DeformX solver packet. It calls measure_supported_distal_contact before reducing any values. All original mesh, binding, pair-capacity, overflow, same-clock, aggregate-force reconciliation and material-side checks remain.

Pinch retains its own two-pad denominator;Tripod retains three pads. Each required pad contributes coverage only when its normal load is strictly above0.05N in all12 solver substeps. Opposition is required in all12. Any wrong material-side contact invalidates contact progress. bilateral_support is taken from the original reducer, preserving the separate total-hand threshold strictly above0.3N; coverage1 and opposition alone cannot replace this test.

The result keeps original raw-rod SHA, identity/timestamp, complete evidence and per-substep external-support force norms. Table and both stands remain separate. Norms are accumulated per raw pair, so cancellation cannot hide contact. This module reports table_contact_failure but emits no reward/action/termination by itself. No success or qualification is granted. Power is rejected here;it needs the original five-pad/palm source.

## Tests and real-data replay

Six RED cases first failed because the source module was missing. Final related CPU regression:40 passed,0 failed in3.77s. Exact command is in readiness. New cases cover all-substep coverage,a missing last-substep pad,low total-hand load despite full coverage/opposition,wrong material side,table/support separation,invalid Reset contact and independent Pinch denominator. The partial-substep fixture rebuilds full solver capture and reconciled aggregate arrays;the validator is not mocked.

An offline replay of all64 original v11 native physics rows uses the same intact bytes and pinned SHA. Results:both required-pad coverages0, bilateral0/64, support-free0/64, table failure0/64. Every raw packet is separately hashed in the private replay summary. This is old native evidence replay,not new simulation,training,or proof of grasp.

## Remaining integration

The real session is not switched to this source yet. Next:build contact context after explicit physical priming,add independent contact/persistence reward without Lift/Hold/slip/success,wire the existing Collector and original PPO,and freeze a new source-bound finite native diagnostic. The original FULL slip-anchor gate and30mm/5s/0.5mm success contract stay intact. Formal BC/DAgger/Student admission stays blocked.

Antigravity R2 agreed the conservative physical criteria;Codex independently checked original code and actual buffers. No claim is made that Antigravity read this separately added implementation.528 protected files,3 old checkpoints,fixed stereo and physical parameters remain unchanged. Raw logs/data/models remain private.
