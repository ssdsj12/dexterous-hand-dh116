# Event-bound contact context and Actor input

## Concrete problem and behavior

The prior no-step Reset2 has no measured contact;after separate source priming,the first policy starts atSDK3. A context still tied to Reset2 cannot be packed atSDK3. Sampling contact only at an8-step control endpoint also misses mid-window contact loss. Caller-supplied contact durations would hide this error.

PrimedContactContextOwner freezes the existing S0/dref at the actual Reset,leaves current unavailable,and commits the first current context only after the complete priming event SHA matches PrimedNativeActionHistory's recorded receipt. Each subsequent physical event must first pass the original history.record;the complete event SHA must equal its accepted row hash. Epoch,SDK+1,rod+12 and raw/accepted clocks must match before the original contact reducer runs. Missing,foreign,modified or replayed events poison the context and cannot backfill.

The owner computes persistence from original bilateral force/opposition evidence,with the first qualified endpoint at0s,then(tick-start)/480. Every actual SDK row is observed,so a middle-window loss resets the duration. Full per-substep coverage and the original0.3N hand-total threshold are retained. Wrong-side physical contact gives0 progress and preserved rejection evidence;identity/source corruption is a hard error.

The independent approach_contact_prelift_v1 context has a fixed Close stage,not a declaration that grasp succeeded. Fields4/5/6 are measured coverage/persistence.7/47,unlatched anchor stations and disabled deadline stay rawNaN/false with explicit reasons. Lift/Hold phases,caller-supplied physical_fields,phase injection and success events are rejected. Original30mm/5s/0.5mm thresholds remain in the frozen context.

TaskContextOwner's original Reset reference setup is extracted without changing smoke/full logic. Direct construction of the base class in contact mode is rejected;only the exact event-bound implementation may create this mode. Its close/failure guards also apply before commit_post_context,to prevent half-writing state after shutdown.

## Actual Actor/PPO connection and limits

NativeContextTeacher admits the new mode under native_contact_teacher_<grasp>_cpu_v1,with the3 new source module hashes included in its dependency identity. The existing6 inputs,50-field schema,network topology,Tanh distribution and original PPO are retained. Tests prove nonzero Actor input gradients through fields4/5/6 and zero gradients through excluded7/47,plus an original-PPO CPU update. Old smoke identity is rejected and unsupported Power is rejected by both model and source owner.

These are two distinct CPU checks:one uses the original physical-episode/history adapters with explicit CPU SDK/solver doubles;the other uses fresh synthetic CPU model packets. This package does not claim a native Collector has consumed these values. The actual NativeEpisodeOwner/source/reward/Collector gates remain unchanged until the contact reward and source adapter are complete. No Isaac/native training was launched and no successful Teacher trajectory is admitted.

Power still requires its own five-pad plus palm measurement and independent physical qualification. It cannot alias the two/three-pad reducer. Both fixed RGB cameras,assets,physical parameters,24D and371D contracts remain unchanged.

## Verification and review

Initial9 tests failed with the missing owner module. Model integration then failed on the old mode whitelist before minimal extension. A later failure reproduced direct-base construction bypass;it is now rejected. Antigravity independently read the actual implementation,agreed the timing and wrong-side semantics after Codex's challenge,and identified a close-before-commit half-write. Codex reproduced that defect in a failing test and added a pre-commit failure guard. Broad regression:273 passed,0 failed in472.31s. After the isolated4-line closed-commit override,all18 contact-owner/model tests passed in9.72s,including17 overlapping cases;the union is274 distinct passing cases,not291. The broad run loaded the previous contact-owner version;the final18 explicitly re-exercised all changed contact paths after the guard fix. Old mode source/behavior was unchanged by that last override. Exact commands and SHA evidence are in readiness;intermediate runs are not added.

Review was performed with antigravity-desktop-review in the visible SSH reviewer conversation,with branch/HEAD/file SHA independently checked by Codex. Private review transcripts and test logs remain outside Git. No review message substitutes for native qualification.

## Next executable package

Extend the existing TrainingReward with a separate contact-only profile,using distance/contact/opposition/persistence and the same owned per-event measurement/counter. Keep Lift/Hold/slip/success excluded and the original FULL anchor requirement unchanged. Then connect the real source adapter and existing Collector,bind initial reward scaling to Reset2 while the first endpoint is primed3,and test exactly-once settlement/GAE/update. Only after this coherent source closure is complete may a fresh frozen bounded native diagnostic provide new evidence. Do not repin or replay old spent recipes.

An offline AST comparison against baseline66d9c33d confirms all50 field records and both Context/Input schema hashes are exactly unchanged. This is a source-contract check,not native validation.
