# Actual Teacher resumed worker and separate evidence audit

The B worker now uses the original owned simulation lifecycle and a separately verified fresh Reset. The trainer restores the authorized checkpoint after the live source pack is bound and before the first collector policy epoch. Only A seeds a new exploration stream. B carries over the checkpoint's random state and runs one additional update.

The evidence derivation and post-exit audit retain the original ledger and share the original physical clock, action, rod-state and source-pack checks. B has separate result and seal identities. Its counters must move from update 1 to 2, the first policy weights must equal A, the accepted minibatch count must be cumulative, and the Reset must have a different episode UID. The existing A auditor still rejects resumed results. Worker authorization and archived parent/child provenance remain a separate scheduler step before any B launch.

The Ubuntu CPU regression passed 169 tests. It includes the actual factory ordering, mismatched role and origin, altered RNG, incorrect initial policy epoch, duplicate episode, changed minibatch count and raw ledger retention. No Isaac session or native PPO ran.

[Readiness](../../contracts/dh116_teacher_actual_resume_provider_v1.json).
