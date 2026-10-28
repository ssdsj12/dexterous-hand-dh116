# Force reward starts after close

User superseded the proposed 0.5s positioning grace: score vertical force only after close completes. Interpret completion as first entry into hold. The close-to-hold transition substep is still close; following substep starts force scoring. Latch scoring for the remainder of the episode, including a later fallback to close. Only selected episode reset clears the latch. Entire physical force telemetry and safety remain active from reset. Existing upward continuous rate, downward linear rate, actual lift/hold rewards unchanged.

1. Add failing phase-boundary, sign, fallback, reset, terminal and CUDA tests.
2. Optional --force-reward-start after_close (historical default episode_start) requires continuous signed reward. Append scoring-enabled observation bit (151 dimensions) to make latch visible. Include contract version and monotonic latch semantics.
3. Exactly migrate current continuous contract: preserve 150 old inputs and Adam columns, add zero-initialized 151st column and moments. Reject unrelated changes/reverse migration. Keep policy and training counters; no fresh reset requested.
4. Verify tests and independent review, restart one Isaac instance from retained checkpoint with new flag, verify migration and actual close zero force rewards followed by after-close force scoring. Update TensorBoard gate metric so zeros before hold are distinguishable from zero physical force.
