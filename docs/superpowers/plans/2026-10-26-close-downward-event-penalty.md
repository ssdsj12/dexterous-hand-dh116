# Close completion downward force penalty

User confirmed an additional instantaneous -3 on close completion when summed hand-to-object vertical force is negative. Keep continuous downward penalty after close, and upward reward during close/lift.

Scope: current tensor measured-load close-direct-lift configuration. Add explicit --close-downward-penalty flag, valid only for skip-hold + continuous vertical rewards + downward_after_close. At a valid nonfailed phase1->3 transition physics sample with Fz<0, apply -3 exactly once. Fz=0 or positive does not pay; force magnitude and dt do not scale this event. No changes to stage readiness, actual lift guards, success or physical settings. Phase3 persists until reset, so event cannot repeat within the episode; no extra observation/state bit needed.

Add a separate reward term close_downward and checkpoint contract. Exact opt-in migration preserves network/Adam/counters, rejects reversal and unrelated changes. Validate boundary sample sign (not previous sample), repeated lift substeps, selected reset, invalid/failure/inactive isolation, CPU/CUDA, and dt independence. Resume latest completed checkpoint in a new single Isaac run after tests; verify new TensorBoard term and real update, preserving original records.

Additional user request: display force at close completion / lift admission. Capture only the same valid phase1->3 physical sample, before reset; aggregate events per rollout into separate lift_entry_force fields and TensorBoard lift_entry_force_N tags. Include event count, signed sum, split upward/downward and per-hand values. No-event force values are null, do not write false zero-force samples. Preserve all-step curves separately.
