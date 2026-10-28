# Vertical contact force telemetry and fresh PPO

User scope: compute total upward force from both hands, then restart PPO from zero. Preserve old runs. No reward or lift-gate change requested.

Implementation: enable Isaac Lab filtered friction reporting for the existing 34 hand-link sensors, each filtered to Wire. Negate normal+friction forces measured on hands, sum signed world-Z components per hand. Weight uses runtime body mass and downward gravity. Aggregate only live physics substeps before terminal resets; publish rollout means, instantaneous peak and fraction above weight separately. No new task observation or reward. Contact data capacity16 per body per env; native friction reduction currently contains CPU synchronization, throughput must be measured.

Validation: three CPU tests cover signed cancellation, ignored horizontal forces, inactive/reset samples, peak-vs-mean and invalid measurements. Existing reset test verifies selected vertical buffers are cleared without altering others. 89 related tests pass. Independent reviewer38 tests passed and injected terminal-step/reset case passed. GPU wall/sliding-block fixture at480Hz measures vertical friction.4000002N; momentum-derived.3999784N, maximum discrepancy6.23e-5N, reciprocal forces identical. Fixture result persisted before Isaac close hung; terminated completed fixture, not a physics failure.

Fresh run: v122 stopped at554 with old checkpoints preserved. v123 launches4096env,24steps,1024episodes/env without resume or initialize-from. training_start.json audits iteration0/steps0/episodes0/emptyAdam. Verified first2updates/196608samples, finite policy/Adam. First rollout18.39s, second signed force mean−.8631N and peak+.5130N vsweight1.1772N; aboveweight fraction0, success0. Background watcher records vertical_contact_force.

Limits: telemetry is completed-rollout statistics, not lifetime or stable lift proof. A spike above weight is insufficient for lift success. Exceptional partial-rollout vertical telemetry is not published yet. Existing task rewards, geometry, friction, lift gates and protections remain unchanged.
