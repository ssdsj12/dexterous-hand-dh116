# Tabletop contact recovery implementation plan

Goal: make close exploration physically reachable and remove the observed approach tracking overload, then run staged PPO autonomously with original physical acceptance gates.

Scope remains the user-approved rigid proxy, scripted palms, and 12 finger residuals. The user explicitly requested diagnosis and fixes after monitored v8 failed; no additional design approval is needed for these corrections. No source assets, friction, effort limits, grasp thresholds, or physical failure thresholds are relaxed.

Evidence: v8 zero probe at first contact (1.50 s) has left index reference 0.55158 rad, command 0.42036 rad, measured 0.41436 rad. At 3.35 s tracking has recovered but all four loads are zero and root remains 10.798 mm above target. v8 episode 17 has growing thumb mimic deviation before the hard 0.02 rad stop. A warning pause is a hypothesis to test, not an established solver fix.

- [x] Add failing integration tests for close contact reacquisition, approach tracking pauses, and joint-coupling warning pauses.
- [x] Implement bounded, clearance-limited reacquisition and tracking-aware palm motion; expose coupling state and version checkpoint contract.
- [x] Run focused software tests, then serial real Isaac zero/random probes to distinguish controller reachability from policy skill.
- [ ] Based on measured results, train with saved checkpoints and stage evaluation; never claim lift without measured gates. Preserve failures and document limitations.

Alternatives considered: only longer PPO leaves the permanent palm latch unchanged; changing solver iterations was already unsuccessful in earlier experiments. Use measured motion feedback first. A pause must not bypass the hard guard; a persistent deviation still terminates the episode.


Evidence refinement: v10 measured cylinder translation ~15 mm; prepared geometry
and object-relative XY tracking were added, then reviewed to gate *all* palm
axes on pause/readiness. v15 safely trained/evaluated; v16 reached 81,920 samples
but final policy failed approach, motivating an earlier 5 mm pregrasp handoff
with unchanged grasp/physical safety thresholds. Explicit weights-only transfer
preserves learned parameters without importing optimizer state or qualifications.

Additional deliverable: serial automatic continuation after safe unmastered
budget exits. Subprocess regression tests cover automatic resume, stopping on
physical failure, truthful budget exhaustion and refusing runtime source changes.
Do not automatically retry physical failures or launch parallel Isaac instances.
