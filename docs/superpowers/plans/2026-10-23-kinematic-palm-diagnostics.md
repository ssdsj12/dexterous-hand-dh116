# Moving-palm physics diagnosis

The user requested continued PPO learning, with intervention only when control
becomes unstable. V29 stopped at 33,996 cumulative transitions and v30 stopped
after 612 new transitions on the existing 0.02 rad mimic guard. These are actual
physical stops, not interventions for low reward or ordinary unsuccessful grasps.

The old backend moves a fixed-base articulation with root-pose writes. PhysX
5.6.1 assigns zero root velocity to fixed-base articulations during dynamics;
this does not represent the commanded moving palm's contact velocity. This is a
structural hypothesis. The controlled anchor replay below does not support it as
the main cause of the observed v30 mimic failure, so the anchor backend is not
being deployed.

- [x] Preserve checkpoints and reproduce both failure traces.
- [x] Reject 100 ms settling pauses that prevent palm lift; reject faster recovery
  that still trips the guard. A smaller step passes selected replays/evaluation
  but v30 random exploration disproves a complete fix.
- [x] Build a throwaway backend: floating hand, external kinematic anchor, fixed
  attachment excluded from the articulation. Move anchors with kinematic targets.
  Teleport hand state only during reset. Keep 480 Hz physics, 60 Hz control,
  200 inverse-second mimic frequency, gains, limits, rewards and 12D actions.
- [x] Replay the v30 failure serially in one Isaac instance; inspect actual palm
  velocity and attachment tracking as well as guard violations. Check complete
  palm travel, not merely absence of faults. Do not synthesize object support.
- [x] Reject deployment: after correcting prototype initialization, the anchor
  variant still triggers the unchanged mimic guard at control step152, compared
  with step153 in the original backend. No production backend change follows
  from this diagnostic.

Observed results on 2026-10-23:

- Initial manual kinematic Xform/Sphere probes jump to the origin on the first
  physics step despite correct authored USD poses; the SphereCfg-spawned object
  holds its initial pose but ignores that first requested displacement. A
  subsequent pose reset plus USD target moves all three correctly. This is a
  prototype initialization issue, not evidence about loaded grasp stability.
- The initialized anchor replay is recorded in
  `runs/tabletop_vector_replay_20261023_v30_anchor_warmup/summary.json`.
  It stops at152 control steps with env2 `mimic_error`, maximum reported error
  `0.0202644302 rad`, and exit2. The other three environments are interrupted,
  not completed safe episodes. Object bottoms remain approximately0.300m;
  no grasp or object-lift success is observed.
- A separate controlled candidate retains the original fixed-base backend,
  480Hz physics and60Hz control, changing only mimic natural frequency from200
  to400 inverse seconds with damping ratio1. The v30 failure-action replay in
  `runs/tabletop_vector_replay_20261023_v30_nf400/summary.json` replays the original
  failed action prefix, then extends with zero actions to901 control steps and
  four ordinary timeouts, exit0, with full palm travel. Its reported
  peak mimic error is `0.0184418335 rad`; this is a finite-trajectory result with
  limited margin below0.02rad. All four objects remain at about0.300m and success
  is0/4. The original guard runs at each physics substep, while the reported
  control-step peak does not quantify the maximum across all unlogged substeps.

Further evidence required before judging the400 inverse-second candidate:

- [ ] Replay the independent v29 failure and evaluate the saved v30 policy over
  eight episodes, retaining all physical guards and recording actual palm and
  object travel separately.
- [ ] If accepted for a new training run, permit only explicit200-to400
  weights-only migration under otherwise identical contracts, preserving source
  hashes/contract/counts and starting fresh optimizer/counters. The old
  checkpoint must not acquire a success qualification.
- [ ] Monitor actual stochastic PPO exploration and reset transitions. A new
  physical fault remains a counterexample to general stability; ordinary
  unsuccessful grasps remain learning data. Complete evaluations and replay
  traces do not establish that PPO has learned a grasp.

No changes to the hard guards, no blanket conversion of physical faults into
successful or safe episodes, and no deployment of diagnostic variants without
actual validation. No commits/uploads requested.
