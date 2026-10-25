# Adaptive grasp PPO implementation plan

**Goal:** Let PPO adjust finger closure, each palm's grasp point, and grip stiffness while learning actual object pickup.

**Authorization:** The user explicitly requested “让ppo可以这样调” after requesting diameter-level contacts, increased grip and small position corrections when lifting fails. This supersedes the earlier finger-only/controller-palm split.

**Object update:** The user then authorized replacing the object with a hollow straw. Use a rigid open tube, 16 mm outer diameter, 0.5 mm wall, 0.5 m length; declared simulation density 900 kg/m³ gives about 10.96 g. Keep the outer grasp envelope; create 32 separate convex annular collision segments so the bore remains open. Record asset SHA, dimensions and mass in the new checkpoint contract. Success on this object does not qualify the former 120 g solid cylinder.

**Architecture:** Add an opt-in 20D policy: 12 joint targets, 6 palm translation targets, 2 active-drive stiffness targets. Preserve the existing 12D policy contract. A new geometry initialization is clearly labelled as authored, not a measured simulator checkpoint. Start fresh because both observation and action dimensions change.

**Constraints:** One Isaac process, CPU0 and one numerical thread. No four-pad success gate. Never assign object pose except reset. Keep mimic 0.02 rad, force 30 N, table and drop guards. Existing checkpoint remains available. Do not commit or upload runs.

- [x] Environment: validate 20D actions; route palm/stiffness commands every physics step; expose 125D observations including actual pad positions relative to the object, absolute bottom height, palm offsets and applied stiffness. Preserve existing object-height/stability reward and logging.
- [x] Backend: palm offsets bounded to XYZ ±3/3/4 mm, rate 2 mm/s, downward clearance margin 2.1 mm; stiffness 10–20 Nm/rad, rate 10/s, reset 10; damping .2, effort cap .20 Nm. Disable scripted palm correction in adaptive mode. Synchronize PhysX and actuator gain state.
- [x] Initialization and contract: validate explicit geometry/source hashes, coupled joint state, roots and wire pose; hash initializer and all new actuator/control semantics. Refuse old 12D resume or implicit weight transfer.
- [x] Hollow asset: real open collision bore, explicit mass/inertia, asset hash, preserved outer dimensions and separate success provenance.
- [x] Tests: invalid-action finite termination, control/observation routing, frozen and bounded palm motion, stiffness slew/reset, strict initialization validation, old-mode regression. Final related suite: 484 passed.
- [x] Single-instance Isaac: v33 compared real pad/world geometry and initial clearance; repeatable reset/isolation and all12 bounded exploration admission episodes passed; fresh four-environment PPO started (first audit1388 transitions/3epochs). Live commands alter positions and stiffness. This establishes training launch, not successful pickup or final budget completion. GUI demonstration, when requested after learning, remains live Isaac rather than video.

Reward remains `object_support_progress_v3`: bounded contact progress for learning, actual height gain and lifted dwell, real lift/hold success, slip/excess-force/time/smoothness costs. Stiffness or commanded force itself earns no reward. Increasing stiffness is a response-gain control, not a claimed direct Newton-force actuator.
