# Loaded palm motion validation and repair

User approved validating and correcting downward dragging before reward edits.
Use the paused v155 checkpoint, preserve its policy/optimizer/counters, and run
only one Isaac instance. No reward, torque, material or hard-guard changes.

Hypothesis: downward palm compensation during loaded close contributes to
downward object friction. Existing v153 trace supports association, not causality.
Compare identical coordinated control with and without a measured release ->
reposition -> reclose controller. User clarified that downward repositioning is
allowed: do not ban downward motion after contact. During close only, if a
loaded hand (>0.1N) requests >0.3mm downward displacement, snapshot its target,
hold the palm and release the thumb and opposing index. Require <=0.05N object load for
0.05s before moving. Once measured load falls to0.05N, hold all finger commands rather than continuing to open. Regained load >0.05N stops
motion and returns to release. Project the released movement goal onto the existing2.1mm table-clearance margin and record any clipping. At target within0.2mm for0.05s, reclose with the
normal controller. Release/move timeout2s cancels movement and recloses at the
current position. Reclose settles for0.05s of contact, with1s timeout. At most2
attempts per hand/episode; selected resets clear all state. Block close-to-lift
while adjustment is active. No release sequence is initiated during lift.
Position/table/velocity/force/mimic hard guards and joint slew remain unchanged.
Unloaded downward positioning remains allowed. This is a command sequence, not
an assertion of contact-point velocity or successful grasp.

Alternatives: cancel all compensation (confounds X/Y), or change rewards
(does not isolate controller). Use the one-axis loaded-motion ablation first.

- [ ] Preserve source, active metadata and verified checkpoint after shutdown.
- [ ] Add failing tests for per-hand release/move/reclose, mask, inactive rows, selected
  reset and preserved unloaded positioning; integrate with coordinated control.
- [ ] Add8-env inference-only paired probe: fixed4N, fixed8N, PPO mean twice,
  each baseline versus candidate. Same checkpoint, physics, reward, start and budget.
- [ ] Record commanded and actual palm motion, signed normal/friction force,
  contact load, link-center velocities (not contact-point velocities), lift
  readiness, measured height and physical failures. Evaluate force direction
  together with retained grip; loss of contact alone is not improvement.
- [ ] If supported, qualify exact one-way contract migration and production
  enablement, review, regressions, then resume paused policy in a new run.
  If disproved, retain baseline and investigate measured contact geometry.

Initial v156 evidence: all8 episodes completed, no fatal guard;4 candidates reduced downward impulse but ALL adjustments timed out. Left index-only release left0.56N thumb load; right hand unloaded but continuing opening reduced table clearance to1.1mm, blocking the move. Candidate was rejected for deployment. Revision opens thumb/index together, holds once unloaded, projects the goal without relaxing table guards; swapped-slot v157 validates this revision.

Final validation: v158 8-env swapped probe completed all requested episodes without hard guard failure. Fixed 4N candidate first6s downward force averaged -1.53N versus -1.56N baseline; fixed 8N candidate -0.74N versus -2.29N baseline in this paired slot set. Candidates executed release events, completed move states and reclose events; no candidate reached stable3mm/30mm success. PPO v159 resumed at iteration99 with release_reposition contract, monitor reached iteration101 and reported nonzero release/move/reclose fractions, no failure/safety stop.
