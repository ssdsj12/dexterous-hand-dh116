# Teacher reaction-force closure repair

Base: 56e7f16303e13e7e49eae75492e3ffbaef78d53b (11-17). Reuses the approved fixed-S0 v2 four-phase design; does not alter old Teacher or models.

## Evidence and limits
The 20261117T052535Z native cohort shows Pinch/Tripod wrist ascent without sustained bilateral loads. Locked SDF/actual SDK pose audit finds initial Pinch pad gaps 5.660--12.088 mm. Geometry predicts reachability only; closure is decided by measured simulated rod/pad reaction forces, not joint posture. Power needs actual palm/proximal contact sources, currently absent.

## Package 1: force source, phase and execution wiring
1. First fail tests for absent, one-sided, non-opposed, support-only and flickering normal contact.
2. Reuse the complete DeformX contact-pair packet reconciliation; separate pad, table and support loads. Both hands need their required normal-loaded, opposed pads in every solver substep.
3. Add independent N=1 diagnostic runtime, SHA-bound to frozen source. Before every 480Hz step, hold both wrist targets until force-qualified prelift contact persists. Preserve original finger commands and limits. After each solver step, record raw/issued actions, joints/targets, contact pairs, clocks and source identity. Contact loss closes the lift gate before the next step.
4. Preserve old source and default paths; no PPO optimization or expert admission in this adapter. Reuse PhaseOwnerV2 math but never promote its CPU receipt to native proof. Final slip and successful grasp remain unqualified.
5. Run CPU integration/regression, review, preserve protected files and models, report/readiness, Ubuntu commit and Windows publication.

## Package 2: acquisition and Power geometry
Use real per-side FK plus original cooked/SDF geometry for an independent reachable S0/approach candidate, table clearance and original joint checks. Do not repeatedly reset 1mm recovery budgets to cross the known initial gap. Derive force feedback from actual contact only. Audit palm/proximal collider sources for full-wrap Power; five pad forces cannot qualify it.

## Native boundary
No new Isaac run is authorized by this code package. Prepare frozen SHA, single-instance lock, exact selected style, solver/physical/control steps, ledger rows, time and exit verification before requesting the next bounded scope. Prior recording/render budgets cannot be reused for new contact experiments.

## Completion claims
CPU wiring is not measured contact success. The unchanged final criterion is net lift >=30mm, hold >=5s, fixed-anchor slip <=0.5mm, bilateral opposed contact and no external support. RGB/Student and Teacher/MPC successes stay separate.
