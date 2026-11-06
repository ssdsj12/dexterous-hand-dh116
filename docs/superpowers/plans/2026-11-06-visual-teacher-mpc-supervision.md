# Teacher reference and MPC supervision for the visual Student

User-approved scope: use the successful historical Teacher as the action reference,
optimize training labels with MPC, and let the visual Student learn those actions.
Keep the existing Tanh Gaussian PPO, sensor-only 3844D actor, privileged 371D critic,
canonical 24D commands, physical parameters and native success criteria unchanged.

## Data and control design

1. Replay the verified 371D Teacher corpus's canonical 24D action trajectories in
   ordinary native TensorEnv. Capture real, same-tick pre-action RGB-D and build the
   existing four-frame sensor history. Historical privileged observations are not
   visual actor inputs. A historical trajectory is a reference, not an oracle for
   arbitrary off-trajectory Student states.
2. An independent training-only label optimizer uses the measured current native
   state and the existing multi-step local finger/pad MPC. It may refine twelve
   finger commands while the twelve wrist commands retain the Teacher reference.
   Preserve its activation and fallback gates. This is local kinematic prediction,
   not full object/friction dynamics or a demonstrated 24D joint optimizer.
3. Record reference, actual issued action, state/frame/time identity, optimizer
   activity and fallback. Stream raw RGB-D in bounded chunks. Admit only first,
   unrecovered episodes that independently meet ordinary native success into the
   successful supervision pool. Failed experiments remain separate evidence.
4. Fit a separate actor-only BC artifact in bounded `tanh(mu)` action space using
   an independent Adam. Explicitly derive a new v3 warm-start; preserve the original
   critic, log_std, PPO Adam, counters, RNG, LR and all migration provenance.
   Retain new supervision provenance across subsequent PPO saves/resumes.
5. PPO rollout remains unchanged: execute its own sampled actions and corresponding
   log probabilities. Teacher/MPC transitions never enter on-policy storage.

## Verification and delivery

- Red/green CPU checks for frame capture, source and output contracts, alignment,
  episode qualification, split leakage, MPC fallback/state identity and actor-only
  optimization. CPU fixtures are not counted as native experiments.
- One Isaac process at a time; independent bounded Teacher baseline then Teacher+
  MPC collection with the same reset, camera, seed and physics. No long PPO run.
- Only if qualified data exists, run a bounded BC fit and separate teacher-free
  Student inference. Report fit error, MPC activity and actual native success as
  distinct results. With only one original trajectory per grasp style, do not claim
  an independent per-style heldout validation set.
- Review, compare the full CPU baseline, update README/CHANGELOG/validation report,
  and publish to the existing repair branch with PR #1 remaining Draft.
