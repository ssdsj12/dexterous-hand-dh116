# Tabletop PPO training readiness

The user approved a rigid cylinder with two DH116 hands, 12 joint residuals,
scripted palm movement, independent exploration admission, and an approach,
close/hold, lift curriculum with final full-sequence evaluation. This document
records that approved scope; it does not claim physical grasp success.

## Control and evidence

An actual wire contact with at least 2 mm table clearance can finish approach
at its current root height. A clearance-only stop is an explicit blocked task,
never grasp success. Original four-pad load, continuous hold, slip, force,
mimic, joint, and whole-object lift success limits remain unchanged.

Nominal joint references evolve independently from PPO residual commands.
Residual targets have a configurable radius, initially 0.02 rad and never above
0.1 rad, from the nominal reference and joint limits. Executed commands retain
the 0.3 rad/s limit. Residual changes are separately limited to 0.05 rad/s;
the 12 filter states are included in the versioned 94D observation. Hold/lift freeze the
nominal reference. Observations include reference, task timers, clearance,
relative object pose/velocity and grip reference. No synthetic successful
contacts or object attachment are permitted.

## Training lifecycle

Each curriculum episode starts with a real scene reset. Approach ends upon
entering close; close ends upon entering lift after the original hold checks;
lift/full uses the original full grasp success. Stage completion is distinct
from grasp success. Later stages retain the learned policy through all earlier
phases. Promotion requires multiple evaluation episodes, not a fabricated
advanced-stage reset.

Before learning a stage, run zero-residual and bounded random-action probes.
Require finite observations, repeatable resets, no physical safety violations,
and observed access to that stage in at least two probes. The zero baseline may
time out without reaching the stage; random probes retain the policy in preceding
phases and randomize only the stage being admitted (close includes hold).
Full-stage admission requires lift access. Failure to grasp may be used for learning;
physical instability may not be bypassed. Persist admission evidence.

Progress-based rewards replace positive idle approach/tracking rewards in the
PPO entry point. Existing standalone reward users keep their old interface.
Use gamma=0.995 (documented tuning starting point, not proven optimal).
Periodic checkpoints, explicit resume/model evaluation, checkpoint contract
checks, machine-readable training logs and multi-episode evaluation are needed.
An interrupted run must preserve its latest checkpoint and truthful status.

## Validation boundaries

Pure regression tests cover the observed 10.08 mm contact deadlock, residual
drift, stage completion vs grasp success, hidden-state observations, admission,
curriculum promotion, and PPO save/load. Simulator tests use one Isaac process,
one CPU core and one numerical thread. A synthetic backend test proves software
wiring only. A real short training run, if admitted, proves updates only; it
does not prove learned lift, deformable-wire handling, or hardware readiness.

## Evidence-driven revision

The initial 0.1 rad probe passed once but a later resumed close probe hit the
unchanged 0.02 rad mimic limit. At 0.02 rad a fresh short run passed, while a
resumed stochastic trajectory again exceeded that limit during the approach/close
handoff. Those failures are preserved. Residual rate limiting is a subsequent
control-bandwidth hypothesis, not a claim that the PhysX root cause is proven.
Its new observation/contract rejects old 82D checkpoints instead of silently
reusing an incompatible policy. Runtime safety stops remain active after admission.
