# Shared-physics multi-environment PPO implementation plan

User requested multiple environments, autonomous PPO training and live monitoring.
Use 4 spatially separated scenes in one CPU Isaac instance, single numerical
thread/CPU0. Preserve direct finger action, object-based reward, and hard guards.
No additional confirmation needed for this requested implementation.

- [x] Refactor TabletopEnv physical stepping into resumable substeps while single
  environment behavior remains identical; SB3 VecEnv issues all commands before
  exactly one global physics step. Test early termination/reset isolation.
- [x] Build independent scene backends with unique prim paths/sensor filters and
  2m origins. Only world coordinator advances physics; individual resets perform
  no physics steps. Set reset_settle_steps=0 in new contract and start fresh PPO.
- [x] Vector admission, per-environment independent episode accounting and final
  observations, checkpoint/resume, evaluation and atomic monitor reports.
- [x] Serial real Isaac isolation probes (2 env then4), reset repeated check,
  bounded exploration and short actual PPO; investigate any measured failures.
- [x] Launch longer training after checks and start monitoring progress/failures.
- [x] Validate a480Hz candidate against the measured v29 failure and the saved
  policy before launching further training. Reject stability achieved by
  preventing the lift attempt; long-run stability is still unproven.
- [x] Investigate the new v30 training failure;480Hz did not prevent it. Reject
  the initialized external-anchor prototype after it reproduces the mimic fault.
- [x] Replay v29/v30 failed action prefixes under400s^-1 mimic compliance,
  followed by zero-action extensions; verify full palm travel and unchanged guards.
- [x] Implement explicit200-to400s^-1 weights-only migration with every other
  contract field fixed and source provenance retained; fresh optimizer/counters.
- [x] Attempt eight deterministic evaluations of the v30 policy under400s^-1;
  env1 fails the original mimic guard at149 control frames. Eight episodes were
  not completed. Reject the proposed v31 nf400 training launch.
- [x] Probe immediate per-env termination/reset for finite mimic faults at the
  original threshold:20 actual faults over2869 vector steps;29 reset isolation
  checks exactly preserve other environments. These remain failed episodes.
- [x] Wire a default-off protected adapter with strict reset acknowledgments,
  preserved terminal reward/observation, cross-tick validation of final-substep
  resets, unchanged strict evaluation and explicit weights-only migration.
- [x] Complete the production-module probe:2259 vector steps,10 actual faults
  (target9, concurrent faults on one tick),17 exact isolation checks and one
  final-substep reset followed by successful first-physics-step validation.
- [x] Launch v31 protected production CLI from v30 weights, with fresh optimizer
  and counters; initial monitor is in admission, not yet learning.
- [ ] Complete the longer run and assess actual object lift/stability; no grasp
  qualification yet.

Files: tabletop_env.py generator interface; new tabletop_vec_env.py and tests;
new tabletop_vec_training.py and tests; new tabletop_vec_isaac.py scene adapter;
new scripts/train_dh116_tabletop_vec_ppo.py CLI; docs/evidence.

Existing v25 naturally stopped at36268 steps due mimic_error before changes.
Keep its diagnostic checkpoint. No concurrent Isaac processes. No source or
artifact upload. Success remains actual object lift/stability, no four-pad gate.

Actual validation: v27 completed8192 transitions safely but deterministic
evaluation failed on mimic error. Controlled replays identified lift/contact
constraint conflict; vector-only finite mimic compliance is recorded in the
contract, with the .02rad hard guard unchanged. Fresh v28 completed8192
transitions/69 optimization epochs, then8 safe deterministic evaluations, all
timeouts and0/8 grasp successes. Status completed_budget, exit0; this is short-run
physical safety evidence, not a qualified grasp.

V29 launched from v28 with4 environments and262144 additional transitions,
8 final evaluations and checkpoints every2048 transitions. It stopped at33996
total transitions (25804 new),287 optimization epochs: env2 right finger22
mimic error -.0247419rad exceeded the unchanged .02rad guard. Exit2 and a saved
checkpoint; budget and final evaluations incomplete. The user asked for learning
time with parameters fixed; this intervention follows an observed physical fault,
not a poor reward. V29 remains stopped and its checkpoint is retained.

Baseline replay reproduces the error at144 control steps. A100ms stable-wait
variant is rejected because it prevents almost all palm lift; faster warning
recovery also fails. A480Hz physics/60Hz control diagnostic completed901 steps
with actual palm lift and no hard fault, but no object pickup. The saved v29
policy subsequently completed8 safe deterministic evaluations at480Hz,0/8 grasp
success,exit0; control-frame peak mimic error .01018767rad. The stopped run is in
[v29 monitor](../../../runs/tabletop_vector_train_20261023_v29_4env/monitor.json).

V30 used4 environments,480Hz physics/60Hz control, a262144-new-transition
budget,8 final evaluations and checkpoint2048. It imported the v29 learned policy
with a new optimizer/counters and source33996 transitions recorded separately.
It passed admission, then stopped at612 new transitions/5 optimization epochs:
env2 training episode3 right finger22 error about-.020618rad exceeded .02rad.
Exit2 and a saved checkpoint; budget and evaluation incomplete.480Hz did not
resolve the training failure, despite the earlier safe diagnostic evaluations.
Seed provenance remains240Hz and other contract fields strict. Stable-wait,
fast-recovery and external-anchor variants are not deployed. V30 remains stopped;
the later v31 protected launch is recorded below. See the
[v30 monitor](../../../runs/tabletop_vector_train_20261023_v30_4env_480hz/monitor.json)
for the stop. The long budget and actual grasp qualification remain open.
Exact command and evidence are in
[the vector PPO record](../../tabletop_vector_ppo_20261023.md).

The initialized external-anchor prototype reports correct palm velocity and
maximum recorded attachment tracking error about17.3µm, but repeats the v30
mimic failure at152 control frames, error.0202644rad. This does not support
deploying the proposed backend as the remedy.

Keeping the original backend at480Hz/60Hz, the400s^-1 candidate replays both
v29 andv30 failed action prefixes, then extends with zero actions to901 control
frames. Both runs end in four safe timeouts with palm root_offset reaching.05m;
objects stay near.300m and grasp success is0/4 in each run. The60Hz sampled error
peaks are.0163013 and.0184418rad respectively; per-physics-step guards still run,
but these are not measured maxima across every physics substep.

The attempted8-episode deterministic evaluation using v30 weights in
`runs/tabletop_vector_evaluate_20261023_v30_nf400` stops at149 control frames /
596 transitions: env1 mimic error.0215569431rad, exit2, physically_safe=false.
The terminal frame has finite state, force about.8047N and maximum joint speed
about.195rad/s; this does not waive the.02rad guard. Eight episodes were not
completed. The proposed v31 nf400 run is rejected and must not be launched.

The temporary protected-reset probe completed2869 vector control steps and20
actual faults across env1/2/3, maximum terminal mimic.02106757rad, with29 exact
reset isolation checks and exit0. Tail validation counts include0,1,2,3,4,7;
fault8 at tick1507 has no remaining substep. These are failed episodes with their
original approximately−25 terminal rewards, not fault-free or grasp-success
evidence. No PPO training transitions were collected by this probe.

The default-off production candidate now preserves those terminal samples for
PPO after a strictly acknowledged local reset. It additionally requires a
last-substep reset to pass all original guards on the first subsequent physical
substep before any new recovery. Nonfinite, other-guard and reset-validation
failures stop globally. Evaluation remains unsafe on every physical failure.
An explicit old-policy import adds only the new failure-handling contract field;
nf200,480Hz/60Hz, layout and all other semantics remain fixed, with fresh optimizer
and counters. The production-module probe then completed2259 vector steps and10
actual faults (target9, with concurrent faults),17 exact reset isolation checks
and one zero-tail next-step validation, exit0. It performed no PPO training and
does not establish a learned grasp or universal physical recovery.

The main process ran390 related software tests successfully, then launched
`runs/tabletop_vector_train_20261023_v31_4env_protected` with4env,480Hz/60Hz,
nf200,262144 new transitions,eval8 and checkpoint2048. Explicit flags
`--protected-mimic-resets --initialize-from ...v30_4env_480hz/ppo_checkpoint.zip
--allow-protected-mimic-init` preserve learned weights with fresh optimizer and
counters. First monitor inspection shows admission,0 new transitions and0 epochs;
check the [live monitor](../../../runs/tabletop_vector_train_20261023_v31_4env_protected/monitor.json)
for later progress. The full actual command is in the vector PPO record. The
nf400 launch remains rejected; final budget completion and grasp qualification
remain open.
