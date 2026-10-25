# Shared-physics multi-environment direct PPO

## Current adaptive hollow-straw run (v33)

The latest user request gives PPO control of grasp-point corrections and grip
stiffness, and replaces the solid object with a rigid hollow straw. The new
policy has20 actions (12 finger targets,6 per-hand XYZ offsets,2 stiffness
targets) and125 observations. Palm targets are limited to ±3/3/4mm and2mm/s;
stiffness is10–20Nm/rad with10/s slew, damping.2 and effort cap.20Nm. Measured
pad positions, applied gain state and absolute object bottom height are visible
to PPO. No action earns reward just for commanding more force. The existing
object-support/height/stability reward and actual lift success gate remain.

The straw has16mm outer diameter,.5mm wall and.5m length. Its32 separate convex
annular collision segments preserve an open bore; it is not a whole-object
convex hull. Nominal mass10.956304g and hollow-cylinder inertia were verified
against the actual loaded PhysX values. It remains rigid, not deformable.
Success on this object is separate from the old120g cylinder task.

New geometry puts the target opposed pads at cylinder-center height. Actual
pre-open reset pad heights differ from the center by approximately−.31 to−.54mm,
with2.18/2.21mm minimum hand/table clearance. Initial height jitter is0 for this
first run. V32 failed before training because the old extra pre-open operation
put right finger12 below the USD lower limit; adaptive spawn now uses the legal
target center before the explicit reset. V33 passed independent/repeatable reset
and all12 actual admission episodes, then started PPO. Admission reached lift
but did not lift the object. The first training audit read1388 transitions and3
optimization epochs; subsequent counts and physical results are in
[the current monitor](../runs/tabletop_vector_train_20261023_v33_hollow_adaptive/monitor.json).
The2048-step periodic checkpoint has been written. Training remains in progress;
neither the65536-transition budget nor final8-episode evaluation is complete.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_vec_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --adaptive-grasp \
  --grasp-initialization runs/tabletop_adaptive_geometry_20261023/initialization.json \
  --object-kind hollow_straw \
  --output runs/tabletop_vector_train_20261023_v33_hollow_adaptive \
  --num-envs 4 --physics-hz 480 --mimic-frequency 200 --protected-mimic-resets \
  --reset-height-jitter-m 0 --timesteps 65536 --eval-episodes 8 \
  --checkpoint-every 2048 --headless --device cpu
```

The output directory must be new. This process is already running; do not launch
a second Isaac instance. The old12D checkpoint is preserved but not loaded into
the20D policy. Exact contract equality is required on resume, including the
straw asset SHA, mass/dimensions, authored initialization and actuator bounds.
V31 was intentionally interrupted for the live demo/new task and saved12736 new
transitions/103epochs. V18 is historical and no longer the active monitor.

Final related software regression:484 passed. Real SB3 with synthetic physics
also completed64 transitions/2 updates and exact save/reload; that result only
validates policy plumbing. Live admission and subsequent learning are separate
evidence. No code or run artifacts were uploaded.

## Historical 12D setup

User requested multiple environments and live monitoring. The current setup is
4 environments in one Isaac SimulationContext, CPU0 and one numerical thread.
Each has two DH116 hands, its own rigid cylinder/table and exactly filtered
contact sensors. Environments are spaced 2m along XY, with explicit cross-env
collision filtering. The existing object-stability reward and 12 target actions
remain unchanged; four-pad contact is not a success condition.

## Execution and isolation

`TabletopEnv.step_iter()` yields a command before each physics substep. The
vector scheduler gathers every command, advances the shared simulator exactly
once, then resumes every task's measurements/reward. V30 uses480Hz physics and
60Hz control: eight shared substeps per control tick regardless of environment
count (v29 used240Hz and four substeps). No subprocess environments or threads.
Single-env `step()` still drives the same generator via its original backend.

Each completed environment saves terminal observation/info before auto reset;
other environments continue their own episodes. A reset never steps physics.
The old single-env 12-step reset settling is intentionally absent in vector mode
(`reset_settle_steps=0`): otherwise resetting one env would secretly advance the
others. Contact observations are zeroed at reset and remeasured after real shared
physics. This is a contract change; no plain migration of v24/v25 checkpoints.
The initial vector model is trained fresh with the same bounded-mean architecture.

All hard physical guards remain. The default adapter stops training on a physical
failure in any environment and saves a diagnostic checkpoint. The opt-in protected
mimic adapter under validation below preserves a bounded mimic fault as a failed
episode followed by a checked local reset. Evaluation remains strict. Safe failure
to pick up the object is learnable and may time out; no success labels are fabricated.

## Evidence

- Historical v25 single-env continuation stopped at36268 total steps due right
  finger22 mimic error -.020902 rad in early lift. Earlier v24 short-run safety
  did not prove stability for arbitrary exploration. Its checkpoint is retained.
- v26 2-env actual Isaac probe: independent resets and repeatability passed;
  three batches × two envs =6 complete episodes, all close/hold/lift, all timeout
  with no physical failure. This verifies reachability, not pickup success.
- v27 4-env: 8192 PPO transitions completed safely, then deterministic evaluation
  stopped at150 vector steps due mimic failures in env0/1/3. Planned8 evaluations
  did not complete. No pickup success. Checkpoint retained.
- The v27 failed action/offset prefix reproduced the failures. Halving lift speed
  still failed (env2 at153 control steps); disabling root lift ran901 steps without
  a hard fault. Both are diagnostics only, not deployed task changes.
- Finite mimic compliance (natural frequency200 s^-1, damping ratio1) with the
  original lift speed ran the same4-env prefix plus zero actions through901 steps
  without a hard fault. Observed control-frame peak error .0124977 rad. All timed
  out; no object pickup. An initial API-method attempt failed before simulation;
  the successful test used the installed PhysX107.3 codeless USD attributes.
- v28 completed fresh4-env training under compliant mimic:8192 aggregate
  transitions and69 optimization epochs, followed by8 deterministic evaluation
  episodes (2 per environment). All8 evaluations were physically safe timeouts;
  grasp success0/8. Its [summary](../runs/tabletop_vector_train_20261023_v28_4env/summary.json)
  reports completed_budget, exit0, passed=false and grasp_qualified=false. This
  establishes completion of this short training/evaluation run, not reliable
  pickup or stability for arbitrary future actions.
- v29 launched from the v28 checkpoint with4 environments,262144 additional
  training transitions,8 final evaluations and checkpoint interval2048. It stopped
  at33996 total transitions (25804 new),287 optimization epochs after env2 right
  finger22 mimic error reached -.0247419rad. The unchanged .02rad physical guard
  caused exit2. The checkpoint and matching counts are saved; the requested budget
  and final8 evaluations were not completed. V29 itself remains stopped.
  This intervention follows a measured physical fault, not low reward or lack of
  grasp success during early learning.
- The saved v29 policy then completed8 deterministic closed-loop evaluations
  under480Hz physics/60Hz control: all8 physically safe timeouts,0/8 grasp
  successes,exit0. Control-frame peak mimic error was .01018767rad. This is new
  candidate-configuration evidence, not a completed v29 training evaluation or a
  long-run stability guarantee. See the
  [evaluation report](../runs/tabletop_vector_evaluate_20261023_v29_480hz/summary.json).
- V30's4-env production entry launched with480Hz physics and the v29 learned
  policy weights, a262144-new-transition budget,8 final evaluations and
  checkpoint2048. It passed admission and performed actual PPO training, then
  stopped at612 new transitions/5 optimization epochs due env2 training episode3
  right finger22 mimic error about-.020618rad. Status physical_safety_failure,
  exit2, model/counts saved; budget and final evaluations incomplete. Source33996
  transitions remain separate provenance from the new optimizer/counters. This
  counterexample shows that480Hz and the previous safe deterministic evaluations
  did not resolve stochastic-training stability. V30 remains stopped.
- An initialized floating-hand/external-kinematic-anchor prototype reports correct
  palm velocity and maximum recorded attachment tracking error about17.3µm, but
  the original v30 action prefix still fails at152 control frames with mimic
  error.0202644rad. This backend is rejected for deployment; it did not fix the
  observed failure. See the
  [anchor replay](../runs/tabletop_vector_replay_20261023_v30_anchor_warmup/summary.json).
- Keeping the original backend and480Hz physics/60Hz control, changing only mimic
  frequency200→400s^-1 gives two completed diagnostic extensions: the failed
  v29 andv30 action prefixes, each followed by zero actions to901 control frames.
  Each ends in four physically safe timeouts; palm root_offset reaches.05m and
  object bottoms remain about.300m, success0/4. The60Hz sampled mimic peaks are
  .0163013rad forv29 and.0184418rad forv30, not all-physics-substep peak claims.
  See the [v29 replay](../runs/tabletop_vector_replay_20261023_v29_nf400/summary.json)
  and [v30 replay](../runs/tabletop_vector_replay_20261023_v30_nf400/summary.json).
- The attempted eight deterministic episodes using the saved v30 policy under
  400s^-1 stop at149 control frames/596 aggregate transitions. Env1 mimic error
  reaches.0215569431rad; exit2, physically_safe=false, eight episodes incomplete.
  The terminal state is finite, with force about.8047N and maximum joint speed
  about.195rad/s, but still violates the unchanged.02rad guard. The
  [evaluation report](../runs/tabletop_vector_evaluate_20261023_v30_nf400/summary.json)
  disproves the candidate's readiness for continued training. The proposed v31
  nf400 training must not launch. Successful finite replay extensions did not
  establish closed-loop safety or a learned grasp. The later v31 launch below
  keeps nf200 and changes only validated terminal/reset handling.

Software tests verify one shared step per substep, early terminal observation
preservation, reset isolation, fixed evaluation quotas, terminal NaN inspection,
per-env admission, global safety stop, checkpoint/resume and real SB3 updates.
Historical related verification ran246 passed,1 skipped, followed by49 targeted
passes after layout-equality regressions. For the protected-reset candidate, the
main process subsequently ran390 related software tests successfully. This is an
executed new check, not a sum of historical counts. Full repository tests were not
run, and software checks are not evidence of long-run physical or grasp success.

## Monitoring

The runner writes atomic `monitor.json` and `training_progress.json` updates
about every2seconds (also at256 aggregate transitions), plus model checkpoints.
`timesteps` sums transitions over all envs; `steps_per_env` shows each env's share.
The per-env section contains episode count, return, phase visitation and success,
plus latest/measured-maximum object bottom, slip, contact force and mimic error.
These values are sampled at control frames (not guaranteed physics-substep peaks).
Terminal physical measurements are copied before resetting that environment.
Success and physical safety are separate. The JSON file continues updating while
the process runs; no chat message is required for these updates.

Current launched run: [v31 protected monitor](../runs/tabletop_vector_train_20261023_v31_4env_protected/monitor.json).
The first inspected monitor reported `admission`, zero new PPO transitions and
zero optimization epochs. It has a262144-new-transition budget,8 final evaluations
and checkpoint2048. The model imports v30 policy weights with fresh optimizer and
counters; source612 transitions are separate lineage. Subsequent admission and
learning progress must be read from the live monitor rather than this snapshot.

Latest stopped run: [v30 monitor](../runs/tabletop_vector_train_20261023_v30_4env_480hz/monitor.json)
and [final report](../runs/tabletop_vector_train_20261023_v30_4env_480hz/summary.json).
Its new counter stopped at612 after weights-only initialization; source33996
transitions are separate lineage and are not added to the new model count.
Earlier stopped run: [v29 monitor](../runs/tabletop_vector_train_20261023_v29_4env/monitor.json)
and [final report](../runs/tabletop_vector_train_20261023_v29_4env/summary.json).
This monitor now records the stop; it is not evidence of active training. The v18,
v25 and v28 monitors are also historical.

The separate400s^-1 candidate evaluation at
`runs/tabletop_vector_evaluate_20261023_v30_nf400` has also stopped with a physical
failure. The proposed
`runs/tabletop_vector_train_20261023_v31_4env_480hz_nf400` training was not launched
and is rejected following this result. The distinct v31 protected run uses200s^-1.

The [temporary protected-reset probe](../runs/tabletop_protected_mimic_reset_probe_20261023/summary.json)
completed2869 vector control steps and20 actual bounded mimic faults across
env1/2/3, with no global stop and exit0. All29 local reset isolation checks left
other environments exactly unchanged. Fault-state mimic maximum was.02106757rad;
checked remaining-substep counts covered0,1,2,3,4,7. The zero-tail case was fault8
at tick1507. This is recovery evidence: those20 episodes still failed the original
.02rad guard, received their original approximately−25 terminal rewards, and
earned no success qualification. The probe performs zero PPO updates/transitions.

The independent production candidate `ProtectedMimicVecEnv` is now wired behind
`--protected-mimic-resets`, off by default. It checks both hands and the unchanged
geometry/load guards before considering a finite mimic failure below.03rad for
local reset. The.03rad boundary restricts eligibility for recovery; the failure
threshold remains.02rad. Nonfinite, other-guard, reset and post-reset faults stop
the whole world. Remaining shared steps settle the reset state without advancing
the new episode clock or adding simulator steps; the returned observation is the
latest measured reset state. A reset on the final substep additionally remains
pending until the first following physical substep passes all guards, preventing
immediate re-recovery of a bad reset. Terminal evidence is not mutated afterward.

Training acknowledges only the strict versioned reset protocol, retaining the
original reward, done and failed terminal observation in the SB3 rollout. Admission
and evaluation still reject every physical failure, including recovered mimic
episodes. `--allow-protected-mimic-init` permits only an explicit weights-only
import that adds `training_failure_handling=validated_mimic_reset_v1`; all other
contract fields and vector layout must match. It requires480Hz/60Hz and nf200,
retains source provenance, and starts fresh optimizer/counters.

The [production-module probe](../runs/tabletop_protected_mimic_production_probe_20261023/summary.json)
completed2259 vector control steps,10 actual faults and17 exact reset-isolation
checks, with exit0 and no global stop. Its target was9 faults; simultaneous faults
in one tick brought the actual count to10. It includes one final-substep reset
followed by successful next-physics-step validation. No PPO transitions were
collected. This separately validates the stricter production adapter, including
its cross-tick pending check, for the measured trajectories. It does not establish
contact quality, object pickup, or safety for arbitrary future exploration.

V31's actual entry has now launched with4 environments,480Hz physics/60Hz control,
nf200 and explicit protected handling. The initial monitor is in admission, so
launch is not yet a claim that PPO collected training data or completed its budget.
The actual command is below (the existing output directory must not be reused):

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_vec_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --output runs/tabletop_vector_train_20261023_v31_4env_protected \
  --num-envs 4 --physics-hz 480 --mimic-frequency 200 --protected-mimic-resets \
  --timesteps 262144 --eval-episodes 8 --checkpoint-every 2048 \
  --initialize-from runs/tabletop_vector_train_20261023_v30_4env_480hz/ppo_checkpoint.zip \
  --allow-protected-mimic-init --headless --device cpu \
  > /tmp/dh116-v31-protected.log 2>&1
```

A read-only audit of the matching107.3/PhysX5.6.1 source provides a stronger
basis for that diagnostic: CPU tensor joint-position and root-pose writes use
`applyCache(ePOSITION/eROOT_TRANSFORM)`, whose SDK implementation updates each
link's contact manager and drops cached contact data. The CPU implementation of
`updateArticulationsKinematic()` is empty, so this global call does not itself
modify other environments; the selected articulation writes perform the work.
See [CPU tensor writes](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/omni/extensions/runtime/source/omni.physx.tensors/plugins/cpu/CpuArticulationView.cpp#L465),
[SDK cache application](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-physx-5.6.1/physx/source/physx/src/NpArticulationReducedCoordinate.cpp#L153)
and [CPU kinematic update](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/omni/extensions/runtime/source/omni.physx.tensors/plugins/cpu/CpuSimulationView.h#L91).
This supports testing the existing local reset path; it does not replace actual
fault-reset-following-step validation or justify adding a global scene reset.

The implemented mimic refinement accepts only200→400s^-1 at fixed480Hz physics,
60Hz control and damping ratio1. Every other checkpoint field must match,
including vector layout/backend, reset seed, reward and controller. It cannot be
combined with physics refinement or resume. Policy tensors are copied exactly;
optimizer/counters start fresh, source SHA/contract/counts are recorded and no
success qualification is imported. Exact same-contract400s^-1 resume remains
available. The default remains200s^-1 and the hard guard remains.02rad.

The actual v30 launch command was:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_vec_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --initialize-from runs/tabletop_vector_train_20261023_v29_4env/ppo_checkpoint.zip \
  --allow-physics-refinement --physics-hz 480 \
  --output runs/tabletop_vector_train_20261023_v30_4env_480hz \
  --num-envs 4 --timesteps 262144 --eval-episodes 8 --checkpoint-every 2048 \
  --headless --device cpu
```

Physics refinement accepts only240→480Hz at unchanged60Hz control and unchanged
remaining checkpoint contract, including vector layout/backend. It preserves the
240Hz seed provenance as `seed_physics_hz=240`. The source model's policy weights
are copied exactly into a new PPO model; optimizer state, counters and success
qualifications are not imported. Source SHA/contract/counts are recorded. A plain
`--resume` across physics frequencies is rejected. No reward, controller rate,
guard or grasp-success threshold was changed, and neither rejected diagnostic
controller variant is deployed.

The actual v29 launch command was:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_vec_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --resume runs/tabletop_vector_train_20261023_v28_4env/ppo_checkpoint.zip \
  --output runs/tabletop_vector_train_20261023_v29_4env \
  --num-envs 4 --timesteps 262144 --eval-episodes 8 --checkpoint-every 2048 \
  --headless --device cpu
```

The requested additional budget would have brought the model counter to270336
transitions. It was not completed: v29 stopped at33996 and preserved its diagnostic
checkpoint.

Launch example (output must be new):

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_vec_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --output runs/NEW_VECTOR_RUN --num-envs 4 --physics-hz 480 --timesteps 65536 \
  --eval-episodes 8 --checkpoint-every 2048 --headless --device cpu
```

Use `--resume path/to/ppo_checkpoint.zip` only with the exact vector contract,
including num_envs/reset/physics. `--probe-only` collects physical admission only.
CPU sharing provides simultaneous trajectories/diverse sampling; it does not
promise4× wall-clock speed under the required single-core restriction.

## V29 failure and controlled diagnostics

All variants below were diagnostic replays of the failed action sequence,
executed serially before deployment. Each provides evidence only for its tested
trajectory. The480Hz physics change proceeded to additional policy evaluation
and the v30 production entry;400s^-1 compliance subsequently failed closed-loop
evaluation, so its proposed training launch is rejected.
The .02rad hard mimic guard remains in place.
Full extensions end in timeout, not a successful grasp.

| Variant | Actual result | Interpretation |
| --- | --- | --- |
| [Original configuration](../runs/tabletop_vector_replay_20261023_v29_base/summary.json) | Failure at144 control steps; peak mimic error .0247418958rad | Reproduces v29's measured fault exactly. |
| [100ms stable wait before palm motion](../runs/tabletop_vector_replay_20261023_v29_settle/summary.json) | 901 steps without a hard fault; peak .0120302rad; three environments show zero palm-root rise, the fourth rises only .208mm | Rejected: it avoids the task's lift motion rather than demonstrating stable lift. |
| [Faster affected-finger warning recovery](../runs/tabletop_vector_replay_20261023_v29_fast_recovery/summary.json) | Recovery rate .05→.3rad/s; failure at145 steps, peak .0228095rad | Rejected: physical failure remains. |
| [Half physics timestep](../runs/tabletop_vector_replay_20261023_v29_half_dt/summary.json) | 480Hz physics,60Hz control,8 substeps; all4 environments finish901 control steps without a hard fault; peak .0194060rad | Candidate trajectory evidence; followed by8 safe policy evaluations before v30 launch. |
| [External kinematic palm anchor, v30 prefix](../runs/tabletop_vector_replay_20261023_v30_anchor_warmup/summary.json) | Initialized velocity/tracking correct, recorded tracking error about17.3µm; mimic guard fails at152 control frames, error.0202644rad | Rejected; the root-motion architecture change does not resolve the recorded failure. |
| [400s^-1 compliance, v29 prefix](../runs/tabletop_vector_replay_20261023_v29_nf400/summary.json) | Recorded failure prefix plus zero actions to901 control frames;4 safe timeouts, root_offset=.05m,60Hz error peak.0163013rad | Limited candidate evidence; object remains on the table,0/4 success. |
| [400s^-1 compliance, v30 prefix](../runs/tabletop_vector_replay_20261023_v30_nf400/summary.json) | Recorded failure prefix plus zero actions to901 control frames;4 safe timeouts, root_offset=.05m,60Hz error peak.0184418rad | Limited candidate evidence; object remains on the table,0/4 success. |
| [400s^-1 compliance, saved v30 policy](../runs/tabletop_vector_evaluate_20261023_v30_nf400/summary.json) | Fails at149 control frames/596 transitions; env1 mimic.0215569431rad; exit2, eight episodes incomplete | Reject v31 nf400 training. Safe recorded-prefix extensions did not establish closed-loop safety. |

The half-timestep replay actually raises the palm roots44.81–45.22mm, unlike
the rejected wait variant. The object bottoms nevertheless remain at about
.300m, so all four episodes fail to pick up the object. Its measured error peak
is also close to the hard threshold. The subsequent8 closed-loop evaluations were
safe, but v30's stochastic training failed after612 new transitions. This preserves
the limited scope of the earlier evidence:480Hz is not a demonstrated solution
for ongoing exploration. Neither replay nor evaluation picked up the object.

## Contact stability change

Only vector scenes enable finite mimic compliance; legacy single-env authoring
is unchanged. PhysX107.3 uses `physxMimicJoint:rotX:naturalFrequency` and
`physxMimicJoint:rotX:dampingRatio`. The [official stability guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance)
explains that rigid contact and instantaneous mimic constraints can compete.
The initial frequency/timestep product was200/240=.833; v30's480Hz physics makes
it200/480=.417 while natural frequency200s^-1 and damping1 remain unchanged.
The candidate400s^-1 setting at480Hz returns that product to.833, with damping1;
this numerical ratio alone establishes neither stability nor physical fidelity.
These are numerical tuning values, not calibrated DH116 hardware properties. Compliance
changes physical coupling and can change grip forces; it is not merely a logger
fix. The .02rad hard guard, force/penetration limits and object success criteria
remain unchanged. Both compliance parameters are in the checkpoint contract,
so the earlier hard-mimic vector checkpoint cannot silently resume.
