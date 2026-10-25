# DH116 tabletop PPO lifecycle — 2026-10-23

Scope: two DH116 hands and a rigid cylinder proxy; 12 joint residuals. Palm
descent/lift is scripted. This is not flexible-wire, Panda, or hardware training.
Source code and synthetic tests cannot certify physical learning success.

## Implemented behavior

- Measured early wire contact with >=2 mm table clearance can finish approach
  and hand over to close at the current palm height. A clearance-only stop is
  `approach_blocked`. The current prepared controller can also hand over with
  measured root-to-moving-goal error <=5 mm and clearance >=2 mm, after joint
  tracking catches up and without a coupling warning. This is a pregrasp
  workspace entry, not contact or grasp qualification.
- Prepare thumb/index within 0.04 rad of the calibrated pose before descent.
  Scripted XY palm motion tracks measured cylinder translation at <=0.01 m/s
  before hold; approach descent is <=0.01 m/s and close reacquisition is <=0.005
  m/s. Losing all pad contact in close releases the old permanent descent latch.
  A 0.1 mm extra stopping margin avoids rounding onto the 2 mm guard.
- A measured mimic error >=0.008 rad pauses finger commands and all palm
  translations; the original hard failure remains 0.02 rad. Actual joint lag
  also gates palm motion. Persistent pauses can still cause task timeouts.
- Nominal close references move toward the calibrated pose, independently of
  executed PPO commands. Hold/lift freeze the nominal reference. This is a
  rate-limited reference generator, not a physics-rollout MPC optimizer.
- Residual radius defaults to 0.02 rad; maximum supported radius is 0.1 rad.
  Residual changes are limited to 0.05 rad/s. Total commands remain limited to
  0.3 rad/s and coupled joint limits. Safety thresholds were not relaxed.
- `dh116-tabletop-v4` observations have 104 values: active positions/velocities,
  prior command, independent nominal reference, four wire contact loads, phase,
  task timers, root offset, close progress, stop latch, clearance, relative object
  pose and velocities, grip reference/validity, ten measured mimic errors, and residual filter state.
  Distances, joints, velocities, force and time use explicit normalization.
- The PPO reward uses measured progress and nonpositive idle/tracking costs.
  Stage bonus is separate from the original +25 grasp success / -25 failure
  events. The old standalone `task_reward` API is preserved for its consumers.
- Every episode resets the physical scene and traverses earlier phases. No
  synthetic advanced-stage reset, attachment, or object pose write after reset
  is used. Reset height varies by +/-1 mm (seeded) by default.

## Admission, curriculum and success

Before each stage, collect >=3 actual episodes: an explicit zero-residual
baseline plus bounded random probes. In later-stage random probes the policy
executes preceding phases. Require finite observations, repeated-reset agreement,
no physical safety failure, and >=2 probes that reach the requested stage. A
safe timeout is learning data; mimic, penetration, excess force, dropped-object,
joint-limit and nonfinite failures block admission. Finite sampling never proves
all possible actions safe, so the same guards stop learning at runtime too.

`--curriculum` trains approach, close (including hold), then lift with one PPO
model. The per-stage budget is `--timesteps`; promotion requires >=80% success
over the requested evaluation episodes (default five). Approach completion means
entering close, close completion means entering lift through real hold checks,
and only lift/full can report `grasp_qualified=true`. A failed stage keeps its
checkpoint and stops promotion.

The original final gates remain: all four pad loads >=0.1 N continuously for
0.2 s, then 0.5 s hold with <=5 mm translational slip; during lift all contacts
must remain and slip must be <=10 mm. The entire rigid cylinder bottom must be
>=0.33 m (table 0.30 m), linear speed <0.03 m/s, continuously for 0.5 s. These
are the current proxy metrics, not full angular-slip/contact-patch certification.

## Checkpoints and commands

Always run from the project root with `scripts/python.sh`; it fixes CPU affinity
and numerical threads. Only one Isaac process may run. Choose a fresh output
directory each time. Local pose summaries and manufacturer assets are required.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --output runs/my_tabletop_curriculum --curriculum --timesteps 1024 \
  --checkpoint-every 512 --eval-episodes 5 --headless --device cpu
```

This is a short diagnostic budget, not a convergence claim. CSV training metrics,
`training_progress.json`, `training_report.json`, per-episode traces, periodic
`checkpoint_*_steps.zip` and final `ppo_checkpoint.zip` are retained locally.
Each checkpoint has JSON metadata containing its stage, qualified prefix,
observation/action/physics contract, asset hashes, sample count and optimizer
epoch count. A radius, schema, solver or pose mismatch is rejected.

Resume an interrupted or unfinished stage with `--resume path/to/ppo_checkpoint.zip`
and a fresh `--output`. An omitted `--stage` restores the saved stage;
`--curriculum --resume ...` skips the qualified prefix. Resume retains model and
optimizer state and starts a new physical episode; it does not reproduce the
exact in-flight rollout or RNG stream. Old 47D/82D/94D checkpoints are incompatible.

Saved-model evaluation uses `--evaluate-only --checkpoint path/to/ppo_checkpoint.zip`.
Without a checkpoint, evaluation uses zero residual. Default evaluation stage is
`full`; explicitly use `--stage approach` or `--stage close` for stage diagnostics.
At least two evaluation episodes are required; five are the default. Seeds vary
the initial root height, not all physical parameters.

`--probe-only --stage ...` records fresh admission evidence without training.
Physics/error/admission rejection requests exit 2; a safely executed unsolved
task may exit 0, so always inspect `passed`, `stage_success_rate` and failure
fields. The CLI restores Python SIGINT handling after Kit startup and flushes partial
trajectories in finally. The earlier v14 external interrupt bypassed Python
cleanup through Kit (exit 0, no final summary); its periodic/stage checkpoint
survived. Do not interpret that old exit as task success. Revised signal handling
requires separate live interruption validation; synthetic interruption tests
cover the training lifecycle.

## Preserved intermediate physical evidence

These are local run directories, not files uploaded to GitHub:

| Run | Configuration | Actual result |
| --- | --- | --- |
| `tabletop_readiness_approach_probe_20261023_v1` | 82D, 0.1 rad | 3/3 approach probes complete handoff, no physical failure. |
| `tabletop_readiness_curriculum_20261023_v1` | 82D, 0.1 rad, 256 steps/stage | 512 samples, 20 optimizer epochs; approach 5/5, close 0/5 timeout. |
| `tabletop_readiness_resume_20261023_v2` | Same radius, stricter stage-directed probes | Admission rejected: right thumb mimic error 0.020634 rad; no added learning. |
| `tabletop_readiness_curriculum_20261023_v3` | 82D, 0.02 rad, 256 steps/stage | 512 samples, 20 epochs; approach 5/5, close 0/5 timeout. |
| `tabletop_readiness_resume_20261023_v4` | 82D, 0.02 rad | Admission passed; stochastic learning stopped at sample 611 before another optimizer update, left thumb mimic error 0.020812 rad. Checkpoint saved; actual shell exit 2. |

These results motivated the observed residual-rate filter. They do not qualify
the final filtered configuration. In particular, a passed finite probe set is
not sufficient reason to remove runtime safety stops.

## Final filtered configuration: actual evidence

`tabletop_readiness_curriculum_20261023_v5` used 94D/v3 observations, 0.02 rad
radius and 0.05 rad/s residual rate. Actual command:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --output runs/tabletop_readiness_curriculum_20261023_v5 --curriculum \
  --timesteps 256 --checkpoint-every 128 --eval-episodes 5 --headless --device cpu
```

Actual shell exit 0; 512 environment samples, 20 optimizer epochs. Approach
admission passed and evaluation was 5/5; close admission passed but evaluation
was 0/5 (timeout). Promotion stopped at close. Logged control samples had minimum
clearance 9.283 mm, maximum mimic error 0.015569 rad, maximum pad load 2.571 N.
These sampled maxima are not claimed to be exhaustive physics-substep maxima.

Actual resume command:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --output runs/tabletop_readiness_resume_20261023_v6 \
  --resume runs/tabletop_readiness_curriculum_20261023_v5/ppo_checkpoint.zip \
  --timesteps 128 --checkpoint-every 128 --eval-episodes 5 --headless --device cpu
```

Actual shell exit 0; restored close rather than restarting approach training,
admission passed and cumulative counts reached 640 samples / 25 optimizer epochs.
Close evaluation remained 0/5; no lift training. Logged minimum clearance was
9.508 mm, maximum mimic error 0.013414 rad, maximum pad load 0.568 N. Runtime
safety guards did not trigger in v5/v6. This finite evidence does not establish
global exploration stability or identify a unique PhysX defect.

The latest compatible checkpoint is local
`runs/tabletop_readiness_resume_20261023_v6/ppo_checkpoint.zip` with adjacent JSON
metadata. Admission is rerun on resume. `ppo_updates` counts SB3 optimizer epochs,
not episodes and not one update per environment sample.

Final software verification:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest \
  tests/test_dh116_tabletop_training.py tests/test_dh116_tabletop_env.py \
  tests/test_dh116_tabletop_readiness.py tests/test_dh116_tabletop_task.py \
  tests/test_dh116_runtime_mimic.py -q
```

Result: **101 passed, 1 skipped in 2.23 s**. The skipped schema test needs Isaac
plugins. Real SB3 toy tests cover update/save/load/resume/interruption, but are
explicitly synthetic. The filtered Isaac runs above provide separate real-physics
evidence. Later exit-policy bookkeeping changes were covered by the focused tests;
they did not change controller dynamics. Compilation and `git diff --check` passed.
Full repository tests were not rerun and prior unrelated failures remain unclaimed.

Final real CLI negative check (`tabletop_readiness_contract_rejection_20261023_v7`):
loading the old 82D v3-run checkpoint with the current 94D observation contract
was rejected with `ValueError: checkpoint contract differs from current
observation/action/physics contract`. The actual shell exit code and recorded
`process_exit_code` were both **2**; no evaluation or learning occurred.

## Remaining physical work

Close has not established continuous four-pad contact. The former fixed-height
handoff has been replaced by prepared joint posture, actual-object XY tracking,
and guarded contact reacquisition. v17 evaluations now reach close, but none
sustain the required contact interval. Lift promotion still requires measured
close/hold evidence. The automatic continuation budget is not a convergence
or successful grasp guarantee.

## Contact recovery investigation and continuation

The previous v8 monitored training stopped at 13,352 samples / 515 optimizer
epochs on left finger13 error -0.021094 rad (hard limit 0.02); it never reached
hold. Its zero probe showed index reference 0.55158 rad vs measured 0.41436 rad
at early contact. Later contact disappeared but the palm latch remained.

v9/v10 diagnostics isolated a second issue: the cylinder translated about 15 mm
while the root target remained at its spawn location. Added logs record relative
object pose/velocity, joint velocities, and applied torque. The prepared posture
and measured-object goal avoid treating this translation as a finger-only task.
v11/v12 exposed boundary rounding and a stale fixed-goal distance after XY
tracking; both were corrected before training.

- v13 (`tabletop_moving_goal_20261023_v13`): three close probes admitted; up to
  three pads contacted, no full grasp. Review caught an XY pause bypass.
- v14 (`tabletop_contact_curriculum_20261023_v14`): approach 16,384 samples,
  610 optimizer epochs, 5/5 evaluation; manually interrupted to repair the pause
  bypass. No final summary was emitted by the old Kit signal handler.
- v15 (`tabletop_contact_curriculum_20261023_v15`): pause fixed, 16,384 samples,
  613 epochs; approach 5/5 and close 0/5, physically safe, actual exit 0.
- v16 (`tabletop_close_resume_20261023_v16`): 65,536 added samples, cumulative
  81,920 / 3,072 epochs, exit 0, close evaluation 0/5. Training produced 69
  approach-blocked episodes and 53 timeouts; final evaluations all timed out in
  approach. Sporadic four-pad sampled contacts did not satisfy the phase timer.
  This motivated the 5 mm pregrasp handoff rather than more training at the
  clearance boundary. Counts of simultaneous sampled contacts are not proof of
  continuous physics-rate contact.

`--initialize-from` is an explicit weights-only transfer across the pregrasp
handoff contract change. All other observation/action/assets/physics contract
fields must match. It creates a fresh optimizer and resets sample/update counts
and qualifications; source sample count is recorded separately. It does not
silently bypass normal `--resume` compatibility checks.

The serial continuation entry is `scripts/continue_dh116_tabletop_ppo.py`.
It resumes the same policy after a safe `curriculum_stage_not_mastered` budget
exit and stops on a qualified grasp, hard physical failure, execution error,
missing artifacts, source changes, or configured round limit. It holds a
supervisor lock, checks for an existing trainer, runs one child at a time and
pins runtime source hashes. Per-round logs and `monitor.json` show progress.
A finite round limit is not a convergence guarantee.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/continue_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --resume runs/tabletop_pregrasp_transfer_20261023_v17/ppo_checkpoint.zip \
  --output runs/tabletop_continuous_20261023_v18 \
  --timesteps 65536 --max-rounds 8 --checkpoint-every 2048 --eval-episodes 5
```

Do not start the supervisor until the previous trainer has exited. Successful
subprocess orchestration tests use a synthetic trainer and do not establish
physical grasp success.


## Latest execution: v17 and v18

v17 (`tabletop_pregrasp_transfer_20261023_v17`) finished safely with 8,192 new
samples and 307 optimizer epochs after explicit transfer from v16's 81,920-step
model. All three admission probes and all five final evaluations reached close;
close success was 0/5 and no hold/lift was established. Its fresh counters do not
include the source training as uninterrupted optimizer history.

Review found pregrasp readiness was sampled before physics advance. Two failing
regressions reproduced new tracking/coupling errors during that advance; the
handoff now checks post-step measurements. The six-file focused suite finished
with **116 passed, 1 skipped**:

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest \
  tests/test_dh116_tabletop_env.py tests/test_dh116_tabletop_readiness.py \
  tests/test_dh116_tabletop_task.py tests/test_dh116_tabletop_training.py \
  tests/test_dh116_tabletop_continuation.py tests/test_dh116_runtime_mimic.py -q
```

v18 was launched with the continuation command above after confirming no Isaac
process remained. Supervisor PID at launch: 52358. It is detached from the chat
and writes `monitor.json`, per-round logs, progress, checkpoints and summaries.
The supervisor lock serializes this entry; its process check detects the tabletop
trainer specifically, not arbitrary Isaac scripts. Operators must not launch
other Isaac applications concurrently. No runtime source changes are planned
while it runs. Current success must be read from its results, not inferred from
an active process. The new Kit SIGINT restoration has not yet been verified by interrupting a
real Isaac run.

At the initial live check, v18 passed all three close admission probes and
continued PPO beyond 12,000 cumulative samples, with no reported safety stop.
Both supervisor and trainer were pinned to CPU 0. This records startup and
learning only; autonomous round-to-round continuation is covered by synthetic
subprocess tests and has not yet completed a real round at this checkpoint.


## Direct-close revision: latest user decisions

The user replaced the earlier four-pad curriculum requirement: start directly
at close and let PPO learn a useful contact topology. New `joint_target` mode
restores a measured handoff seed and gives PPO 12 position targets around the
calibrated pose (±0.08 rad, coupled limits, 0.05 rad/s). The old residual mode
remains for historical reproducibility. New 108D observation/action/reset/reward
contracts reject old checkpoints; the bounded-mean policy is trained fresh.

The seed is the first close row (index 213) of v18 round 1 zero probe. The object
has already translated 17.725 mm; world pose and nonzero velocities are restored.
Joint velocities are mapped by names. Root offsets are reconstructed relative to
the original layout. Twelve real settling substeps rebuild contacts with no
reward/timer credit. Logged contact forces/torques are never restored. This is a
new close initial condition, not a bitwise PhysX checkpoint or a no-contact proof.

Scripted close adjustment is per hand, up to 2 mm down at 1 mm/s with measured
clearance and contact guards; XY tracking is bounded to ±3 mm about the seed.
Coupling warning unloads only the affected master toward measured position;
other fingers remain controllable and palms pause. Hard safety guards remain.

The task attempts lift after 2 s close + 0.3 s hold regardless of pad count.
These timed phases never qualify grasp. Actual object bottom >=0.33 m, speed
<0.03 m/s and relative slip <=0.02 m for 0.5 s qualifies close-to-lift only.
The reward uses aggregate contact-load progress and object-height progress;
keeping forces just below the old threshold does not earn continuous reward.
Object-stability dwell requires actual elevation. Slip cost is disabled while
the object bottom is <=0.305 m because failed pickup is not established-grasp
slip. Reward v3 additionally charges only slip beyond the accepted 0.02 m;
accepted 0.015 m slip must not make successful pickup worse than no pickup.
Original four-pad task remains selectable only through the legacy mode.

Evidence:
- v18 rounds 1–5: all close evaluations 0/5. Round 6 interrupted; last periodic
  checkpoint 337920. No final summary: Kit interruption did not unwind cleanly.
- v19: direct-close reset repeatability and zero/random exploration admitted 3/3;
  no successful grip, all timeout. Zero probe was warning-frozen ~95% of time,
  motivating per-master unload instead of freezing all hand actions.
- v20: object-based gates and unload reached close/hold/lift in 3/3 real probes,
  all physically safe, object remained on table. The previous slip cost totaled
  about -280 per failed pickup; reward v2 now gates that cost on elevation.
- v21: 16384-step budget interrupted by physical guard at 1048 steps, 24 PPO
  update epochs, right finger22 mimic -0.021319 rad during early lift. Exit 2;
  checkpoint saved. No object pickup, no deterministic qualification completed.
- v22: warning reduced from .008 to .004 rad, reward v3 and correct close-stage
  object-success reporting. Fresh policy, 8192-step budget, stopped at 1957
  steps/55 update epochs, right finger22 mimic -0.020175 rad. Exit 2; checkpoint
  saved. Three admission probes and two full training episodes timed out safely,
  but the third training episode failed at 2.575 s. No object pickup.
- In v22 episode 5, palm offset stopped changing by 2.35 s; finger22 error still
  grew from -0.010668 to -0.020175 rad. The follower frequently reported its
  .3 rad/s velocity limit while sampled positions oscillated. Earlier warning
  and palm pause did not solve the physical instability; this is evidence for
  isolating coupled joint/contact dynamics, not proof of a specific solver cause.
  No training is currently active; do not automatically resume physical faults.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/train_dh116_tabletop_ppo.py \
  --pose-summary runs/reference_aligned_native_20261023_fix1/summary.json \
  --output runs/tabletop_direct_object_train_20261023_v22 \
  --control-mode joint_target \
  --close-reset-snapshot runs/tabletop_direct_close_seed_20261023/snapshot.json \
  --curriculum --timesteps 8192 --checkpoint-every 1024 --eval-episodes 5 \
  --reset-height-jitter-m .0003 --headless --device cpu
```

Current focused tests: seven files (`tabletop_direct`, `tabletop_env`,
`tabletop_training`, `tabletop_readiness`, `tabletop_task`,
`tabletop_continuation`, `runtime_mimic`) **134 passed, 1 skipped**. Includes real
SB3 bounded-policy update/save/load, direct-reset authority, reward return
ordering, per-master unload, and object-based success without four pads.
Allowed-slip episode return and explicit close-stage success reporting were
also checked against temporarily restored old behavior: both failed, then both
passed with the fixes restored. Full-suite tests were not rerun.

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest \
  tests/test_dh116_tabletop_direct.py tests/test_dh116_tabletop_env.py \
  tests/test_dh116_tabletop_training.py tests/test_dh116_tabletop_readiness.py \
  tests/test_dh116_tabletop_task.py tests/test_dh116_tabletop_continuation.py \
  tests/test_dh116_runtime_mimic.py -q --tb=short
```
