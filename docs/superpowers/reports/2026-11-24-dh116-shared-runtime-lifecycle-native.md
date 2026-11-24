# Shared Teacher/Student physics and eager-only lifecycle

## Implemented runtime

`NativeTensorPhysicsRuntime` is the policy-independent physical owner.
Teacher's 87-line adapter creates only the Teacher/model/PPO and delegates every
substep, guard, reward window and selected Reset to that shared owner.
The real archived-contact runner now constructs the model-free owner directly.
There is no second Student physical stepping path in this change.

Optional capture is disabled for the current no-camera Teacher runs. It accepts
owned measured snapshots and per-lane payloads, without an SDK/model handle.
Startup seed/reference/critic fields are excluded by a measured-field whitelist.
History is bounded, and a trusted settled selected Reset clears only its lanes.
Clock/owner/transaction/mutation fences fail closed. This is a trusted Python
observer attachment point for future TiledStereoCapture, not a sandbox or a
claim that native tiled cameras/Student training have already been validated.

`tabletop_runtime_execution` is used by the physical owner, Tensor Teacher,
Student Actor/factory and Student PPO factory. Only eager/cuda_cabi geometry is
allowed; compiler configuration, compiled wrappers and Module.compile are
rejected. No torch.compile is called. Student latent/estimated_task_state/hybrid
and AsymmetricPolicy state_dict keys and initial tensors were independently
compared with HEAD and are unchanged.

## Exit ordering and ownership evidence

The original stat-cache helper was moved verbatim to the shared lifecycle module;
the old provider retains its compatibility alias. Original clear/drop-alias/
STOP commit/freeze/manager checks and post-STOP update ordering are retained.
A source-freeze audit and local Timeline weakref check run immediately before
normal app.close. STOP's temporary `timeline` is explicitly cleared in finally.

Nested exception causes/contexts are recursively stripped of completed native
frames. The replacement text-only exception is raised outside the handler so
Python cannot reattach a native-owning __context__ chain.

Weakref-dead proves only the run-local Python Timeline wrapper was released,
not that every SDK singleton/native factory is gone. Normal app.close return
is separate from the outer debugger's natural process exit0; no forced success,
os._exit, SIGTERM or SIGKILL is used.

Startup and pre-close audit bind 22 source hashes to the parent attempt file.
The receipt gate pins attempt/audit/Timeline artifact hashes and checks current
source equality before accepting a native diagnostic for bounded PPO.

## Fresh verification

- Broad CPU: 603 passed, 15 CUDA-only skipped, 96.42 seconds,40 files.
  Log `shared-runtime-final-cpu.log`,
  SHA256 `9e3c66dab808fe289b704dc1277859486627a133f2f50aa4f158e9345b1063fa`.
- A subsequently added actual entry-signature regression failed before the fix;
  focused source-audit suite:6 passed. It catches an invalid audit keyword before
  any Isaac launch.
- Policy/Adam/RNG CUDA continuation:1 passed, GPU2,6.40s,
  SHA256 `72537e5897cc384771c7d982f64cd43391907c4099949bed36101f4382c31779`.
  This resumes algorithm state only; physical state is rebuilt by Reset.
- CUDA numerical kernel retains v6 binary SHA
  `57baa26f73b2973bee46d3e27faa021cb061e37ea1fb4ac366db3a0b7280942a`.
  Production GPU13 tests use a poison compiler entry; no compiler was invoked.

## Real model-free two-lane native cohort

GPU3,2 lanes,2 regex Articulations,218 SDK steps (2 initialization+216 measured),
27 unchanged archived command windows,0 Teacher sampling/rollout/PPO updates.
Natural inferior exit0, no signal/force,57.400s total including startup.
524 actual hand pairs in active lane0,0 in peer;15 captured nonzero hand
reaction steps and14 next-SDK applications; max force0.1269579364N.
Lane0 Reset at SDK210, continuation211; peer fields remain byte-identical.

436 side snapshots passed both before/cache/after raw-read oracles with no
mismatch. Official-cache mode is explicitly recorded; the N128 timing path
turns off diagnostic extra reads.

Timeline weakref supported, alive=false before App.close, local release=true.
Source-freeze admissible=true and no drift. These results do not establish
grasp success or Teacher admission.

Receipt: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-cabi-cache-n2-sealed-gpu3-_20c0k_a/outer.json`
SHA256 `ced355f07006ad0d9a95e702c2d03e9e69002a60f75c63d6fb7788a6c9d82bcb`.
Original scalar replay:216 SDK samples/27 windows pass all original guards;
issued actions/targets are exact, distance error2.4218643e-9m, force5.66838e-8N,
torque7.04483e-10Nm, original-SDF reward8.560969e-7.
Replay SHA256 `36c4e60069df51f9815fb659b563726ba648d0f6186e2451bcaccc12b879139f`.

## Preserved failed/excluded evidence

1. Compiled geometry trial `tensor-128-pinch-fused-gpu3-2dgvrgbt` completed
two windows then SIGSEGV11 after App.close; it remains failed and is not reused.
2. `tensor-common-cabi-cache-n2-gpu3-6kr5hrd0` naturally exited0 but optional
capture whitelist source changed after its freeze. It is explicitly excluded.
3. `tensor-common-cabi-cache-n2-final-gpu3-21k6k9qe` failed before Isaac launch
because the new audit keyword was missing from the shared entry signature.
Exit1,0 SDK steps; failure log retained, signature regression added, then fixed.
4. Final current-source sealed run above is the accepted two-lane receipt.

N128 current-source two-window probe is running in `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-cabi-cache-128-gpu3-f8z9ct1t`.
No new CABI N128 PPO update or6–8s performance/grasp qualification is claimed
until real completed update metrics and outer exit are available.


## Current-source N128 completed probe

128 lanes,2 regex Articulations,18 SDK ticks (2 initialization+16 measured),
2 on-policy sampling windows,0 optimizer updates. Window seconds:
[0.4959245091304183, 0.29345822893083096].
Warm second-window phases in seconds:
{
  "observation_policy_sample": 0.01835083798505366,
  "reward_window_begin": 0.006383019033819437,
  "project_reaction_submit": 0.047937959199771285,
  "physx_step": 0.06563877803273499,
  "sdk_readback": 0.008598911808803678,
  "raw_source_guard_mesh_submit": 0.017706817016005516,
  "rod_flat_export_reduce": 0.024771560914814472,
  "geometry_r0_and_post_guards": 0.08830571197904646,
  "value_and_reward_settlement": 0.015161254908889532,
  "window_reset": 3.441236913204193e-07
}.

Whole-GPU sampled peak3044MiB; Torch allocator peak146683392 bytes.
Source audit has no drift and local Timeline weakref is dead before AppClose.
Natural inferior exit0, no signal/forced exit. Outer total420.270878s includes
371.14s from SimulationContext ownership to authored-hand ownership, plus startup,
Reset and close. This exceeded the420s soft wall target by0.271s; no additional
physics windows or automatic termination were used. Cold launch/authoring is
reported separately from PPO update latency.

Probe outer receipt SHA256 `19808ae7aa3295f48a74de5e65de57182e82ba782702868535aaf588642061c8`.
The current real N2 and N128 receipts passed the new source/Timeline/cache gate.
Neither .2s/window nor6–8s/update is claimed met. Full real128x32x2 PPO is now
running in `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-ppo2-128-gpu3-xc6n1p7x`,
with exactly64windows/514SDK/2updates and no cameras. Its result will be a
separate completed update report, never inferred from the two-window projection.


## Release checks

The first Registry check found four failures because the new entries used an
unsupported executor/status vocabulary. The records were changed to the existing
pytest executor and pending status plus a separate execution_state=running;
no scheduler behavior was relaxed. Red log25329e0225c9ebaab21da5153d17a042929bce59afb98f1b55e8bf77b8453892.
Fresh Registry/source/gate release regression:120 passed,17.94s,
log SHA256 ac53a26ced5153a1e8881594303c40d888aaa4973357fbe4c3ce93ac39b23a57.

All528 protected source/script/config bytes match the original inventory.
The two historical /tmp Teacher checkpoints are absent both before and after;
their bytes cannot be rehashed and no claim is made to have restored them.
No raw models, .so binaries, images, experiment logs or private data are staged.
PR#2 was re-read via GitHub: open, draft=true, merged=false; it is not modified.
