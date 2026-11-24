# Full tensor native PPO: actual foundations and remaining integration

## Preserved scope
Target remains real 128/512-environment Pinch/Tripod/Power Teacher PPO.
The requested 6–8s per 128x32 update (<=10s upper target) is NOT demonstrated.
The old 128-lane two-update baseline is complete: 537.8467/542.5839s per iteration,
natural exit0, 3440MiB sampled peak. Optimizer itself took only 0.2050/0.1009s.
No grasp success is inferred.

## Implemented source paths
- TensorNativeTeacher reuses the original shared network/distribution, with an explicit
  flat-candidate profile and separate tensor_native_teacher_*_v1 checkpoint namespace.
- DeviceDeformablePPOTrainer and DeviceOnPolicyRollout keep observation/pending/GAE/
  minibatches on device; preserve original Adam, clipping, rollback, RNG, KL backtrack.
  Full4096 exact KL uses one actor forward. Adam step counters and CPU RNG control
  snapshots retain original PyTorch semantics; no claim that every control scalar is CUDA.
- pack_shared_tensor_inputs reproduces common66+rod41x13+context50, unknown305 NaN/false.
- TensorTrainingReward vectorizes original SMOKE/FingerGoal and FULL arithmetic;
  qualified live Contact owner adaptation remains unsupported and is explicitly rejected.
- TensorGeometry keeps complete hull vertices, FP64 bounds/interpolation and device masks.
  Sparse ambiguous cases invoke original certify/_sdf before another SDK step.
  Tripod support/routes/raw-quaternion additional checks remain required.
- TensorMeshBridge calls original CUDA set_mesh_poses_all once; retains original
  default-stream ordering and static slots; actual material is rechecked at Reset.
- TensorRodOwner consumes the new flat ABI once and preserves reaction/Reset epochs.
- Regex Articulation/setup modules author original physical USD through a source-SHA-bound
  independent function namespace. Exactly two SDK views; no per-lane view wrappers.
  Protected tabletop_vec_isaac.py is unchanged. Selected Reset preserves peers.

## C++ flat candidate: real build and real GPU proof
Independent candidate (not system installed):
/mnt/mydisk/xk/projects/dh116-flat-pairs-lqrrc_r4/backend
Binary SHA: 24ddf340cf697bf4cf8e3e50bd0ac8da66a446d404f703be33724c696ded4a6f.
CUDA12.8.61, SM89, Python3.10.12. Physical kernels unchanged; source/License inventoried.
contact_pair_samples_flat_cuda_v1 returns owned GPU indices[int32,P,4],
vectors[float32,P,12], lane_meta[int32,N,4], node forces and mesh wrenches.
Original batched API stays unchanged. Per-step read-once and post-Reset invalidation enforced.

Physical GPU2 pure solver proof: 7 advances across isolated cases, no Isaac/PPO.
One lane empty, other 492 actual pairs; raw indices/vectors bit-exact to original ABI.
FP64 scatter max difference 2.7755575615628914e-16, well within unchanged 3e-4/5e-5.
All-empty, overflow rejection, read-once, owned buffers surviving later solver mutation,
selected rod Reset continuation and GPU all-mesh submission pass.
This is native solver ABI evidence, not Isaac hand/rod coupling admission.
Raw evidence: flat-cuda-native-parity.json and flat-cuda-final-parity.log.

## Real two-regex Isaac proof
GPU3, N2, exactly two real Articulations, 26 SDK steps (2 init+8 setup+16 measurement),
two windows of distinct24D diagnostic commands, zero rod/PPO steps.
Actual target readback, dynamic response, all34 source colliders/lane and original
table/joint guards pass. Selected Reset preserves peer state and queued composer scratch
bitwise. Natural inferior exit0, no forced exit, app.close returns and lifetime releases.
Evidence: regex-two-hand-native-gpu3-e2ybw2st; API qualification explicitly marks
legacy equivalence, rod/reward exchange and full-chain admission false.
This is not the required full coupled N2 parity proof yet.

## Tests and honest synchronization limits
Fresh CPU integration:214 passed/4 skipped, including final source/control adapters.
Independent real GPU3 tensor geometry:3 passed,9.31s; all28,893 vertices across44 hulls,
N128 bounds error <=1.11e-15; rotated SDF <=6.94e-18; sparse fallback/FP32 sign flip verified.
Independent pure GPU3 pack/rollout/devicePPO:34 passed,28.49s, including4096-sample
update and full policy/Adam/RNG rejection rollback. These counts overlap earlier subsets.
Scalar TrainingReward regression:302 passed; CPU reward difference <=1.11e-16.
Original528 protected hashes unchanged; two historical /tmp checkpoints remain absent.

Exact-length flat Torch tensor compaction may synchronize a small length scalar.
It transfers no bulk contact rows to CPU, but is not literally synchronization-free.
Original PhysX APIs do not expose a GPU failure-mask abort hook. A suspicious geometry
row therefore must be resolved before the next SDK step, not delayed until window end.
Antigravity R10 accepted these constraints and withdrew the unsupported no-sync claim
and 256x16-for-128x32 equivalence suggestion. Its MCP remains pinned to an older tree;
this was excerpt review, not current-source read access. No fixed microsecond timing
claim from that review is treated as measurement.

## Next executable integration
Connect new task setup, tensor control, flat owner, tensor source/context, geometry,
reward and GPU PPO into one independent native collector. Then run actual N2 coupled
parity with selected Reset; only afterwards N128 two-window timing and >=2 real32-window
updates. Target0.2s/window remains unmeasured. Pinch/Tripod/Power must retain separate
checkpoint identities; Tripod source penetration guard cannot be replaced by SDF alone.

Final flat ABI also rejects a nondefault CUDA stream before exposing a view.
New task setup/source/controller adapters are implemented and tested; integrated native collector remains next.

## Independent review corrections before release
Eight failing regressions reproduced FP32-before-30N acceptance, nonfinite reaction writes,
duplicate/unsubmitted transactions and premature FP32 world-COM arithmetic.
The controller now validates complete same-device finite payloads before any native mutation,
checks the original FP64 30N norm before the final SDK cast, and enforces one submission.
World COM is computed from raw SDK link/local values in FP64, retaining the original
unnormalized quaternion formula. The previous 48m-origin case lost 1.67824um of lever arm.
Reaction packets now retain actual solver/run/SDK/rod/episode/Reset identities; submission
requires that exact owned mesh-wrench tensor and matching immediate-next-SDK metadata.
An explicitly empty initial reaction is available once before any solver advance.
Failure logs remain outside Git in tensor-control-review-red.log and tensor-reaction-identity-red.log.
Focused corrected source/control/setup tests:31 passed.

The shipped audit_dh116_flat_cuda_pairs.py CLI itself ran successfully on GPU2:
flat-cli-final-iqprjabr/flat-cuda-native-parity.json, 7 solver advances, zero Isaac/PPO.
This supplements the earlier working-proof script; it does not replace full coupled validation.

Independent quality review also reproduced four pre-action NaN/command faults; complete q/qd/backend/root/base/clearance and final target checks now reject before native writes. Final focused source/control/setup:36 passed; full foundation:214 passed/4 skipped.
