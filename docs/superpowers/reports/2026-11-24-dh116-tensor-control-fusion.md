# Batch numerical command validation, unchanged physical predicates

Base 9ddb3ce37168e21b05799671978fe70b759d1176.
New tabletop_tensor_wrist_projection.project_tensor_policy_wrist_targets retains
original palm_transport/per_hand_safe_z and intrinsic XYZ quaternion multiply/
normalization order. Original project_policy_wrist_targets stays untouched.
Metadata is checked before math; device finite/unit1e-5/pi/2/nonzero decisions
are combined before producing any command. Full24D and all lead/rate/hard bounds stay.
TensorControl combines state/action and reaction validity checks before mutation,
FP64 30N norm precedes SDK cast, COM arithmetic remains FP64, actual reaction source
and one-shot transaction are unchanged. No mutable state guard is cached.

CPU/CUDA oracle and negative controls:147 passed. Expanded CPU:123 passed,2 skipped.
N2/N128 FP32/FP64 every projected tensor is bitwise equal to the original.
Last-row30N±1e-7, old buffer reused then NaN, wrong metadata/device/shape, pi/2 and
quaternion tolerance boundaries are covered. These are pure Torch tests, not Isaac.

GPU1 RTX4090 /Torch2.7cu128/CPU one thread; same-process30warmup+150samples,
explicit benchmark synchronization outside each measured call:
N128 projection2.47755→1.93003ms; prepare4.18486→3.40535ms;
submit0.81898→0.57715ms. Fake SDK write sinks; no native speedup claim.
GPU host numerical decisions:prepare3,submit1, before next SDK. Fixed CPU constants
are not GPU synchronization. This still does not implement eight-step deferred abort.
The actual PhysX API has no device-mask abort hook, so hard guards remain per SDK.

Evidence copied from agent-owned /tmp to persistent tensor-control-fusion-evidence.
No Isaac process, dependency change, protected-source change or training admission.
Fresh release CPU command:
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -B -m pytest -q tests/test_dh116_tensor_control.py tests/test_dh116_tensor_wrist_projection.py`
84 passed, 2 CUDA-only skipped; log SHA256 8f2b9de3420cc57cc10d3aa4df552d18bfe482bcedf35786191e218f560290c9.

A subsequent real two-lane contact replay used this control path unchanged:
`tensor-contact-replay-gpu3-nggazmc4`, 218 SDK steps, 27 windows,
14 actual nonzero hand-reaction applications in lane0, peer lane1 zero,
selected lane0 Reset at SDK210 and continuation SDK211, natural exit0.
Original-path comparison: targets/actions exact, force maximum error
5.66838e-8 N, torque7.04483e-10 Nm, reward8.56097e-7.
This qualifies command/reaction arithmetic for that cohort, not grasp success.
Independent CUDA numerical geometry and shared Teacher/Student runtime are
separate subsequent packages; their native lifecycle evidence is still pending.
