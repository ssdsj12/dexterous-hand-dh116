# GPU vector expansion

The user requests up to 512 environments using available GPU memory. Preserve the
20-action/136-observation force-feedback task, geometry, rewards, physical guards
and measured grasp/lift requirements. Continue using one Isaac process and one
CPU thread. Save the current CPU checkpoint before GPU experiments.

The existing per-environment CPU implementation cannot use spare GPU memory.
Use cloned scenes, two batched hand articulations, one batched object and four
batched object-filtered contact sensors. Batch physics writes and measurements;
first retain the tested per-environment controller/task. A 512-env measurement
showed only 71 transitions/s, so batch the same host equations and compare every
step against the legacy controller, including terminal and nonfinite cases. Compute collision
geometry across all environments together. GPU environment IDs isolate contacts.
Only sample detailed traces for eight slots; retain all episode summaries and
physical failure detection. Backend identity must be recorded in the contract.

Validate two-environment GPU compatibility, then batch implementation at small
counts, then 512 capacity and physical admission. Reject a count with allocation,
nonfinite-state, isolation or physical guard failures. Record actual throughput;
do not equate GPU memory occupancy with training speed or grasp success. A full
GPU task/reward rewrite is separate and unnecessary for initial capacity testing.

Status: implementation and runtime verification in progress. No 512-env success
or grasp success claimed before observation.
