# Successful tabletop teacher trajectories and PPO student

The user approved behavior-cloning initialization of the current PPO actor,
followed by autonomous PPO optimization. This replaces blind continuation of
the unsuccessful PPO run. The last run stopped at iteration57 on a distinct
3.14855rad/s contact impulse; its checkpoint and diagnostics remain preserved.

Use the current rigid-cylinder Isaac scene and the real TensorWorld/TensorTask/
TensorEnv stack. Keep object geometry, material, physics, force limits, rewards,
the signed-clearance restoration and every physical guard. Run only one Isaac
process at a time. Ordinary failures reset selected environments; a hard fault
stops the experiment and cannot produce qualified training data.

The teacher is a measured-feedback search controller, not a previously trained
expert. It searches the six contact depths -4,-3,-2,-1.5,-1,-0.5mm, with shared
world-Z actions, per-hand force targets between2 and12N, bounded small XY, and
joint targets inside the existing envelope. Candidate actions use exactly the
same20-dimensional normalized action interface as PPO. Measured PhysX contact
centers/normals guide target selection and closed-loop Z corrections; missing
contacts are not replaced by palm proxies. No hidden task-state writes,
synthetic object force, grasp attachment or bypassed admission is allowed.

Search small parallel candidate batches first, rank measured outcomes and
refine promising joint/load/depth combinations. Record observations BEFORE
their associated actions, executed20D requests, next terminal observations,
phase, contact evidence, force channels, reward, termination and configuration.
Keep failed trials as diagnostics, never as successful imitation labels.

A qualifying trial must pass the existing task success predicate: object bottom
at least0.33m above the0.30m table, speed below0.03m/s, slip at most0.02m and
nonnegative normal-plus-friction vertical force for0.5s in lift. Close retains
the current nonforce dwell followed by the instantaneous normal-only force
gate, with4s timeout. Entering lift or briefly gaining height is insufficient.
Retest the selected controller/trajectory from fresh resets at least three
times. Explicitly record repeated results and retain only successful episodes.

Train the existing156→128→128→20 bounded actor by supervised observation/action
imitation. Split by whole trajectory for evaluation; do not randomly leak
adjacent samples across train/validation. Load actor weights only into a fresh
PPO instance; critic, optimizer and counters start fresh. Use the existing
allowed noise floor for the initial student. Keep the original PPO objective
and rewards. Validate the student in closed-loop Isaac before continuing4096
environment,1024-episode-per-environment PPO optimization. Failed student
validation returns to teacher collection/imitation rather than claiming success.

Artifacts carry full contracts, source hashes, episode identities and explicit
physical/replay qualification. A teacher artifact is not a PPO resume checkpoint;
the initialization path rejects mismatched observations, actions, controls,
physics, rewards, missing qualification or nonfinite weights. Only deployment
batch size and PPO batching metadata may differ between teacher and student.
No successful dataset means no imitation training or student optimization.
