# DH116 V2 simulation identification design

User authorized correcting the simulation against the V2 manual, increasing useful grasp force, validating, then restarting PPO. Prior audit is docs/dh116_v2_parameter_audit_20261024.txt. No additional design approval is required for this requested correction.

The result must distinguish manufacturer facts (V2 ranges, 490+-10 g, fingertip14..16 N) from simulation identification and unknown real actuator characteristics. No uniform conversion from fingertip N to joint Nm. Do not invent measured hardware calibration.

Stop v47 and preserve artifacts. Run serial small-scene controlled probes before changing production behavior. First isolate contact/coupling load transfer using identical targets with varying coupling compliance, then gains/effort and force range. Keep failure reporting and .02rad mimic hard protection. Do not fabricate contact or hold the object in place. Tests may explore rigid coupling without changing the production default until physically validated.

Use an explicit versioned candidate profile for changed parameters and checkpoint incompatibility; old runs remain reproducible. Capture actual PhysX readback for gains, limits, masses, inertia, material and solver parameters. Preserve detailed contact/estimated-torque labels. Validate force tracking and bounded random actions before PPO. 512-env capacity remains allowed, but expensive diagnostics run at small N first.

Manufacturer mass/range inconsistencies require declared assumptions or extra data: source CAD links cannot be treated as a measured complete hand. The current correction preserves the source mass distribution and explicitly marks its finger32 anomaly unresolved; adding base mass would not repair that anomaly. Do not expand mechanical travel without verifying coupled constraints/geometry. No real robot commands.

Single Isaac instance, CPU0/numeric single thread, existing dirty10-23 workspace preserved. No commits/push. Restart PPO with a fresh policy when observation/action/physics contracts change; retain old checkpoints. Grasp success is actual lift+hold, not command effort or safe timeout.
