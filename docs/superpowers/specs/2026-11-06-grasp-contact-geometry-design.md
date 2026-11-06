# P0 grasp contact diagnostics and geometry screening

Scope is the user's P0-first request. Keep the existing tanh Gaussian PPO, v2 path, 3844D actor, 371D critic and 24D physical action contract. No long PPO job is admitted in this phase. Historical checkpoints/data and the original physics, actuator limits and 30 mm / 5 s / 0.5 mm native criterion are immutable. PR remains Draft.

## Read-only physical evidence

An opt-in observer runs after the current physical substep and before any terminal reset. It binds each record to run, environment, episode, absolute physics step, control substep, simulator timestamp and dt. Object-filtered PhysX manifold points, signed normal-load scalars, normals and separation retain their native provenance and slot identity. CAD/FK reference positions never become measured contact points. Unknown measurements remain unknown.

Each left/right distal pad reports contact establishment, ongoing sampled duration, loss and episode-end censoring. Geometry-in-pad and correct-facing-normal are distinct gates so rejected normal directions remain diagnosable. Report NO_CONTACT, ONE_SIDE_CONTACT, WRONG_NORMAL, CONTACT_LOSS, NON_OPPOSED, TORQUE_LIMITED and SLIP as potentially simultaneous reasons. Unobservable data is a separate quality state, never a successful contact or zero slip. Torque fields explicitly name the implicit actuator simulation estimate; no hardware claim is allowed.

Pinch and tripod groups have separate duration denominators, validity coverage and bilateral/per-hand opposed-contact numerators. The diagnostic style mask is explicit; historical all-five-pad opposition statistics keep their original definition. Episode resets cannot connect contact durations across episodes. Raw records and events are streamed to independent files with hashes and schema versions; no unbounded Python heap accumulation.

## Static geometry

Generate left/right arc-length positions along a finite nondegenerate centerline. Use each hand's source USD collision hulls, actual affine mimic FK and the conservative CAD distal-pad atlas. Optimize only candidate reset joint/wrist configurations within the original limits; physical parameters and actuator limits do not change. Screen reach, radial normal orientation, opposing bearing faces, central pad footprint, object/hull penetration, self and inter-hand collision, table clearance and upward support feasibility. Straight-cylinder and bent-centerline proxy limits must be labelled. Check full available hulls after any optimization proxy. Candidates contain explicit source hashes, thresholds, rejection reasons, grasp style and per-hand targets. Never fall back to failed candidates in the accepted list.

Geometry-only results are proposals, not force closure, cooked-PhysX validation or native success. Native contact experiments require an accepted frozen candidate and an independent directory. Initial object/reset writes are permitted; no external object forces, attachments or support changes may manufacture contact or success.

The bounded historical observation smoke validates the new read-only diagnostic pipeline on original cold reset cases. It does not execute optimized candidates or certify their grasp geometry. The candidate-to-native acceptance/reset gate is not implemented in this P0 increment, and all geometry candidates retain `native_experiment_eligible=False`. Once static candidates exist, that separate gate must bind the accepted report, frozen source physics and actually applied reset before optimized-candidate closed-loop experiments are admitted.

## Execution and evidence

Use CPU tests first, then offline source-CAD screening. Wait for the existing actual1439 evaluation to release the one-Isaac gate before any bounded contact probe. Short probe mode creates no PPO/Adam object, writes no trained checkpoint and cannot issue a 100-case native evaluation certificate. Preserve the paired historical runner unchanged. Keep CPU, native contact evidence and true native success counts separate in README, CHANGELOG and validation report.

Future stages remain separate: actual129/1439 same-state report; physical-criterion Grasp/Hold/Lift curriculum; single-factor R0-R5 and Adam comparisons; generalization tests. R4/R5 require supported factors. None is silently started by the P0 observer or geometry CLI.
