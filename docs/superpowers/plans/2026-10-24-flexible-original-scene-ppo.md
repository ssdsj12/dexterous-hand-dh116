# Original scene flexible-wire PPO integration

User correction supersedes the independent demo: replace only the original Wire object with the bundled DeformX flexible rod. Preserve original wire CENTER/axis and hand root/joint reset poses exactly, no X contact probe and no table-height correction. Original center comes from authored seed (z=.308), not inferred table contact (.305). Preserve 20D joint/palm/force action, reward formulas/weights, measured grasp gate, close/hold/lift control; local fixed-material regions remain user-selected lift geometry. Run 512 environments, target1024 episodes/environment; new physics gets fresh PPO optimizer and counters, preserve rigid checkpoint. Live GUI remains unfinished until actual policy is connected and rendered correctly.

## Tasks

- [x] Batched CUDA Cosserat: preserve constitutive/integration equations of existing TorchCudaCosseratRod; vectorize across environments, all34 convex hand collision meshes, normal-only pad forces and total reaction wrenches, per-slot reset without advance, noCPUperenv simulation loop. Validate against single rod at smallsteps and contactimpulsebalance, measure512env throughput. New files vendor/DeformX/co_sim/batched_cosserat.py and optional batched_contact.py, tests/test_batched_cosserat.py.
- [x] Isaac/PPO adapter: share original GpuWorld control/handgeometry; external object adapter provides local-material measurements, true contact forces/reactions, same handseed and center. Trainer flexible option/entry uses same PPO actions/rewards and fresh1024eps512envbudget. No successful grasp prerequisite; bounded finite initial state allows failed-contact episodes tolearn. Testresetisolation and unchangedcontrol.
- [ ] GUI renderer fix: inspect rest-pose-only Hydra versus correct offline UsdSkel. Render actual original mesh deformation, preserving mesh and material; original skin cannot be silently replaced by proceduralcylinder. Add targeted regression and actualGUIverification.
- [ ] Actual singleIsaac smoke, then512envstart and verify PPO rollout/update. Keep source/asset/checkpointprovenance, refreshactivepointers and README/CHANGELOG.

## Batched backend interface

`BatchedCosserat(config: GpuRodConfig, num_envs: int, device='cuda:0')`.
State tensors in per-env LOCAL coordinates: `position(B,3,N+1)`, `velocity(B,3,N+1)`, `director(B,3,3,N)`, `omega(B,3,N)`, `radius(B,N)`.
`register_meshes(vertices: list[np.ndarray], faces: list[np.ndarray])`: constant link-local convex hulls in stable orderM; coordinates no envorigins.
`advance(duration, positions, rotations, velocities, omegas) -> dict`: torchinputs positions/velocities/omegas(B,M,3), rotations(B,M,3,3), allenvlocalworldaxes; velocitiesatlinkorigin. Predict at substepmidpoints. Returns total `linear_impulse(B,M,3)`, `angular_origin_impulse(B,M,3)` aboutenvlocalorigin, `normal_impulse(B,M,3)` for padforceobservations, `peak_force(B,M)` and balance diagnostics. Physicalhand reactions useTOTALforceandtorque inclfriction. All shapes interact, independent envs. Accumulateimpulsesoverrodsubsteps <=cfg.dt. `reset(indices)` restores initialfree state/buffers forchosenenvs withoutadvancingothers. Numericalbadstate must fail explicitly.

Parametergeometry remains originalpaperasset L1m/R.005/N20/rho700/E200000/G=E/3/damping.1/dt1e-5. Contacts k10000/damping10/mu1/tangentdamping10/10samplesperelement; tablefinitebox. No fixedattachments or ropeselfcollision. No physicsapproximations added for speed without explicit documentation/validation.

## Work constraints

CPU0, one numericthread, onlyrootlaunchesIsaac, no second instance. All agentoffline tests use `DH116_NUM_THREADS=1 bash scripts/python.sh`. Preserve dirtyexistingwork andsourceassets. No commits/push. Independent work domains share only this fixed interface; agents own distinctfiles. Root integrates trainer/world.

Actual integration evidence: 2-env Isaac startup validated at original seed, profile run exit0. Initial12 PPO transitions stopped for slow exact near-hull queries, no safety failure; radius-bounded exact BVH optimization passes11CUDA tests, actual-pose512 standalone interval0.256s. v66 512env/1024eps-per-env fresh training launched; first update and GUI still pending.

512 actual startup passed: center error5.55e-17m, handroot maxlocalerror6.236e-7m, same136Dobs/20Dactions, no physical protection at last sampling check. Fresh PPO is running; first update not yet observed, GUI runtime still unverified. Final integration read-only review found no blockers.

First actual flexible PPO rollout completed:32768 transitions,5 optimization epochs,512env subsequent sampling, no safety stop. Evidence runs/tabletop_flexible_20261024_v66_env512_eps1024/first_update_verification.json. This monitoring pass did not change training parameters or restart the process. GUI verification remains outstanding; no grasp-success claim.
