# High-friction object PPO trial

User authorized changing the current grasp object to a higher-friction material and retrying PPO. Keep the rigid hollow straw, geometry, mass,20D controls, rewards, success criteria and hard guards. Use a fresh same-seed4-environment run to compare with v33; do not silently resume a checkpoint with different physics.

- [x] Confirm v33 has ended:65536 transitions,472 optimization epochs,8 safe evaluation timeouts,0/8 success. No Isaac process remained.
- [x] Add opt-in object material profile: static friction1.2, dynamic1.0, restitution0, friction combine=max, restitution combine=min. These are simulation parameters, not a measured material claim. Object/table contacts change along with object/finger contacts.
- [x] Bind physics material to all32 straw colliders without changing visual bindings or other objects' materials. Version parameters in the checkpoint contract; preserve the default profile's historical contract.
- [x] Red/green tests and related regression:222 passed. Compile and diff checks passed.
- [x] Launch one CPU0/single-thread Isaac instance. Verify each of4×32 actual PhysX shapes reads static1.2000000477, dynamic1, restitution0 and the requested combine modes.
- [x] All12 physical admission episodes passed. Training audit:3136 transitions/13 optimization epochs, no safety stop or protected reset; no grasp success yet. The2048-step checkpoint loaded successfully with finite weights,20D actions/125D observations and exact configuration contract. Budget65536 new transitions and final8 evaluations remain in progress; checkpoint every2048.

Run: `runs/tabletop_vector_train_20261024_v34_high_friction`; log: `/tmp/dh116-v34-high-friction.log`. The command is the v33 launch with `--object-material high_friction` and the new output path. Original assets and v33 outputs are preserved. No uploads.
