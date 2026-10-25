# Rigid training capacity

User authorizes maximum practical training resource usage. Keep the repository CPU0/single-numeric-thread and one-Isaac-instance constraints. Preserve rigid object/hand materials, positions, action/reward/control/physics/success contracts and episode budget. Existing model/optimizer/counters are saved before capacity probes.

Measured bottleneck: collision clearance rotates all 28,561 hull vertices into three coordinates despite needing only minimum world z. Equivalent z-row transform preserves every vertex, quaternion behavior, per-hand clearance, and lowest-link index. CPU0 component benchmark (trainer competing on CPU0): original 2.9673 s, proposed 0.1759 s, maximum error 8.33e-17 m, identical lowest-link indices. Not an end-to-end throughput claim.

1. Test exact rotation-height equivalence using float32 PhysX-like poses and asymmetric hulls; retain safety-related batch/reset tests.
2. Replace full XYZ transform only within collision_clearance with the exact z row. Do not change physical model or checkpoint contract.
3. Sequentially probe 1024, 2048, 4096 (current supported maximum) with identical saved policy, three timed inference/physics steps after two safety steps. Record throughput, per-component timings and free GPU memory; preserve >=1.5 GiB headroom.
4. Select highest stable throughput (prefer larger capacity if comparable throughput), explicitly resume expanded environment count, retaining weights, optimizer, counters and episode progress. Short probes do not prove training stability.
5. Confirm live learning, first optimizer update where feasible, actual GPU allocation and observer status. Report achieved usage and unverified long-run limits.

Result: 70 tests passed (one intentional overflow-test warning). Actual rigid probes: 1024 773.92/s, 2048 804.23/s, 4096 742.14/s; chose 2048. Fixed capacity-only undefined obs by obtaining a fresh vec.reset observation before timing. Resumed v71 from v70 checkpoint 7168 steps; first updated checkpoint at 154624 steps, one optimizer epoch (KL early stopping at .04), changed finite policy and finite nonempty optimizer. Runtime throughput approximately 706/s, no safety stop. Observer active. This is successful training startup/update, not successful grasp; GPU memory ~4.3 GiB and CPU0 remains the limit.
