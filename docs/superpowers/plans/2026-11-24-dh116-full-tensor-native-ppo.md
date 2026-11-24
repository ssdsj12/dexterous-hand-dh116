# DH116 full tensor native PPO implementation

Active objective: accelerate actual 128/512-lane three-grasp Teacher PPO.
User-authorized scope: five linked changes; original physics and guard predicates unchanged.
Base acceleration commit c79558170a9b9aaaeb4fbebc5ba525549bd7202a already passed N128 two-window native proof.
Its 16.3 s/window is insufficient. Do not relabel it as the requested final acceleration.

1. Regex articulation ownership
   Reuse original authored USD/Mimic/actuator settings; author only, then create exactly
   one left and one right Articulation with actual wrist_fixed roots.
   Bind native prim_paths to env IDs through permutations, not lexicographic assumptions.
   Every hot-step write/read/update runs once per side. Selected Reset preserves peer
   composer buffers and native targets. Oracle: original two independent Articulations.
2. Tensor guard/reward owner
   Keep full original hulls and FP64 mathematical bounds on GPU. Negative coarse bounds
   and near SDF decision/lattice boundaries use original exact CPU algorithms.
   Preserve every 480Hz failure check. Replace per-lane RewardEndpoint construction with
   a tensor state machine, comparing SMOKE/FingerGoal and full reward arithmetic to
   original TrainingReward. Unqualified contact fields remain unavailable.
3. Tensor rod bridge
   Submit all mesh targets once through existing CUDA-array interface ABI.
   Build isolated device pair-view extension without changing physical kernels, and
   reduce original contact samples on GPU. Preserve index/normal/force/duplicate/overflow
   and native total reconciliation. Cache immutable actual material parameters at
   startup and each Reset; retain solver time/error checks every substep.
4. Tensor inputs and rollout
   Batch common66 and frame-normalized rod41x13 exactly as original feature pack.
   Store owned observations/actions/logprob/value/reward/done on rollout device.
   Preserve full371 ledger unknown masks, source freshness and GAE/terminal semantics.
   Exact KL uses whole flattened rollout.
5. Integration and evidence
   First CPU/device parity, including selected Reset isolation and fault rejection;
   then N2 real native regex/coupling proof; then N128 2-window same-cohort timing.
   Only actual timing proves the latest ~0.2s/window and 6–8s/update targets (<=10s requested upper bound). Then actual N128 *32 windows >=2 updates.
   Distinguish wiring updates from successful contact/Lift/Hold. Pinch/Tripod/Power
   have independent configurations/checkpoints and separate native qualification.
   Publish each tested package through Windows identical-commit import; no system
   upgrades or raw assets/models/logs uploaded. PR2 remains Draft.
