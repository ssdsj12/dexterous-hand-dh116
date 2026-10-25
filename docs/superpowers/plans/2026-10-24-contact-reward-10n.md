# Contact reward continuation, 2026-10-24

User requested total measured pad force 0..10 N mapped proportionally to
0..5 points. Preserve progress shaping: reward is the change of that potential,
not an absolute payment at each physics substep. Loss of contact reverses the
corresponding reward; forces above 10 N saturate. Other rewards and physical
limits are unchanged. This does not require four simultaneous pad contacts.

Implementation: share the potential between scalar and batched tasks; identify
the new reward as `object_support_progress_v4_10n_5` in all task entrypoints.
Continue the previous policy/optimizer and per-environment episode statistics
using an explicit reward-only checkpoint transition. Reject all other contract
differences, including physics changes. Retain the original checkpoint and run.

Verification: exercise actual scalar/batch transitions over total loads
0, 0.5, 1, 5, 10, 12, 5, 0 N; expected increments are
0.25, 0.25, 2, 2.5, 0, -2.5, -2.5. Verify ordinary incompatible resume rejection,
explicit reward-only transfer, counter/optimizer preservation, and existing
scalar/batched safety and training regressions. These are software tests, not
evidence of physical 10 N capability or successful grasping.

v64 stopped via SIGINT and saved 1710592 transitions, 256 optimization epochs,
and 9 completed episodes in each of 512 environments. An unfinished episode
restarts on resume, and unoptimized partial-rollout samples are not replayed.
New training retains the per-environment target of 1024 episodes. Compare
physical success metrics across reward versions, not raw episode returns.
