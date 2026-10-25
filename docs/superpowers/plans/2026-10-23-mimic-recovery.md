# Direct-close mimic recovery

User request: resolve v22 right index mimic instability and continue PPO. Existing
rigid-proxy, finger-target, palm-controller, no-four-contact design is unchanged.

- [x] Replay exact v22 episode 5 offset/actions in serial Isaac. Failure reproduced
  at 2.575 s, right finger22 error -0.020175296172618845 rad.
- [x] Isolate causes. Passive velocity .3 -> 3 rad/s produced peak .0188141857;
  follower-consistent master recovery alone produced peak .0084474299 through
  the same prefix. Choose recovery, retain original physics and velocity limits.
- [x] Add red/green unit test and implement q_master + error / ratio recovery,
  named mimic mapping, coupled position limits and existing .05 rad/s command
  rate. Preserve .004 warning and .02 hard guard; update checkpoint contract.
- [x] Full replays and fresh PPO completed: recovery-only v23 failed at3754;
  add passive solver limit3 after isolated replay, v24 completed8192 plus5 safe
  evaluations. Grasp0/5; continue learning with v25 supervisor.
- [x] Review and186-passed/1-skipped focused checks, compile/diff, evidence docs.

The old warning target was measured q_master, which did not eliminate displaced
mimic constraint geometry. Correct recovery derives the master target from the
measured follower. It changes the actuator target only, never measured joint or
object state. Solver velocity clipping was confirmed as a second contributing factor on the
v23 action prefix; direct passive limit3 is now deployed, active limit unchanged. All runs use CPU0/one numerical thread, one Isaac.
