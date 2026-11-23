# Tripod 902 guarded approach runner

The opt-in entry `scripts.verify_dh116_tripod_902_approach` reproduces the original supported 902 seed and all 39 planned controls through SDK 552. The restored source emits the same 39 original commands reconstructed by the independent earlier check. The two additional eight-step commands come from the sealed R5 counterfactual. The existing SDK 272 waypoint, SDK 432 precontact and SDK 528 right distal probe stay in the execution path.

At SDK 240 the worker checks the parked fixture with the original parked-schema comparator, then checks the exact support pairs, reaction route and rod clocks. At SDK 552 it checks original measured link geometry and same-clock hand readbacks, support pairs, wrist projection and issued wrench. Either mismatch terminates before the next command. All new rows require synchronized ticks, intact guards, and load at both original supports on every solver substep. Any hand or table contact stops this precontact experiment. These historical rod rows record node positions and velocities. They lack per-node quaternions for an equality test.

Antigravity R9 read the original fixed checkout and identified the two existing acquisition call sites. The current integration consumer already differs from that checkout. The implemented route was independently checked against its current source, and the existing distal probe is active at the later interlock. There is no interpolated pose or torque signal.

## Verification and limits

Use `bash scripts/python.sh -B -m scripts.verify_dh116_tripod_902_approach` with the original Tripod archived-settling, approach, waypoint, precontact, distal, palm and proximal dependencies, plus the sealed R5 command draft and original outer receipt. Pass `--verify-existing-plan` through the same launcher and GDB for an execution-free verification of an existing freeze. The optional native path requires the exact digest and a new output directory.

The Ubuntu launcher CPU preflight verified 2,248 files, with GDB inferior exit 0. The focused and shared runtime regression passed 121 tests. The first invocation without `DH116_PYTHON` failed before Python due to the legacy default interpreter path. The successful invocation used the existing `/mnt/mydisk/xk/envs/dh116/bin/python` and retained the failed log.

The plan budgets 1 environment, 1 Reset, 2 initialization SDK steps, 238 parked SDK steps, 41 control windows, at most 568 total SDK steps and 6,792 solver substeps. It has 0 camera captures, 0 model loads, 0 Teacher queries, 0 PPO updates and a 420 s soft limit. This package used no Isaac launch, acquired no native result, and makes no grasp, Lift or PPO qualification claim.

| Local evidence | SHA-256 |
| --- | --- |
| Historical outer receipt | `5306afb0af3892b0da3c2edf8c3f5560e7bf5798578b26fe3060f681417f6b06` |
| R5 command draft | `dbbd67046b5dd3b051c84b9ae24710b7ab123d7e115f57d17b9ccac311db196a` |
| Integration worktree freeze fingerprint | `681b503ae2a998c269c2098770df930bf43f4a32ac47f021316b4bde8e365d48` |
| CPU GDB verification log | `299ac5ba42db793db9b9ec0729670a614cc6a441095e3282740e6bb6dcc8499f` |
| CPU regression log | `5680b74322aebb797f1d6c362c2f666cb640692014a333d3ebaefd862d3aff99` |
