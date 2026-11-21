# 902 staged approach: preserve terminal grip, open before wrist transit

Baseline:29c41609f599171000ae47ad2ec60cf9c2ce1a36, branch11-17. This is an executable CPU command compiler and independently configured candidate, not a frozen native recipe.

## Implemented behavior
scripts/compile_dh116_902_staged_approach.py::compile_candidate consumes the original66×24 right-hand approach and measured boundary command. It produces16 middle-opening windows,66 wrist-motion windows and54 fixed-wrist closure windows. The middle finger temporarily opens another0.02 rad; the closure phase returns that extra opening and preserves the original902 terminal grip. Initial/path constraints may differ; the terminal target does not.

The other hand's twelve action coordinates remain zero. During wrist motion every finger action remains zero; during opening/closure all wrist actions remain zero. Numerical float32 correction applies only to intentionally moving fingers and remains bounded by1 microradian. Every nominal waypoint and all8 float32 substeps per window pass the existing linear_coupling.expand active/mimic limits. It neither clamps illegal targets nor edits joint limits.

Wrist source actions and compiled translation/rotation vectors must have L2 norm≤1, in addition to component bounds. This matches the original wrist_requested_targets/limited_step/palm_transport behavior and prevents a shorter schedule from silently changing its terminal pose through vector normalization.

configs/dh116_tasks/flexible_902_tripod_staged_cpu_v1.json is an isolated CPU schedule referencing the unchanged v2 target request by SHA. It is not a duplicated physical scene, new USD, reset configuration or training environment. native_admissible and teacher_admission remain false. The output explicitly requires measured wrist settling and opposed contact with delivered reaction to count closure success.

## Actual validation
Failure-first tests captured missing implementation, an over-speed9-window expansion fixture, and four diagonal wrist cases that previously escaped per-component checks. The nine-window fixture was corrected by allocating16 windows; the original0.3 rad/s rate was retained. The four wrist cases reproduce both unsafe source vectors and valid vectors made unsafe by shortening the movement phase.

Sixteen new tests pass; final related CPU regression passes234 tests,0 failures. The full-path harness reuses original command_step, source/cooked geometry, nine archived rod snapshots, support/table guards and both hands. It is an ideal-tracking CPU replay, not a simulation measurement. Actual path results, elapsed time, endpoint errors, final compiler/candidate SHA and all retained evidence SHA are in readiness.

The final compiler was rerun after the vector-limit fix. All1088 ideal command substeps pass existing geometry guards. Final target errors remain about1.14e-8 m position,7.21e-8 rad rotation and2.16e-7 rad joints. For wrist motion, the contact-envelope lower bound improves from−0.130631 mm in the prior staged schedule to+0.182355 mm. This is a conservative lower-bound comparison, not force evidence. Source/cooked guards, original action rates and final902 target were unchanged.

## Antigravity discussion
R4 read real Ubuntu HEAD29c41609 and compiler SHA749c64f2… / tests SHAa326f8a5…, matching independent SSH. Its connector cannot read the new configs JSON; Codex checked it directly rather than claiming independent config review. Antigravity identified the missing vector-norm guard. Codex confirmed this against tabletop_command_kernels.limited_step and the original wrist projection, wrote four failing regressions, and fixed both source and compiled vectors. The final changed SHA is recorded in readiness; no claim is made that Antigravity reread that final patch.

## Remaining physical gate
The previous native attempt remains failed atSDK476, with zero hand wrench and natural exit1. This package adds no Isaac run, solver step, Teacher query or PPO update.

An ideal gap of0.182 mm does not cover the earlier approximately1.14 mm measured wrist/link tracking discrepancy. A runtime must control measured error DURING wrist transit, not merely wait at its endpoint; then it must verify bounded same-clock wrist pose/velocity settling before closing. Fixed-wrist closure must respect live source/cooked/contact guards and stop on real opposed loading plus delivered reaction. Reaching the original joint pose alone cannot authorize Hold or Lift.

Do not run this candidate as native merely because the CPU path passes. Do not copy this Tripod/right intermediate path into Pinch or Power. Their final grip identities and independent qualification remain separate. Both fixed RGB cameras remain byte-identical; Teacher/Student visual alignment is preserved.

528 protected files and three historical checkpoints are verified before commit. Raw geometry, trajectories, logs and models remain private. Publication uses Ubuntu commit→Windows exact import/push; PR#2 stays Draft.

## Executed command and identities
Full replay: existing Ubuntu CPU environment + /mnt/mydisk/xk/envs/dh116/bin/python -B <private>/ref902/command-v3-extra-opening-path-final.py.
This private bounded harness calls the published compile_candidate with the new config and reuses the unchanged original command replay and hand geometry validator. Runtime129.051 s, exit0. Proof SHA f8d457e16bbaf245c51d73bf063d038fa5ad5da0ad80d7ab61039dd8209e630b. Compiler SHA 29607bff39add4a448134f70171f13e1c926b7d204ffb047aa3a759bb77d8b62. Candidate SHA be408ba17a470fe2c752252559ac3b636ee0e9a3a7a08cf9b278065784174ed8. Action array SHA eb07f81c7fdfb5270f4a8e5c0a2543abc1c14317a005f4e6f175c5da14a1f5fb.
The guard-only vector fix leaves all1088 geometry samples identical to the preceding run. Final maximum wrist action L2 norm0.9978136618770699.

Registry regression60 passed,0 failed; combined package CPU checks294. Log SHA 8a65a7db8806417aa38667ac6a3469346e36ad9a1e6712f7f840e383d78343ee.
