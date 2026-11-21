# SDK240 supported Reset to guarded Tripod contact diagnostic

Base 3d704bc26b7614cf98426772a07ea9a6020a66d9, branch 11-17.

This package connects a real closing run to the independently verified aligned-wrist Reset. It does not train or query a Teacher, load a policy, lift a rod, or grant expert/BC/DAgger admission.

## Implemented execution chain
certify_dh116_902_joint_contact_path now separates the immutable original search receipt (5681a8...1022) from the actual aligned-wrist execution receipt (655a24...add2). The new execution boundary must reference the same source-pinned candidate and pass a freshly reproduced saved-data actual clearance audit. Original default source behavior remains separate. New v2 proof recompiles 64 closing windows plus 16 exact-zero feedback windows and rechecks all 640 physical substeps against real archived material.

dh116_902_joint_closure_contract validates the exact actions, held axes, original side stations, every substep, final three-pad geometry, authenticated F4 cooked vertices and the actual sealed SDK240 settling row. The legacy independent_entry routes only its new explicit version to this validator; old entry types remain unchanged.

SupportedDistalProbe.prime_supported_boundary uses a decoded actual zero-hand-load SDK240 settling sample, checks same-run identity/clock/preceding reaction, and sets only the prior rod boundary needed for check_reaction_delivery. It never invents an action/control sample. run_shared passes power.clock() at SDK240; setup.clock() is still SDK2 and cannot be used. The existing exact physical replay boundary check, measure_links, continuous source/cooked envelopes, SDF checks, station checks and next-step wrench delivery remain active.

verify_dh116_902_joint_closure performs fresh standalone-USD proof generation in a CPU child before Kit import, freezes both receipts, audit, source geometry, boundary, actions, SDFs and all dependencies, and calls the existing reset owner/shared physical action path. It distinguishes complete natural cleanup from internal failure/incomplete records.

## Found and fixed representational held-axis bug
With a measured float32 opening and its original float64 textual target, finger12 had 64 nonzero commands of approximately 1e-9..6e-8 despite an identical representable target. A regression reproduced this. compile_closure now compares exact float32 targets to select moving axes, so equal backend targets issue exact zero. No epsilon, physical limit, collision margin or reward threshold is relaxed.

## Validation and scope
New behavior tests were red before implementation. Full related regression: 242 passed, 0 failed, 75.78s. Final narrow regression after optional hash tightening: 58 passed. Seven new cases include held-axis regression, dual source identity, actual boundary priming, stale/current clock, hand contact rejection, new contract strictness/zero tail and incomplete native outcome (some conditions share a test).

The real main() CPU preflight succeeded. Its pre-publication plan f840d104... is CPU-only and must be superseded after publication. Fresh v2 proof has 80 windows/640 substeps, initial/path/endpoint geometry all passed, action SHA fca43517d288e4b98e504bf17c2618384b3dbff3ac83506a2d0ceb490b5d49e3. No actual new native run at this source commit.

Antigravity R1-R2 read the seven production files and found no blocking issue. Codex corrected the review's setup.clock suggestion to actual power.clock SDK240. Two optional final expected-hash entries for actual audit/boundary were added and tested after R2; final entry bytes were not reread by the reviewer.

## Frozen native scope to seal after publication
GPU2, one Isaac/environment, 2 initialization + 238 settling + 640 controlled SDK steps = 880. Solver substeps 10536, control windows 80, Teacher/model/PPO counts zero. The last 16 windows are exact-zero target holds lasting 0.2667s solely for contact/reaction diagnostics; they do not satisfy the original 5s Hold. Inner/outer soft walls 480/600s, no retries or automatic kill. The previous 228s partial historical control run informs this bounded wall budget. No assets, mechanics, frequencies, 24D/371D contracts, cameras, old models or success standards change.

Success of this diagnostic means complete guarded records and natural exit. Actual contact/opposition/force delivery must be read from the new ledger. Zero contact, guard termination, unexpected station, failed shutdown or missing rows remain explicit failures/limitations, never Teacher success. Raw native logs and models remain outside Git.

Registry regression: 60 passed, 0 failed.

## Launch-context correction (actual pre-Kit rejection)
The first attempt at published HEAD 1a1ed3e5c27323bb52ed45a64ac54a3b36ed3182 froze 0ad0f0a7b728f56ee2a7a485da71a8475d33836b99bb37cb1e3d9960645ce1a3. It exited 1 after 53.456s BEFORE loading Isaac. All source hashes were unchanged; no native output, SDK/solver/control/Teacher/PPO steps, crash or forced termination occurred.

Codex's private launcher had prepended ROOT to PYTHONPATH even though the frozen environment already contained it. Two real CPU subprocess identity reads isolated pythonpath as the only changed field: original env equals the frozen identity, prepended env does not. Antigravity independently read the actual entry and interpreter_identity and confirmed the cause (902-native-launch-context R1). The launch failure is not a physical/contact failure, and is not a qualification.

The private launcher now preserves the original environment and asserts the complete current identity before starting GDB. The project entry also checks approved JSON SHA, exact execution identity (with differing key names), and fresh output BEFORE the costly proof. Its original full rebuilt-plan/fingerprint/output guard remains after proof; no identity normalization or bypass was added. A new entry-boundary regression failed in three branches on the old code, while the matching identity positive control reached proof. All 62 related cases pass. An actual wrong-env entry subprocess now rejects with execution context differs: pythonpath in 2.946s, without Kit/native output.

Old attempt/receipt remain immutable outside Git. The old fingerprint will not be retried. Publish and seal a new corrected-version fingerprint before the next single bounded physical diagnostic. The diagnostic budget and all physical/success gates remain unchanged.

Launch correction Registry regression: 60 passed, 0 failed, 14.83s. The original 242-case physical-helper regression is retained separately from this 62-case entry regression.
