# CPU scheduler fixtures: identity-preserving injection and isolated evidence

## Failures reproduced
The production scheduler has two legitimate __main__ guards: module identity registration and the CLI main call. Nineteen fixtures counted one marker and globally replaced it, failing before their private child ran. The first repair produced 105 passes and 10 failures, retained in local logs.

Those ten success branches then reached the real archived pair derivation before their intended fake audit. A changed archived result SHA rejected them. Some rejection branches had previously passed for this unintended reason. Production locked-evidence gates and the archive must remain unchanged.

## Actual changes
tests/scheduler_cli_fixture.py uses AST to insert a CPU stub immediately before the unique top-level guard that calls main(). It preserves the module identity guard and rejects missing/ambiguous CLI entries. Function/class/lambda bodies are not treated as immediately executed guard calls.

tests/cpu_audit_fixture.py modifies only copied temporary test modules. It preserves scenario_scope and replaces their archived load_pair with a project/path/config-SHA-bound local fixture loader. It rejects .git files/directories and its own repository before writing; path escape and changed config identities fail closed.

Ten fixtures now instrument their intended CPU audit. Each verifies the exact accepted/rejected reason, an audit-call sentinel, audit-before-receipt ordering, native_execution_performed=false and qualification_scope=cpu_fixture_only. Synthetic results never enter the real native success pool.

Twenty-three test/helper files change. No scheduler, production archive, source asset, physics, model, limits or native admission gate changes belong to this package. The unrelated native_source single-guard branch in test_dh116_pinch_scheduler.py remains unchanged.

## Fresh integrated verification
124 passed, 0 failed, 0 skipped in the main Ubuntu worktree: 77 complete lifetime cases, 34 explicitly reviewed candidate parameter cases, 4 CLI helper cases and 9 audit helper cases. Exact argv/environment/log SHA are in readiness. CUDA_VISIBLE_DEVICES is empty; native/GPU/PPO runs are zero. This is not a full-repository or native qualification claim.

Red/green logs and intermediate failures are retained locally. The agent's independent 124-case run and the primary agent's fresh integrated run are recorded separately. All 528 protected hashes and the three original models are verified before commit.

## Remaining gate
Actual native Tensor View ownership, natural process exit and contact qualification are still unproven. A fresh frozen proposal and authorized budget are required for the next two-SDK-step observation. No old spent native budget is reused. Teacher admission remains blocked; Lift 30mm, Hold 5s and fixed-anchor Slip 0.5mm are unchanged.
