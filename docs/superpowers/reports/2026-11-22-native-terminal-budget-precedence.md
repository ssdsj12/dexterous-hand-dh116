# Teacher terminal budget precedence and Tripod v10 failure

## Observed failure and smallest repair
Tripod v10 on e072ca823d5d4a37b7d2e010563b129af0cdb3e5 delivered 7 controls /58 SDK steps (2 setup +56 measured),56 solver calls /672 internal substeps,0 PPO updates,no checkpoint. External wall exhaustion truncated settlement seq269; seq271 then misleadingly requested fresh Reset factories.

run_updates now checks the existing owner.contract.runtime.remaining_steps before asking for any terminal-episode Reset. It does not use the expired episode's remaining_substeps as a session predicate. Absent a session runtime, the original legacy factory requirement remains. No partial PPO window, extra Reset, budget increase or retry is introduced.

## CPU evidence
The added three-case owner-boundary regression uses the real SessionBudget charged by the existing CPU collector. Exhausted session with/without factories stops; short episode with session budget remaining still reaches restart. The source-based intermediate guard failed the last case:1 failed /2 passed. Final training,collector,actual-update and lifetime regression:69 passed /0 failed in156.31s.
Command: /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_native_teacher_training.py tests/test_native_teacher_collector.py tests/test_native_teacher_actual_update.py tests/test_native_teacher_actual_cache_lifecycle.py
Registry:60 passed/0 failed in15.37s;129 distinct CPU tests total. Logs and hashes are in readiness; original failed logs remain outside Git. This is CPU validation,0 native runs on the new code.

## Historical native outcome
The original v10 receipt,outer result,log and four native evidence files were rehashed independently (7 artifacts). Outer/inferior exit1,no signal,196.4262s; App.close returned,14 tracked weakrefs died and one stat-cache release left0 entries. This is a failed budgeted experiment, not natural-exit0 qualification or grasp success. Readback SHA:1350ae70e9dea385863cf2122b887990280164a01bd030f7f9d2e3a479bfe6e2.

Seven measured control windows took24.3226s total (observer8.8699s); only22.1005s remained before update. The spent v10 fingerprint is not rerun. No model enters any success pool.

## CPU resource evidence
Three full static-contract runs per mode, same output signatures: numeric1-thread median15.1603s,4-thread13.2489s,64-thread12.2417s. These are complete independent source checks, not native timing attribution. Small-packet gains cannot justify a global1-thread recommendation. Preserve all independent manifest/source/geometry checks while reducing redundant input reads.

## Review and constraints
Independent read-only reviewer found no blocking issue in the final session/episode distinction; legacy regression remains. Antigravity's live native UI still shows Individual/Baseline quota reached and refresh2026/11/28 12:49:46; no new Antigravity opinion is claimed and no paid setting changed.
Original24D/371D,528 protected files,3 historical checkpoints,fixed stereo cameras,material/physics/Reset sources and Lift30mm/Hold5s/Slip0.5mm remain unchanged. R2/Force Proxy and formal BC/DAgger remain gated.

## Next work
Audit checksum-verified artifact ingestion and measure full initialization, then freeze only a meaningful new finite recipe. Pinch bounded-update evidence,Tripod failed smokes and Power preparation remain distinct; none establishes three-grasp qualification.
