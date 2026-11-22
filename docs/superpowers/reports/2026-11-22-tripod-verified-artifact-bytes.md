# Tripod: parse exactly the checksum-verified artifact bytes

## Actual change
tabletop_tripod_reset_manifest._derive previously checked every inventory SHA then independently reread10 consumed JSON paths. The latter bytes could differ from those checked. It now retains only these10 checked-file return values within this invocation and pops them when parsed. All26 inventory files still verify on every load and every verify; other inventory bytes are discarded. No cross-call manifest cache, geometry/SAT skip, threshold change or claim of atomic multi-file snapshot is introduced.

## Regression and review
Extended the existing real-manifest owner test; no duplicated scene or new production test seam. Sealing finishes before the spy. load must freshly check every inventory file once,verify must check every file a second time, and parsed paths cannot be read outside checked_file. Pre-fix source executed in-memory fails for artifact reread after SHA verification (1 failed). Final six-file regression:98 passed,0 failed in129.64s.

An intermediate broader regression was68 passed/1 failed because the already-spent v10 recipe correctly rejects changed source. Its log is retained. The historical worker test now verifies Registry companion SHA -> core SHA -> training config SHA, retains recipe schema,physical manifest,profile,target,budget and real CPU launcher checks, and explicitly expects stale-source rejection. No historical recipe/fingerprint is refreshed. Current actual-launch controls remain independently covered.

Independent read-only review approved final source/test SHA versions. Antigravity remained quota-limited; no new external review is claimed.

Command: /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_tripod_reset_manifest.py tests/test_native_teacher_tripod_admission.py tests/test_native_teacher_static_manifest_handoff.py tests/test_native_teacher_tripod_frozen_worker.py tests/test_native_teacher_actual_launch.py tests/test_dh116_work_packages.py

## Whole-path measurement
Three alternating baseline/current complete _derive calls on real SHA-bound data, same CPU process/environment: median 8.462857s -> 8.458621s. Every returned field equals baseline; no invariant is dropped. Result SHA:4664764c5ff9fcfaf4ec522e1b778eb6116e62303430aebf43b63086e59785bc. The difference is within measurement variability: no material whole-path speedup is demonstrated. This repairs byte provenance; native startup savings are unverified.

## Parser experiment, not production
The existing orjson3.13.0 parses the198718823-byte real record faster in a separate CPU probe (median parse2.3744s ->0.7709s,parse+decode+digest3.0602s ->1.4800s;decoded digest equal). However an independent semantic probe confirms differences for integers outside64-bit,NaN/Infinity/overflow and lone surrogates. It is not installed,upgraded or substituted into production. A future fast path needs explicit compatibility proof; single-file parity is insufficient.

## State and next work
0 native runs,0 PPO updates and0 grasp qualifications on this code. Tripod v10 remains spent/failed with its original7-controls/0-update evidence. Pinch/Tripod/Power still need separate qualified training and full held-out evaluation. R2,Force Proxy and formal BC/DAgger remain gated. Fixed two RGB cameras,24D/371D,original assets/physics/Reset and Lift30mm/Hold5s/Slip0.5mm are unchanged.
Next: measure complete parser eligibility and initialization cost, then freeze a distinct finite recipe only after the expected path fits its authorized budget.
