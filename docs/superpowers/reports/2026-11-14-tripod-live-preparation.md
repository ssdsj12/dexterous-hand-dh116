# Tripod original-action native preparation — 2026-11-14

Base: `cf0f65f2978db2f8a63e48b1bbc21c217b2f35e0`. Independent worktree/branch; all changes are new opt-in files. No original Pinch controller, asset, material, actuator, action/observation contract or historical experiment is edited.

## Implemented runtime path

- `prepare_dh116_tripod_owned_approach.derive_pair/load_pair/action_schedule/expected_boundary`: freezes the previously audited **independent left Tripod** pose and original 24D, intrinsic-XYZ approach. It retains the right hand's independent original USD/SDF/link/reset identity and zero right action. This does not qualify right Tripod.
- Reset SHA: `b9c06c8d21855ee3a187a269d718905062bda2b0455e6bd5f3f2cef2827b829e`.
- Left start WXYZ pose: [0.5319283577621972,0.13489669694253587,0.3847989429225563,-0.40200419197817483,-0.42156529584810926,0.6125166379123426,0.5343207830966631].
- Left end WXYZ pose: [0.5337131957393896,0.1384745111335029,0.3844794128948734,-0.4081791671129427,-0.4692063685461108,0.5879675397388606,0.5172323689254057].
- Original action limits retained: finger0.3rad/s, translation0.01m/s, intrinsic rotation0.5rad/s; held8substeps per60Hz policy decision,480Hz SDK.
- `TripodQuietGate` and `TripodExchange._step`: require **this instance's** first8 measured endpoints before any approach. The added diagnostic admission thresholds are rod speed0.1mm/s, drift from reset10µm, angular joint speed0.001rad/s and carriage linear speed0.1mm/s, all96 solver substeps table-loaded and no hand load. These are not new physics, grasp-success thresholds, or a5sHold certificate. Original guard failure has priority; incomplete sources remain unknown. No retry/extension is possible after terminal.
- `TripodOwnedGeometry`: exact requested/projected/submitted wrist target and original24D schedule validation; left middle `finger33_link` is added to required pads. Its measured SDF/gradient/lattice guard replaces only its former non-target rod-separation classification. Every source17/locked5 table check, all other non-target checks, right middle classification, original tracking/force/joint/mimic protections remain. Whole-hand source/cooked separating planes are checked. No SDF/geometry prediction is reported as actual contact.
- `inspect_quiet` recomputes saved admission receipts from actual reset and raw physical source rows. Independent director validation retains all old checks and explicitly accounts for the additional quiet terminal layer. The post-exit validator differs from the published original only in the independent operation import.
- Dedicated worker, scheduler wrapper and recipe bind the **actual** published bilateral parent receipt (`8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8`), its artifact SHA values, new source/config fingerprint and exact budget.

## Frozen budget and evidence limits

One new run only, GPU3/singleIsaac/oneenv/0PPO, maximum180s:
**2 setup +8 quiet +200 approach +24 observation =234 total SDK steps**,232 measured ledger rows,29 policy decisions,2784 solver substeps if complete. The24 observation steps are50ms, not5sHold. The source-scoped recipe is a contact diagnostic, not Teacher training.

The CPU original-action reference remains `f0726bb65f7ed3d2e645fb9d93226dd0e83f2100ee1cc928c95fa60fc74156ae`. Its smallest strict cooked-hull path margin is0.444µm and assumes perfect target tracking. Final reference pad SDF gaps are positive: thumb22.690µm/index5.889µm/middle42.850µm; these do **not** establish contact. Live checks can reject the candidate.

The original task has free rod nodes, gravity, a real table and unmodified collisions. Table support at rest is expected; this experiment cannot certify unsupported lift. Hand-table native contact source remains explicitly NA where the static-tensor source is unavailable.

## Failure-driven CPU verification

Initial geometry tests:16 failed before implementation. Quiet exchange tests:8 failed before implementation. Worker tests:10 failed before implementation.

The first full-owner CPU run stopped at SDK160 with `left:wrist_transport`: the inherited CPU fixture did not move the external wrist carriage, accumulating3.008mm requested-vs-actual position error. Its raw fixture ledger and log were retained. **No production guard was relaxed.** The fixture now evaluates the original measured prismaticXYZ plus intrinsicXYZ six-joint chain, while keeping the anchor fixed. The full234-step Python-owner test then passed. An independent moving-rod quiet fixture stops at SDK10, preserves all8 terminal physical rows, prevents the first nonzero action and passes replay; editing only its saved admission status is rejected.

All fixture evidence is synthetic CPU integration, not Isaac, measured contact, natural native-process exit or Teacher success. Frozen-worker tests block early USD/Isaac imports, expanded budgets and unmarked launch.

CPU regression: **109 passed,0failed,0skipped**,191.99s. LogSHA: `1c07fc95519058647f8c8c65fc4c9ae061d453eafd9ab6e96a44c0f029863b58`.

Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_live_plan.py tests/test_tripod_quiet_gate.py tests/test_tripod_projection_scope.py tests/test_tripod_exchange_admission.py tests/test_tripod_native_scope.py tests/test_tripod_full_owner.py tests/test_tripod_worker.py tests/test_dh116_bilateral_owned_contract.py tests/test_dh116_bilateral_owned_geometry.py tests/test_dh116_bilateral_owned_worker.py tests/test_rod_quiet_window.py`.

Private evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-live-preparation-20261114`.
Raw fixture ledgers remain outside GitHub. The original source, physical and success contracts are unchanged.

## Remaining work / readiness

This package admits **preparation only**. Execute the new recipe once under standing bounded authorization after publication and resource checks, then independently inspect all physical rows and outer exit/GDB evidence. No old budget is reused.

Pinch's prior contact execution does not qualify Tripod/Power. Teacher remains blocked on actual per-grasp contact/Reset plus complete qualified371D/R0 sources: existing common-state readback is66/371, and material slip/hold/instability qualification remains unresolved. No PPO update,5sHold or grasp success is claimed here. Power still needs its own usable actual-SDF contact path.
