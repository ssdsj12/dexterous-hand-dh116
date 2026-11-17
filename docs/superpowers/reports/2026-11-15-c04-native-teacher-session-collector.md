# C04 native Teacher session and actual-control collector CPU candidate

Status: software CPU candidate after the independent SPEC remediation below; C04 SPEC/QUALITY re-review pending. No native execution, GPU, training launch, installation or push occurred. Development is solely Ubuntu `teacher-c04-session-20261115`, branch `codex/teacher-c04-session-20261115`, based on reviewed integration `a6a4290407456b4ea2d707e27272885b74e95bc6`.

## Implemented path

- `NativeFactories` and `NativeSessionConfig` define explicit construction inputs. Static source/manifest/budget gate precedes owner creation; the original Reset plus C01 live verification precede model, original PPO trainer and loaded reward construction. Default old actual admission uses the same new source gate, and old three keyword factories remain accepted with an optional explicit independent config.
- `NativeEpisodeOwner` leases an existing original `DualHandPhysicalEpisode` and borrowed scene. Reset uses scoped existing measured mesh/numeric readers and unchanged clearance successor; no second scene/App/physical owner is created. Failures preserve their original exception, close the lease once, and durably retain cleanup errors with a separate journal fallback. Lease close restores callbacks and drops borrowed references; SDK STOP/native release/App.close remain the outer R06 text-only-failure/clear-frames lifecycle's responsibility. These CPU tests do not prove natural native exit.
- Native Pinch smoke source adapter calls the existing current-profile `NativeParameterEndpoints`, the new independent `PolicyOwnedPinchApproach` for policy intervals, original clearance validators and the same-owner solver-health getter. `BilateralMeasuredPinchApproach` is retained only for C01 Reset verification; its frozen diagnostic policy schedule is never used for PPO windows. Reset and physical intervals use C01 `profile_reader_scope`; historical 2fea defaults remain restored. Distance uses the fixed 41 material IDs and Pinch required pads. Context stores reset d0, rather than replacing it with current distance. External remaining time comes from one session-owned monotonic budget clock, with same-clock ContextIdentity sidecars.
- Each decision invokes original `episode.control(sample["actions"])` once. Each advanced raw exchange is fsynced before original boundary conversion; the complete physical event is then fsynced before action-history/reward validation. No added sim.step is present. The exchange observer forwards a trusted stop immediately. All request, SDK target, token, policy weights and clock checks use original action/PPO validators.
- History capture precedes context freeze; actual control precedes history settlement, post-context, issued-for-pending, pre-reset next value, original GAE boundary, record_step, durable transition_recorded, then context commit. Buffer full keeps done=false and retains the physical episode/anchors. Update delegates to the unchanged original PPO; quarantined/incomplete rollouts cannot update through the collector.
- Shared SessionBudget separately charges setup SDK steps, actual solver calls and internal solver substeps. Discarded transitions do not return budget. Fewer than eight available steps do not sample or start a window. Successful termination/external truncation alone permit a fresh original episode/history/context with the same trainer/rollout.

## Approved scope expansion

`tabletop_native_teacher_sources.py` supplies the concrete Pinch adapter. `tabletop_native_teacher_action_history.py` is an explicit v2 type, not a patched v1/global or forged setup=2 clock. It reuses original policy/token/request/target checks, and adds actual no-step Reset-bound sequencing at arbitrary absolute SDK step. The addendum records parent approval and the old-owner failure at N>2.

The first combined regression produced **141 passed, 30 failed**: the old CPU fixture source lock pinned the entire old no-camera source and its admission test. With parent approval, `tabletop_cpu_fixture_owners.py` now has independent VERSION v2 and only two updated fixed SHA entries (the modified no-camera module and its modified admission test). All other source locks, topology and validation logic are unchanged. A frozen AST hash from base a6a429 proves original CPUFixtureCollector, journal, observer, CPU helpers and constants are identical. Tampered temporary copies of either newly locked file still fail. The targeted source-lock suite then passed **63 tests**.

## Evidence and scope

Private evidence: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-c04-session-20261115`.

- `session-red.log`: initial absent new module/API RED (not claimed as a semantic source exploit).
- `cleanup-semantic-red.log`: diagnostic cleanup failure incorrectly replaced the primary exception; corrected and covered by GREEN.
- `collector-red.log`: initial collector fixture/clock boundary RED, subsequently corrected.
- `reset-history-red.log`: old v1 rejects actual SDK10 reset, new explicit v2 initially missing, then new reset-bound test passes.
- `combined-prototype-green.log`: 141 pass / 30 fail source-lock root cause described above.
- `source-lock-green.log`: 63 pass after the fixed source-lock revision.
- `combined-green-final-prototype.log`: **176 passed** in 96.51 s for session, collector, old CPU fixture/owners/device, original action history, C02 context/model.
- `reviewed-c03-green.log`: the same combined suite after syncing exact reviewed C03 fee8de1, **176 passed** in 95.96 s.
- `protection.log`: original scheduler `Repository.verify_protection()` PASS for **528 source/script/config files plus both original checkpoints**. Checkpoints remain ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511 and c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999.

Canonical verification uses `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest ...`. No installation or direct missing-pxr workaround was used.

The three-episode test uses original software with CPU SDK/rod doubles: eight-step external truncation -> same-scene zero-step Reset at SDK counter10; three-step trusted joint failure -> zero-step Reset at counter13; another eight-step normal control -> counter21, synthetic rod.steps19. Same trainer and three stored transitions; reset IDs1/2/3, independent zero previous memory, no event/anchor leakage, absolute SDK/solver clocks maintained. These are **CPU fixture counters**, while **native_steps=0**.

Coverage includes third-step guard/no fourth step, third NaN/foreign identity, actual request and SDK target mismatch, token replay, cross weights, exact pre-reset V, terminal/bootstrap/carry semantics, owned tensors, same-episode PPO update continuation, source exception plus diagnostic cleanup failure, raw exchange survival when original boundary conversion fails, budget no-start and setup/solver accounting.

## Dependency accounting and remaining qualification

Prototype dependency commits are separate from C04: `dca9749` imports exact a249a391 C02 context/model and two tests plus 4a2e44 C03 reward/two JSON; `7050e2d` is only a temporary C02 exact-type allowance for new v2 history. These are not C04 deliverables and did not authorize native execution. `9c20d36` synchronizes the exact three now-reviewed C03 dependency files from `fee8de1b34294c294213e6e2eb77e9f8f826f40e`, individually byte-compared. C02 is now synchronized to the exact seven formal reviewed files in `712f7fcf3e3c0a6992951228071ba08ecf66762f` (same bytes as 12aeee74), in separate dependency commit `c65b14d`; `reviewed-c02-exact.json` records the byte comparison. No prototype dependency commit belongs to the C04 package.

Default production entry remains specifically blocked by missing independently authorized finite native budget/resource lock and resolved current Reset manifest. This package does not claim live Reset, native getter, cooked geometry, physical grasp success, learning, or natural shutdown qualification. The concrete Pinch adapter calls real readers, but those native handles were never executed in this CPU scope. A nonzero native error may also prevent the existing parameter endpoint reader from supplying the required same-clock distance inputs; that becomes retained SourceFailure rather than an invented reward. Full contact/slip/hold/evaluator-common adapters and independent Tripod/Power distance readers are explicitly absent and rejected; callers cannot add Boolean qualification or relabel grasp to bypass them. Original success constants remain .03m /5s /.0005m, and the PPO/GAE algorithms and protected physical/controller sources remain unchanged.

Only the C04 package paths listed in `candidate-files.json` plus this report should be extracted for review/integration. Do not integrate prototype dependency commits as C04 or interpret CPU success as native admission.


## Independent SPEC remediation (supersedes first candidate's actual-source claim)

The first 176-pass candidate failed two independent SPEC checks: the selected interval reader still required the frozen diagnostic action schedule/SDK<=234, and valid original guard rows with passed=false were incorrectly classified as SourceFailure. Those tests had only proved the collector with synthetic FixtureSources. These were actual C04 source defects, not C02 dependency failures. They are corrected in a separate local remediation commit.

New `tabletop_native_teacher_policy_pinch_distance.py` provides an independent version/source SHA, actual SDK pose/pad/SDF/radius validation, fixed41 distance mathematics and durable history-row/request/target/token/window binding. The legacy frozen reader/validators, original physical/history/PPO sources and original success constants remain unchanged. Source completeness and trustworthy constraint failure are separate. A valid original guard remains a usable terminal negative sample; corrupt/unknown evidence remains SourceFailure.

The `policy_pinch_distance` journal operation persists the full new source and history_binding after the physical row has been fsynced and verified by history. Current same-owner parameter packets and source inventory are checked; a caller qualification Boolean is never accepted. C01's old Reset-only reader has its own exact type and remains distinct from the new policy reader.

Actual adapter CPU source-interface tests instantiate NativePinchSources and run its production enrichment/observe functions, real SDK/rod doubles, original policy/history and real TrainingReward. Only unavailable native parameter/health getters are replaced with explicitly CPU-scoped packets. SDK body/41-node buffers are assembled from existing owned CPU fixture templates; no historical native/backend packet is relabelled. Reset admission and native handles are not qualified by this interface fixture. The real original-control preludes establish SDK10 and SDK234 before fresh zero-step Reset; no clock field is assigned to manufacture the higher step.

`policy-source-progress.log`: **6 passed**, 67.41s, including full eight-row legal-policy windows starting at SDK3, SDK11/policy0 and SDK235/policy0, plus a genuine third-step original joint guard that gives exactly three rows, -40 once and terminal Phi0 with source_complete=true/passed=false. `policy-source-green.log`: **31 passed**, 133.09s, adding wrong pose/clock/source/radius/scope and SDK-target rejection plus the session regressions. The final expanded combined run also includes NaN/parameter-clock corruption and the exact reviewed C02/C03 dependencies; its fresh result is recorded below.

Native steps remain **0** throughout these tests. CPU preludes and synthetic rod.steps are separate fixture counters and do not expand or consume native authorization. Full and independent Tripod/Power readers remain explicitly unsupported.


### Bounded Reset source limitation and precise admission

C06 independently found the frozen Mesh/Numeric Reset chain still uses `tabletop_mesh_reset_order._clock` with SDK<=240. Policy interval support beyond234 or240 does not fix or qualify that Reset chain. NativeEpisodeOwner now runs explicit `reset_clock_preflight` before any mutation or reset-budget reservation: SDK240 is within the old source contract; SDK241/258 are rejected with `bounded_mesh_reset_clock_v2_source_missing_for_reset_above_sdk240`. No clock is reset, remapped or forged and the old source bytes remain unchanged. `actual_admission.reset_source_limits` publishes max_reset_sdk_step=240, long_term_reset_supported=false, and an independent Reset v2 release condition. It leaves initial Reset2 plus a single-episode SDK258 smoke available under its independent finite authorization. Long-term multi-episode native readiness is not claimed.

The old e2 candidate baseline ordering probe in `reset-limit-baseline-red.log` demonstrates that it called budget reservation before any late-Reset bound, while the new boundary tests reject before that call. This baseline probe explicitly stops before any native/profile operation; it is not a native replay. Updating this disclosed admission field required one more fixed no-camera SHA revision in CPUFixtureOwners v2; all other source locks/owner topology/validation and the frozen CPU AST remain unchanged. `protection-remediation-final.log` again records scheduler source528+checkpoint2 PASS.

`reviewed-deps-policy-green.log` records the first full exact-dependency remediation combination: **290 passed**, 293.05s. The final fresh combination also checks the derived sample's SHA link, fixed41 target IDs and the explicit late-Reset source bound; its result is **292 passed**, 291.70s, in `final-remediation-green.log`. This uses the final source bytes, exact formal C02/C03 dependencies, fixed41/SHA assertions, cap tests and the original CPU AST/source-lock suite. Original176-pass, source31-pass and original independent SPEC RED evidence remain intact.


Final local delivery scope now adds the policy-distance module/test and policy addendum to the original C04 package. The remediation commit includes only C04 source/tests/reports plus the approved no-camera admission/fixed-SHA revision; the seven C02 dependency paths and three C03 dependency paths remain outside it. Worktree is clean after the local commit; no push/native run occurred. `candidate-files.json`, `remediation-staged.patch`, `remediation-staged-stat.txt` and `unchanged-frozen-source-final.json` record the reviewable file boundaries and byte checks.


## Original native-parameter constructor interface repair

The independent second SPEC review of candidate `6681bfc39514e0fac9ec1873fee96396c75a1570` passed the earlier 17 targeted policy-distance, trusted-guard and bounded-Reset tests but found one more actual entry defect: `NativePinchSources.__init__` passed the SDK tensor's `torch.device` directly to the unchanged `NativeParameterEndpoints` constructor, which requires an explicit `"cuda:N"` string. The broad `Getter(*a, **k)` substitute in the physical-source CPU fixture had hidden this argument contract error.

The repair is a single conversion at the adapter boundary: `str(owner.bindings[0].hand.data.joint_pos.device)`. The existing parameter reader, its device validation, current-profile scope, source inventory and physics are unchanged. New CPU tests instantiate the original real parameter reader through `NativePinchSources` for `torch.device("cuda:0")` and `torch.device("cuda:3")`, asserting the exact original reader type, device string, current backend SHA and unchanged 41 constructor nodes. The SDK device property is an explicitly synthetic CPU interface fixture; constructing a `torch.device` value does not allocate a CUDA tensor or import the native solver. No parameter getter or source-basis function is replaced in these two tests.

- RED: `device-constructor-red.log`, **2 failed, 14 deselected in 17.17s**, both at the original reader's `ValueError("explicit CUDA device required")`; SHA256 `58eac4b3c0cc25dc2a2ef8c17a827dad8af78639fc397047aaf9865cd8937281`.
- GREEN: `device-constructor-green.log`, **37 passed in 176.84s**, covering the complete policy-distance and session test modules; SHA256 `a14f8df9af2d110a6b3288df6523240376cf0b533916ca523a77c4e2d89805c6`.
- Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_policy_pinch_distance.py tests/test_native_teacher_session.py`.
- `protection-device-constructor.log`: fresh source/config **528** plus original checkpoint **2** SHA verification PASS.

The prior 292-pass combination remains evidence for its earlier source version, not a claim that this final one-line repair has already received a new full 292-suite integration run. Root will run the final combined dependency regression and independent review before publication. Native steps remain **0**; no GPU/App/Isaac, natural exit, new qualification or budget was exercised by this change.


## Partial-constructor borrowed-callback cleanup repair

Independent QUALITY review of `bf2ab2719b1a44252b393a6eb6c85266634e228d` found that a failed `NativeEpisodeOwner` constructor could leave its recorder installed on the borrowed original `DualHandPhysicalEpisode`. The episode factory had not yet returned, so `create_native_collector` could not close the partial lease. A real original-reader CPU-device rejection left a closure holding the unclosed owner and its episode/sim/rod/bindings. Setup budget already consumed two fixture setup steps and must not be refunded.

The owner now protects initialization itself, from history/reset callback construction through recorder installation, actual setup charging, setup journaling and source construction. Its restoration callback is registered before changing `episode._record`. Any constructor failure restores the original recorder, closes this lease and drops its borrowed scene references. `close()` also drops its own `owner_resets` dictionary, including wrapped original SDK/controller closures, without mutating the caller's dictionary. A failed cleanup log is recorded by the existing text-only fallback where possible and never replaces the original constructor exception. No SDK STOP, Rope destruction, simulation clear or App.close was added at this borrowed boundary. Outer R06 teardown/exception-frame clearing remains required; dropping this lease is not evidence of native release or natural exit.

Four real-owner failure regressions use the original physical episode and original CPU SDK/rod fixture: original parameter-reader CPU-device rejection; original setup-budget overflow after two charged steps; setup-ledger I/O failure; and original-reader failure accompanied by both close/cleanup journal failures. All assert original recorder identity, one close, closed lease, empty restores and reset-callback dictionary, cleared episode/sim/rod/bindings, unchanged caller reset callbacks, preserved consumed setup cost, zero solver calls/resets and no model construction. The diagnostics-failure case verifies the fallback remains text-only and the primary reader error survives.

- RED: `constructor-lifecycle-red.log`, **4 failed, 21 deselected in 19.48s**, all on the original recorder not being restored; SHA256 `abba501b29f0f20ff9bbd202163feb3f80d6e62b9d75731d33d6ad6428e26401`.
- GREEN: `constructor-lifecycle-green.log`, **25 passed in 19.25s**, the full session suite including those four regressions; SHA256 `7331fe428fa69b953b91314f2d5e2f743ad42228303950683be4f98d7329b9cd`.
- Command: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_session.py`.
- Fresh `protection-constructor-lifecycle.log`: **528 protected files plus two historical checkpoints PASS**.

The earlier device-string fix is preserved. This is a CPU software repair, not a new physics attempt: **native_steps=0, PPO_updates=0, GPU/App/Isaac launches=0**. No new native budget or qualification is claimed. Final combined C04/C05 regression and independent QUALITY re-review remain root's next gates.
