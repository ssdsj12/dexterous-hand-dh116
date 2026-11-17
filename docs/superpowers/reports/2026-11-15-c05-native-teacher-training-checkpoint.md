# C05 independent native Teacher checkpoint and bounded training CPU candidate

Date: 2026-11-15. Scope: `cpu_fixture` software only. Native qualification, Teacher admission, physical grasp success and native PPO update claims remain false. No native/App/GPU execution, install, push or historical model write occurred.

## Dependencies and scope

Ubuntu isolated worktree `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-c05-training-20261115`, branch `codex/teacher-c05-training-20261115`, base `535dcab9f71fd37364132893a3e6f118f174b488`.

C02 exact seven own paths came from candidate `12aeee74e65bf35f60c4f7bfd2c75ea69dcaa140`; root confirmed independent SPEC/QUALITY and formal release `712f7fcf3e3c0a6992951228071ba08ecf66762f` with identical bytes. C04 initial exact eleven own paths came from `e2d3b9178143208686a623cfbb94cf384e65fe01` as a separate pending dependency snapshot. Its synthetic CPU source fixture is not evidence that its actual adapter passed. Formal C04 reviewed combination remains pending until separately recorded below. Integration was not modified.

Own implementation consists of the planned checkpoint, CLI and two test files plus the necessary small orchestration module `tabletop_native_teacher_training.py`. No PPO, GAE, action, physical, reward or historical checkpoint module was changed. The new module delegates original trainer `update`, frozen likelihood, masks, clipping and KL rollback; it does not implement an optimizer algorithm.

## Behavior

`stop_update_target` accepts exact nonnegative integer loaded updates and exact positive integer additions, rejecting bool. Runner uses original `counters["updates"]`, collects only to real rollout full, then verifies original accepted minibatches, true policy/counter advance and finite nonzero parameter change. A zero-accepted original update may clear its rollout; the runner immediately stops with `optimizer_update_rejected`, latches this session failed and never resamples to fill another rollout. Source errors preserve pending and C04 quarantine evidence, never save a replacement checkpoint or reduce consumed budgets. Terminal episodes require explicit original factories to obtain the next verified Reset. Normal rollout boundaries leave the episode and shaped-value behavior to C04 and original GAE.

Independent schema `dh116-native-context-teacher-checkpoint-cpu-v1` stores every model branch and log_std, complete Adam moments/steps/LR, original counters/version, total accepted minibatches, Python/NumPy/Torch CPU RNG, explicit normalization `None`, fixed input scaling contract, model/input/context/backend identity, reward/target/reset/action/source/config SHA, scope and origin. Adam steps must match total accepted minibatches. The safe weights-only load checks payload, policy topology/extra identity/noise, Adam and RNG on disposable candidates before current state mutation. Clean save requires no pending, empty rollout and a verified accepted completed optimizer update. Storage follows the measured checkpoint exclusive temporary-file/fsync/link algorithm and reuses strict validation helpers; existing writers/globals are untouched. Existing paths are never overwritten. Historical four-key/measured formats are independently rejected in both directions. Smoke/full migration is rejected, with no silent actor/critic/Adam transfer.

`checkpoint_trainer_factory` loads during C04's existing trainer factory: static contract, fresh Reset/live verification, model, trainer+load, reward, first collector. That collector freezes loaded version N directly. No existing collector epoch is rewritten. The new physical episode/context/history are fresh; training RNG/state restoration does not claim physical trajectory bitwise continuation. Both physical-state flags are explicit in the payload.

CLI options are `--config --output --additional-updates` (default 1), `--checkpoint`, `--preflight`, `--execute-native`. Preflight imports schemas but constructs no policy, optimizer, App or camera. It records actual missing resolved Reset/reward/budget/source/config requirements and `c06_source_bound_worker_lease_missing`. C05 has no registered native worker provider. `--execute-native` therefore refuses specifically with C06 pending and never treats a boolean as authority. The source-bound runner and trainer factory are available for the separately frozen C06 worker; this is not a completed native launcher. No old 234/240-step, zero-update authorization is enlarged.

## Verification and retained evidence

Private evidence directory: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c05-training-20261115` (mode 0700). CPU fixture binaries/checkpoints and full logs stay private/outside Git.

- Initial RED: 35 failed, explicit missing-module assertions, `red.log`. No production implementation existed.
- First implementation attempt: 20 failed, 15 passed. The failures exposed test fixture subdirectory creation and reusing one exclusive CLI report path; fixed the test harness, not product behavior. Full log `green-attempt1.log`.
- Corrected initial GREEN: 35 passed in 100.39s, `green-attempt2.log`.
- Resume-factory RED: one expected missing-helper assertion, `red-resume-factory.log`; GREEN one passed in 12.05s, `green-resume-factory.log`. This uses actual new model/original PPO and original C04 physical CPU owner, loaded update1 then additional1 reaches update2 with fresh CPU Reset/history.
- Preflight-model RED: one expected failure because invalid model metadata was not reported; implemented schema-only model-contract validation. `red-model-preflight.log`.
- Initial protection: `Repository.verify_protection()` passed (`source528_checkpoint2_protection_PASS`), `protection-initial.log`.

Canonical CPU command prefix: `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= bash scripts/python.sh -B -m pytest -q -p no:cacheprovider`.

Expanded and final reviewed dependency results are recorded after completion. CPU tests do not grant actual backend/source/reset qualification, GPU training, natural-exit proof or task success. Full contact/slip/Hold and Tripod/Power source adapters, fresh worker lease, explicit new budgets and real resource/kernel reservation remain C06/source gates.


### Expanded pending-C04 snapshot result

The expanded pre-final-hardening snapshot command covered C05 training/checkpoint, original deformable PPO, C02 model/context, C04 session/collector, reward, old no-camera collector and old action history. Result: **503 passed in 219.28s**, exit 0, `green-combination-pending-c04.log`. It uses C04 `e2d3b917` and does not represent final reviewed actual-source qualification.

Strict numerical Adam regressions were then added: boolean step and negative second moment both failed as expected before the new validator rejected them (`red-adam-numeric.log`, 2 failed). A separate explicit missing-budget/authorization/kernel-resource gate regression failed as expected (`red-execution-requirements.log`, 1 failed), then separate gate reasons and requirements were added. The final C05 test rerun follows these changes; it is recorded below once complete.


### Stable own-file candidate

Final own tests: **43 passed in 129.79s**, exit 0, `green-candidate-final.log`, after all C05 hardening. Exact command: canonical prefix followed by `tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py`.

The five new Python files also passed in-memory syntax compilation (no bytecode artifacts). Formal C02 seven files were compared byte-for-byte with `712f7fcf3e3c0a6992951228071ba08ecf66762f`, all equal. Candidate `git diff --cached --check` passed. Repeated protection passed with `source528_checkpoint2_protection_PASS` in `protection-candidate.log`. This is a stable C05 CPU candidate for independent review, not final reviewed-C04 integration or native qualification.


## Independent QUALITY P1: immutable input snapshot / atomic restoration

The independent SPEC review passed the CPU candidate. QUALITY then reproduced a P1: `load_checkpoint` changed live model/Adam/counters/progress before reopening the checkpoint path to compute resume origin. A path deletion at that late read raised `FileNotFoundError` with partially restored state; path replacement silently bound origin to bytes different from the loaded payload. This invalidated the candidate's all-errors-atomic claim until fixed.

The original behavior was independently reproduced before the fix in four deterministic CPU cases (after deserialization or after disposable validation, each followed by delete or replacement): **4 failed, 1 passed in 21.63s**, exit 1, `red-quality-atomic-snapshot.log`. The deletion repro explicitly records updates `0 -> 1`, Adam states `0 -> 29`, completed marker `1`, origin `None`, and `rng_restored False`. Replacement cases record a mismatching origin SHA. A separate **1 failed in 9.65s**, exit 1, `red-quality-single-read.log`, asserts that deserialization must receive `BytesIO` from exactly one `read_bytes`, catching a separate-hash / reopen-path implementation.

The fix reads bytes once, hashes those same immutable bytes, safely deserializes them with `weights_only=True` from `BytesIO`, then performs disposable validation. Counters, origin and return metadata are prepared before live commit. No checkpoint path I/O occurs after commit begins. The defined behavior is immutable-snapshot restoration: deletion/replacement after the successful read does not invalidate that already-read snapshot, and origin always identifies the exact loaded bytes. A missing/unreadable input fails before any live-state change.

The original 18 invalid-payload cases now compare the complete training snapshot: model, Adam, all three CPU RNGs, counters/version, present/absent progress markers, populated resume origin, pending state and rollout object. New regression coverage includes the four mutation boundaries, exactly-one-read/BytesIO behavior, and read failure with populated markers. No old checkpoint code or globals changed. Scope remains the original six ownfiles; this fix touches only checkpoint source, its tests and this report.

Repeated protection passed (`source528_checkpoint2_protection_PASS`, exit 0), `protection-quality-atomic-snapshot.log`. The full original 43 tests plus six new regressions are rerun below; final C04 dependency combination and C06 native gates remain pending.


QUALITY fix GREEN: **49 passed in 158.05s**, exit 0, `green-quality-atomic-snapshot.log`. Exact command is the canonical CPU prefix followed by `tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py`. This includes every original 43 case plus six new regressions. All delete/replacement snapshot cases pass, one byte read/BytesIO is enforced, origin binds actual loaded bytes, and invalid payload / read errors preserve all live fields. Changed Python files compile in memory and `git diff --cached --check` passes. The fix is submitted for the same independent QUALITY re-review; it does not close final C04 combination or C06 native requirements.


## Exact C04 bf2ab271 CPU combination; final C04 still pending

This section supersedes the earlier **503-test pending-C04 e2d3b917 snapshot** as the current combination evidence. It is an exact **bf2ab2719b1a44252b393a6eb6c85266634e228d CPU snapshot**, not final reviewed-C04 closure. The upstream independent C04 QUALITY review subsequently confirmed a constructor-failure callback cleanup gap; its minimal session/test/report fix and the new exact dependency snapshot are still pending here. The passing suite below does not close that review finding.

Started from clean C05 `1990e2f3b37c0272379878e59cdfb15abc5f8cad`. C02 formal `712f7fcf3e3c0a6992951228071ba08ecf66762f` seven own paths and C03 formal `fee8de1b34294c294213e6e2eb77e9f8f826f40e` report/reward/test three own paths were compared byte-for-byte and all matched. The explicit fourteen C04 own paths were read from `bf2ab271` without branch merge or cherry-pick. Five already matched; only the nine actual differences were imported as separate dependency commit `33524963e06e5a15c115012cfbe0d3c7c1ade128`. Every resulting C04 path matches the source SHA256 recorded in `final-c04-dependency-import.json`; that evidence file SHA256 is `7656a61ff8c6642b4b8171b356b56cdfcdcabcb4ede68ee4d30b73a2a42ce642`. C05 checkpoint/training/CLI/test implementation was unchanged from the independently re-reviewed `1990e2f3` version.

One necessary CPU combination ran at dependency commit `33524963`: **232 passed in 407.49s (0:06:47), zero failed, exit 0**. It includes all 49 C05 tests, C04 session/collector/policy-owned distance, C02 context/model, original deformable PPO and the affected no-camera collector. The full 269-case reward suite was not repeated. Exact command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py tests/test_native_teacher_session.py tests/test_native_teacher_collector.py tests/test_native_teacher_policy_pinch_distance.py tests/test_native_teacher_context.py tests/test_native_teacher_model.py tests/test_dh116_deformable_privileged_ppo.py tests/test_dh116_no_camera_collector.py
```

Evidence remains in the private directory above. `green-combination-final-c04.command.json` and `.result.json` retain the actual cwd, argv, environment, dependency/source SHA, timestamps and exit status. `green-combination-final-c04.log` SHA256 is `0f409c3225f179f73bce310b7e2d9d71ca75d777d85f647e0dc9736e466e9d9a`. No test failed in this combination; the upstream C04 QUALITY constructor-lifecycle finding remains an independent unresolved item, and no C04 source was locally altered to hide it.

Post-combination `Repository.verify_protection()` passed with the registry's actual **528 protected source/config files and 2 historical checkpoints**; `protection-bf2-combination-report.log` retains the result. Report-only `git diff --check` and staged diff checks passed. The ownbase-to-candidate path classification is retained outside Git in `bf2-combination-path-classification.json` for root integration.

CPU/native accounting remains separate: **actual native runs 0, App launches 0, GPU execution 0, native training 0, installs 0 and pushes 0** in this task. There is still no actual C05 native worker entry point or C06 worker lease, and no native/source/Teacher admission, physical grasp success, full Hold/contact/slip coverage, natural-exit proof or native PPO update qualification is granted. Final status: **pending-final-C04**, awaiting the exact reviewed constructor-lifecycle fix and necessary affected CPU verification. Windows relay publication remains root-owned; PR #2 must stay Draft and historical models/`11-10` remain untouched.


## Exact C04 503d9262 constructor-cleanup delta and affected CPU closure

The current local C04 dependency source is **503d9262e0fe4d9ab716acf6824d600798c13a9f**, whose parent is `bf2ab271`. This supersedes the preceding local `pending-final-C04` constructor-fix status: the exact fix is now imported and its affected CPU combination passes. Independent upstream SPEC/QUALITY sign-off, root integration and Windows publication are recorded by root separately; this report does not infer those outcomes from local CPU tests. The preceding **232 passed / 407.49s** result remains specifically the `bf2ab271` snapshot and is not relabeled as a 503d result.

Only three C04 paths changed from that parent: the session module, session tests and C04 report. They were read directly from `503d9262` and committed as independent dependency commit `799724d238202eb3c714e4bf60c8215d94807ae6`, without branch merge/cherry-pick. All fourteen C04 own paths then matched `503d9262` byte-for-byte; C02 formal seven paths, C03 formal three paths and all six C05 own files were unchanged during import. `final-c04-503d-dependency-import.json` retains every source/current SHA256 and the three-path delta; its SHA256 is `26b9d156d161029e3a52e54ba217a181835280d0567f355f69c460653d74a60f`.

The C04 constructor now owns cleanup if initialization fails after acquiring callbacks, including setup-budget, setup-ledger and source-constructor failures. It restores the episode callback, drops its borrowed reset closures/references without mutating caller owners, and preserves the primary exception if cleanup diagnostics also fail. The added four cases exercise the real owner/reader constructor on CPU SDK/rod fixtures, preserving setup accounting and proving that Reset/model/trainer/reward do not run after failure. Public APIs, control behavior and PPO code did not change. C05 implementation received no functional change.

One affected CPU combination at `799724d2` completed with **92 passed in 206.02s (0:03:26), zero failed, exit 0**: all C05 checkpoint/training tests plus the complete C04 session and collector files, including constructor failures, real three-episode sequencing, guard/Reset and cleanup cases. Exact command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py tests/test_native_teacher_session.py tests/test_native_teacher_collector.py
```

`green-combination-c04-503d.command.json` and `.result.json` retain actual cwd, argv, environment, dependency/source SHA, timestamps and exit status. `green-combination-c04-503d.log` SHA256 is `7b6f777cd1e1d7b0d14870b84e34cff743c2504c0bb18a8b26ca6ce2ce7f0f21`. The unchanged policy-distance asset suite and full 269-case reward suite were not repeated; their earlier evidence keeps its original source provenance. There were no failures in this affected combination.

After this report update, `Repository.verify_protection()` passed with the actual registry counts **528 protected source/config files + 2 historical checkpoints**, recorded in `protection-c04-503d-report.log`. Report-only unstaged/staged diff checks passed; `c04-503d-report-diffchecks.json` retains their commands and exit status. `final-c04-503d-path-classification.json` records the complete ownbase-to-candidate classification: six C05 own paths plus nine C04 dependency paths, with five additional C04 paths already present at ownbase.

Scope remains **cpu_fixture only**: actual native/App/GPU/native-training/install/push counts are all **0**. No actual C05 native worker entry point or C06 worker lease exists, and native/source/Teacher admission, physical grasp success, full Hold/contact/slip coverage, natural exit and native PPO update qualification remain **false**. This is a local CPU-verified exact-source candidate for root review/integration; Windows relay publication and PR #2 Draft verification remain root-owned. Historical models and `11-10` are untouched.


## C05 read-only immutable checkpoint-byte audit extension

Extension base: clean `797e87e218c71e266b0dd188aa5a2d90ca256237`, retaining exact C04 `503d9262` dependency bytes. Root approved this small CPU-only API to let C06 validate a real completed C05 checkpoint instead of accepting schema/SHA-only text or update-0 artifacts. Only this report, `tabletop_native_teacher_checkpoint.py` and its existing test file changed. C06 artifact adaptation and independent checkpoint QUALITY review are separate gates.

New API: `audit_checkpoint_bytes(raw, trainer, contract)`. It accepts **exact `bytes`** only, preserves existing strict CPU/device and source-identity checks, deserializes that same immutable snapshot exactly once through `BytesIO` with `map_location='cpu', weights_only=True`, and calls the existing complete `_validate_payload` (including `_decode_rng`, disposable model/Adam validation and accepted-update checks). It never calls `_clean`, restores RNG, loads state into the live trainer, replaces pending/rollout, executes an update or opens a checkpoint path. Existing `checkpoint_identity` source-file SHA reads remain required. No new PPO/checkpoint framework or schema/device migration was introduced.

The returned dictionary contains exactly nine fields, all deeply owned metadata: `version`, `identity`, `counters`, `policy_version`, `accepted_minibatch_updates`, `physical_state_saved`, `restart_requires_verified_reset`, `checkpoint_sha256`, `policy_weights_sha256`. The checkpoint SHA hashes only the supplied bytes. The policy SHA is the original `tabletop_deformable_r0_replay._digest(payload['policy'])`, matching the digest used by raw collector `policy_epoch.weights`; it is distinct from file-byte SHA. No Tensor, payload, model or optimizer object is returned.

All nine original function source segments, including `load_checkpoint`, remain byte-for-byte unchanged from the extension base. The original loader's one-read immutable snapshot and atomic restoration behavior is retained; its function-body SHA256 is `37f2f5be4c765dac3b800a40c6dc8f8d6f0f6276a64b0a9d66ebb5d355d61f49`. `checkpoint-byte-audit-design.json` records this check and the exact metadata fields. The checkpoint schema remains `dh116-native-context-teacher-checkpoint-cpu-v1`; current source identity naturally changes with the new audit source, and historical checkpoint bytes remain read-only with no rewrite or silent migration.

TDD RED before the helper existed: **37 failed, 40 deselected in 2.89s**, exit 1, all asserting `read-only immutable checkpoint bytes audit missing`. `red-checkpoint-byte-audit.log` SHA256 is `d3d50eee89263187b89833bb4fb66313da9b9a8970b4cc33cf2d6b07f4ae3f43`.

One full affected GREEN command then passed **129 tests in 386.71s (0:06:26), zero failed, exit 0**: original 49 C05 tests, 37 new audit cases, and all 43 C04 session/collector cases. Exact command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_training.py tests/test_native_teacher_checkpoint.py tests/test_native_teacher_session.py tests/test_native_teacher_collector.py
```

New cases audit a genuinely saved accepted-update fixture while the target has nonempty Adam, populated pending and nonempty rollout. They compare model/Adam, Python/NumPy/Torch CPU RNG, counters/version, present and absent markers, pending contents/identity, rollout contents/identity and collection capability identity. Successful metadata is recursively plain data and cannot alias the deserialized payload, contract or live state. Tests forbid `_clean`, update and global RNG-restore calls; delete the checkpoint path and forbid byte-path reads; allow only the existing source `.py` identity reads; and require one exact-snapshot CPU/weights-only load. The RNG validator uses only disposable CPU/local generators.

Rejections cover all 18 original shared invalid-payload variants plus the two Adam numeric variants and six additional update-0/physical-reset/rollout/field/NumPy cases; fake text/empty input, wrong expected identity, five non-exact-bytes variants and a declared non-CPU trainer also fail with every live snapshot field preserved. The declared-CUDA negative test allocates no GPU state and fails before deserialization.

Evidence: `green-checkpoint-byte-audit-combination.command.json` and `.result.json` retain actual argv/environment, tested source/test SHA, source dependency, timestamps and exit code. Log `green-checkpoint-byte-audit-combination.log` SHA256 is `72db1f8e16be2e57543f85abe697fdebe380ec8ac945a882a5714f068eda98ff`. Tested checkpoint source SHA256 is `14b72abc48637f6e6a7f1d9a77824d555044369fe94eb04200966c623b189164`. All logs remain in the private evidence directory above.

After this report update, protection validation passed for the actual registry's **528 source/config files and 2 historical checkpoints**, recorded in `protection-checkpoint-byte-audit.log`. In-memory syntax compilation, complete extension diff checks and explicit three-path staging passed; `checkpoint-byte-audit-diffchecks.json` records the checks. This is a local CPU candidate awaiting the same independent checkpoint QUALITY review and root integration/publication. Actual native/App/GPU/native-training/install/push counts remain **0**, native qualification remains **false**, and no actual native worker entry point or C06 worker lease is provided. Historical models and `11-10` were untouched.
