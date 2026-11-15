# C02 native-context Teacher CPU implementation — 2026-11-15

Status: CPU implementation and independent SPEC review fixes complete for re-review. Native owner/Reset/contact/reward/training admission remains pending C04. No Isaac/native/GPU execution, formal training, installation, push, physics modification, or historical trajectory replay was performed.

Integration base: `a6a4290407456b4ea2d707e27272885b74e95bc6` (published reviewed C01, including c0cc12c8). Branch: `codex/teacher-c02-context-model-20261115`. The original candidate a249a391 was safely rebased as 4afdd21e with `git rebase --onto a6a429... de69778...`, replaying only the five C02 additions. A prior plain rebase attempted to replay the superseded C01 candidate and stopped on add/add conflicts; after checking ORIG_HEAD/scope it was aborted. A missing committer identity on the first --onto attempt was also safely aborted and retried with explicit Codex identity. No conflict was manually resolved, no source was copied, and no reset/force operation was used. Operation/exit records are in spec-fix-rebase*.json.

Cumulative scope from this base is exactly **five NEW files +two authorized C01 integration modifications**:

- NEW `src/go2_pvcnn/control/dh116/tabletop_native_teacher_context.py`
- NEW `src/go2_pvcnn/control/dh116/tabletop_native_teacher_model.py`
- NEW `tests/test_native_teacher_context.py`
- NEW `tests/test_native_teacher_model.py`
- NEW this report
- MOD `src/go2_pvcnn/control/dh116/tabletop_native_teacher_preflight.py`
- MOD `tests/test_native_teacher_preflight.py`

The parent explicitly authorized the preflight metadata correction after its real module-presence path exposed a callable-as-SHA/CLI serialization bug. All other C01 source, protected legacy code, README/CHANGELOG/registry/readiness, models, checkpoints, and physics remain unchanged.

## Tensor/model contract

C01 schema constants remain unchanged: MODEL_SCHEMA_VERSION=`dh116-native-context-teacher-tanh-v1`, INPUT_SCHEMA_VERSION=`dh116-native-context-teacher-input-v1`, CONTEXT_SCHEMA_VERSION=`dh116-native-teacher-context50-v1`.

CONTEXT_FIELDS has exactly 50 index/name/unit/scale/source/validity/applicability descriptors in the schema SHA. Identity/source/applicability/reason/reference UID data is an immutable sidecar, outside the exact six owned tensor keys. Required applicable unknowns reject before encoding. Raw unknowns remain NaN/false; encode_context accepts a generic last dimension and concatenates where(valid,value,0) with validity.

common371 selects exactly 0:24, 78:88,153:185 (66 finite entries); the other 305 remain NaN/false. All 41 original material nodes and all 13 structured features are required, with no mask that can hide an unavailable node. Original _features and _MaterialEncoder are reused.

Actor and critic independently implement common66→256→256, rod13→64→64 with mean+max128, context100→64→64, fusion448→24/1. The original log_std24, Tanh sampling/log-probability/entropy, GAE, and PPO code are unchanged. NativeContextTeacher is a DeformablePrivilegedTeacher instance; _parts still returns exactly (common,nodes,mask).

The independent model namespace/identity binds actual BackendProfile binary/source-tree/patch/inventory SHA, model/context/input schema and dependency source SHA (including the actual original task/history code). The actual ABI binary SHA is `8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340`. Historical 2fea identity is not retagged. Identity-bearing state dictionaries reject old topology/checkpoint identity, another grasp, and altered identity before restoration.

pack_native_inputs independently validates mounted common, same-clock raw rod, current source profile, raw/derived parameter snapshots, current-ABI radius provenance, unchanged locked gravity/material/friction/free-root/mass construction, and reconstructed RodBoundaryState. It never calls historical ActualBackendMeasuredTeacher._actual_inputs or relies on the historical radius checker's global BACKEND_SHA. The positive pack fixture generates NEW synthetic 41-node data from the current locked profile; context/raw rod/parameters all retain cpu_fixture scope. CPU-to-live relabelling rejects.

## Public APIs and C04 ordering

```python
NativeContextTeacher(config, *, backend_profile,
                     context_mode="approach_wiring_smoke_v1")
model.pack_cpu_fixture(ledger, valid, rod_snapshot, common_identity,
                       node_valid, task_context)
pack_native_inputs(pre_action, raw_rod, parameters, task_context, *,
                   backend_profile)
model.pack_native_inputs(pre_action, raw_rod, parameters, task_context, *,
                         backend_profile)

ContextIdentity.from_clock(clock_dict)
TaskContextPacket(values, valid, metadata)
validate_context(packet, *, backend_profile, identities=None)
encode_context(values, valid)

TaskContextOwner(*, backend_profile, mode, scope,
                 episode_cap_s, external_cap_s, grasp="pinch")
owner.reset_from_verified_state(state, *, history_owner, phase_owner=None)
owner.freeze_pre_action(identity, *, history_owner)
owner.advance_physical_endpoint(state, *, history_owner, collection_token,
                                anchor_owner=None, phase_owner=None)
owner.commit_post_context(packet)
owner.close()
owner.current
owner.pending_post
```

ContextMetadata is frozen and carries owner_uid, identities (tuple of frozen ContextIdentity), backend_profile_sha256, scope, mode, sources/applicable/reasons (immutable width50 tuples), reference_uids (immutable string pairs), previous_request_token, and grasp. Scope is cpu_fixture or owner_source_unqualified; neither grants native qualification. TaskContextPacket.native_qualified is always false. Returned tensors are owned copies.

State includes identity, source_scope, backend_sha256, actual reset node_positions [1,41,3], distance_mean_m, rest_length_m, target_reference_uid, and external_budget={remaining_s,cap_s,source:"external_budget_clock",identity}. Reset geometry/distance/dref/target/rest coordinates stay fixed. External remaining time comes from its explicit same-clock source/cap, cannot increase, and is not inferred from physics elapsed or treated as a task deadline.

history_owner must have exactly the original Action24HistoryOwner type or the exact C04 NativeAction24HistoryOwner type when that module is available. The C04 import is delayed; only absence of that exact module is tolerated. Unreviewed subclasses and duck-typed caller objects reject. Reset reads the actual executed controller reset receipt, freeze checks captured history, and advance checks actual issued settlement/previous-request SHA/collection token/episode/endpoint. No historical zero request is guessed.

advance produces a pending owned post context while current remains pre. C04 computes next value using post context BEFORE commit. Returned packet mutation does not change either internal snapshot, and tokens cannot be reused.

### Corrected fresh Reset and first anchor

The original a249 candidate's 117 CPU passes did not cover two genuine loopholes found by independent SPEC: caller-supplied reset anchors and string-selected full phase. Those paths are corrected here.

Fresh Reset always begins unanchored. Omitted anchors or explicit (None,None) pairs leave stations NaN/false and anchor-valid booleans zero. Any caller-provided station/UID at Reset rejects; nonfresh anchored reuse is not supported.

First anchor establishment still reads the original FlexibleGraspEpisode _slip/material_anchor/slip_references with same environment/grasp/episode/reset/rod/endpoint/rest geometry. The original CPU task actually steps to establish material/tangential references. Only post context receives the first stations; commit fixes them. Contact loss cannot clear them; replacement position/UID rejects.

For full mode, phase and anchors share the same actual task object. Once its anchor is known, omitting anchor_owner cannot hide it: the bound original phase owner supplies the anchor. A different anchor owner rejects.

### Actual full phase receipt and observer lifecycle

Full mode requires phase_owner at reset and advance. It must be the original FlexibleGraspEpisode with original reset/step methods, matching environment/grasp/episode/reset/rod/physical clock and backend (outside explicit CPU fixture scope). Reset verifies its actual cleared approach/material_anchor/slip-reference/hold/support state.

After successful full reset, the context owner installs one exclusive INSTANCE step observer. It calls the original bound step exactly once, preserves its return value, and saves an owned receipt of the actual original result and task state. It does not advance SDK/solver physics itself. The next full context requires the same leased object, matching latest original receipt, and unchanged current phase/hold/anchor state. Bypassing the observer, replacing step, changing the owner/clock/episode/grasp, manually changing phase, or requesting Hold via caller strings rejects. The original mapping is approach→Approach, grasp→Close, lift→Lift, hold→Hold; it never uses rollout/update counts.

Original CPU step fixtures actually reach all four phases with real required-pad/opposition/lift/hold decisions. Reset contact/hold history is zero, and post fixture coverage/persistence/hold values come from the original task's actual calculations. The first good PHYSICAL hold endpoint has zero hold duration; later control endpoints preserve the original measured hold_s rather than resetting a sampled context duration.

Call close() in C04's finally block after normal completion or context validation failure. It is idempotent and restores the exact original bound step. An exception from the original step/receipt observer automatically restores it and marks the phase source failed; no exception or traceback is retained. Multiple contexts cannot stack observers on one active owner. Close is required before reusing that task object for another episode/context. Smoke requires no phase owner, installs no observer, and remains Approach.

Full mode still requires explicit physical_fields 4,5,6,7,47 with {value,valid,source,identity}; duration seconds are normalized by .25/5 and hold must equal the actual phase owner's history. Missing applicable contact/support sources reject. Source labels alone do not qualify hardware or native contact. Event changes require same-clock event_history; raw event booleans reject.

## Authorized shared preflight integration

Missing model/context behavior now has an explicit absent-module fixture. Separate real-present tests cover all three grasps while retaining static_ready=false/live_reset_verified=false.

Real module presence also exposed check() treating an imported module.digest FUNCTION as source_sha256, causing CLI JSON failure. The shared helper now accepts digest/sha256 only when type is exactly str and it is full 64-hex; otherwise it falls back to actual reader file SHA. It does not stringify or call metadata functions. Tests cover callable/callable, bool/short, short/valid SHA, and valid digest/callable; current real modules and the original CLI exclusive-output/exit-code tests pass.

## TDD, final regression, and protection

Private evidence: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c02-context-model-20261115`. No logs, fixture state files, trajectory replay, or formal model weights enter Git.

Initial C02 RED/GREEN logs remain retained. Review-fix RED runs include:

- red-spec-reset-phase.log: **3 failed** reproducing two reset-anchor cases and string-only Hold; exit1, SHA `b59a98cb1877f619f7ddc0ce2cc7a2d418bb0c3a31c0e948b64f504e90652277`.
- red-c01-module-presence.log: **3 failed** stale default-absence assertions after real modules exist.
- green-spec-reset-phase-1.log: **4 failed,133 passed**; actual module SHA/CLI serialization bug; SHA `d0aed3500b8d7e412ab9d917d7bd7edd9d796a8ffc54f10e6637b155eb06b1c7`.
- red-source-identity-type.log: **3 failed,1 passed** demonstrating callable/bool/short identity acceptance.
- red-spec-phase-type.log: **3 failed** for untyped phase owners, corrected to explicit ValueError rejection.
- red-spec-shared-phase-anchor.log: **1 failed** for different full phase/anchor owners, corrected by one physical source binding.

Final canonical command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  bash scripts/python.sh -B -m pytest -q \
  tests/test_native_teacher_context.py tests/test_native_teacher_model.py \
  tests/test_native_teacher_profile.py tests/test_native_teacher_preflight.py \
  tests/test_dh116_deformable_privileged_ppo.py \
  tests/test_dh116_deformable_privileged_teacher.py \
  tests/test_dh116_flexible_measured_common_teacher.py \
  tests/test_action24_history_owner.py
```

Actual final: **216 passed,0 failed,0 skipped**, exit0, 69.21s. Counts: **90 C02 tests (59 context+31 model),73 C01 tests (13 profile+60 preflight),53 unchanged old regressions**. Log final-spec-regression-3.log SHA `6b18a60a7c17fac0dd7f9ec0fdd03279ad6e767729e134bdd8a4d01f73af087c`; exact argv/environment/exit are in final-spec-command-3.json.

The actual original DeformablePPOTrainer collects, records, audits frozen likelihood, and updates one minibatch on explicit CPU data; all four rod/context branches change finite parameters. Original deterministic model output and source SHA remain unchanged. copy.deepcopy(model) and torch.save(state_dict)→torch.load(weights_only=True) preserve independent topology/identity/value/action outputs; the old checkpoint writer rejects the new identity.

Repository.verify_protection() passed for **528 protected source/script/config files+2 historical checkpoints**. spec-protection.log SHA `6cab563656f519a6278f6ea95986c201ad855892aec6ca71ff7de62e3921caac`. Cumulative diff/staging is restricted to the seven paths above; diff --check is clean. Other C01 files remain byte-identical to published base.

## Pending actual source qualifications

- C04's new typed NativeAction24HistoryOwner API was read from its exclusive tree. Delayed exact-type compatibility is implemented and old-owner/explicit-absent-module/fake-subclass tests pass. Its actual SDK10/no-step fresh Reset and same-scene three-episode positive tests remain the C04 integration responsibility; C02 does not claim those results.
- CPU source-byte/phase/history-owner proof is not a fresh native owner/Reset qualification. Per-grasp live task source evidence, especially Tripod/Power, remains pending. The Pinch-only distance reader is never reused by changing its grasp string.
- C04 must connect actual reset/clock/history/contact/phase/anchor/external-budget/event sources and evaluate bootstrap with post context before commit. Full/hardware contact/slip, native readiness, physical grasp success, formal training, and publication are not claimed.
