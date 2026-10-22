# DH116 Preload Window Feedback Bugfix Plan

> **For agentic workers:** Use superpowers:executing-plans, Inline Execution. Continue the existing `10-21` workspace; no commit/push, new worktree or subagents.

**Goal:** Fulfil the already approved command-hold contract: actual load/contact loss during a control interval must release the command hold at the next MPC decision.

**Architecture:** Extend the existing `PreloadTargetHold` with constant-size causal minimum-load/maximum-gap/known-observation accumulation. Only the next existing10ms decision consumes that window. The physical qualification observer remains unchanged and continues every100µs.

**Tech Stack:** NumPy, pytest, existing hand-only MPC/complete-v2 collector, Isaac with CUDA Cosserat rod.

## Global Constraints

- This repairs the existing expert-loop requirement to resume regulation on measured load/contact loss; no new grasp target or Panda controller.
- Keep original90% load threshold, touch/opposition,100ms stable preload,1s hold, contact separation, slip, lift and whole-wire-clearance criteria.
- Keep original assets, calibrated points/regions, effort limits, gains, physics timestep,16 samples and CUDA rod.
- Keep12D hand commands and separate Panda-only transport. Parked middle channels remain excluded from pinch holding.
- Preserve previous-action references, raw physical observations and causal collection. Never use future outcomes in a decision.
- Single CPU core/thread and one Isaac. V21 may keep running on its already loaded old code during offline tests. Its two old runtime sources are backed up under its run directory.
- No physics/gain/cache/performance change is bundled with this fix. A new run must requalify close/hold/lift, without approach replay or phase credit.

## Task1: Reproduce and repair the sample-loss bug

Files: `tests/test_dh116_grip_load_reference.py`, `src/go2_pvcnn/control/dh116/grip_load_reference.py`.

- [x] Focused baseline:8 passed in0.11s before new tests.
- [x] Add the failing reproduction using V21-like values:
```python
hold=PreloadTargetHold(np.ones((2,3)),active_mask=[[1,1,0],[1,1,0]])
gaps=np.full((2,3),-.0003);known=np.ones((2,3),dtype=bool)
hold.update(np.full((2,3),.901),gaps,known)
low=np.full((2,3),.901);low[0,0]=.899866925
hold.observe(low,gaps,known)
np.testing.assert_array_equal(hold.update(np.full((2,3),.901),gaps,known),
    [False]*6+[True]*6)
```
- [x] Test window consumption, copied buffers, decision-only release, exact90% equality, transient open gap/unknown contact and ignored parked-middle observations. RED:8 failed/19 passed, all expected missing observer/caller failures. Run before implementation:
```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_grip_load_reference.py -q
```
Expected new tests fail for missing `observe`, while legacy8 stay green.
- [x] Implement `observe(loads_n,gaps_m,known)` as6-value elementwise min/max/AND plus a sample count. It must not change `held` until `update`.
- [x] At `update`, merge the completed window with the current snapshot; use the same existing contact and90% predicate. Copy the used arrays/count to JSON-compatible `last_window`, then consume/reset the window. No observed window retains legacy snapshot-only behaviour.
- [x] Run the same test command;27 passed in1.47s across both focused files.

## Task2: Wire actual physics feedback and preserve collection

Files: `scripts/verify_fixed_dual_panda_dh116.py`, `tests/test_dh116_contact_collection.py`.

- [x] Add failing caller test requiring `preload_target_hold.observe` after actual `previous_loads` production and before `grasp_sequence.update`, not inside the lower-rate MPC block. Require `preload_window` collection and a `preload_feedback` runtime identity.
- [x] Add a collection regression using the real hold window. It must preserve a copied window in `observation.goal` and leave existing267D student/581D five-node teacher feature vectors unchanged.
- [x] Add `hand_preload_window_report=None` at initialization. After a close-phase hold update, build a report from `last_window`, tagging its actual decision step and observed physics interval. Keep that report's source step unchanged during hold/lift.
- [x] Feed the observer once per actual close physics row, with the same loads/gaps/known contact feedback used by the physical gate:
```python
if args.hand_mpc and grasp_sequence.stage=='close':
    preload_target_hold.observe(previous_loads.reshape(2,3),
        np.asarray([samples[c]['gap_m'] for c in BODY_NAMES]).reshape(2,3),
        contact_valid_known.reshape(2,3))
```
- [x] Put the used report in `goal.preload_window`; tag `run_config.preload_feedback='completed_physics_window_v1'` for MPC runs. Do not alter the network/checkpoint dimensions or physical success observer.
- [x] Run affected regression and compile/diff checks:138 passed in7.52s; compile/diff passed. Inline diff review confirmed no changes to rod/physical gates/caps/assets:
```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_grip_load_reference.py tests/test_dh116_contact_collection.py tests/test_dh116_contact_mpc.py tests/test_dh116_integrated_hand_mpc.py tests/test_dh116_complete_contact_policy.py tests/test_dh116_contact_replay.py tests/test_dh116_touch_first.py tests/test_dh116_sequence_runtime.py tests/test_dh116_close_initialization.py -q
DH116_NUM_THREADS=1 bash scripts/python.sh -m py_compile scripts/verify_fixed_dual_panda_dh116.py src/go2_pvcnn/control/dh116/grip_load_reference.py
git diff --check
```
Known old full-suite fixture failures are not a new passing claim. Review the small diff against the unchanged physical gates/caps.

- [x] Additional causal mask replay on14701 actual old physics frames/148close decisions: old masks all match records; window helper additionally releases32decisions (left31/right1), using past samples only. No new actions or physical outcomes generated; this does not complete Task3's actual repaired-loop verification. Evidence is in V21 `preload_feedback_window_replay_v1.json`.

## Task3: Verify the actual repaired loop

- [x] Recheck actual V21 stage/PID. Original V21 enteredhold at14700/time1.4701s on its old loaded code; latest15238 holdelapsed.0538s, four contacts valid. Its manifest has no new feedback tag and existing data has no window field, so this is NOT validation of the patch. Leave this PID running through hold/lift; do not reload now.
- [x] Confirm V21 terminal/PID missing: actual27063/time2.7064s,liftFAIL after fullholdPASS. Isolated CUDA joint experiments subsequently reproduce a separate TGS16iteration/100µs integration freeze; both diagnostics also terminal. Details in complete-expert plan's final section. This does not qualify the window patch or a successful source.
- [x] Run single V22 bounded500step baseline with original16position iterations and every physical option, same active pose and actual prehold14699 SHA94426ef...5dea7. PID167123/session95002 naturalterminalexit0. All500rowchecks pass; summaryFalse/closeFAIL is incomplete50ms.5completeclose intervals/4actual100frame windows validate; no phase credit or successful training source. Original backups/cache settings unchanged.
- [x] Verify real first/control-window data:12D ownership, completed_physics_window_v1, actual counts/source steps and extrema. In V23decision300/400, recovered snapshots exceed.9N but preceding100frame minima remain below.9N; heldFalse prevents false capture. These specific examples were alreadyunheld, not an observedlatched→released transition; existing RED/GREEN and recorded causal replay test the same release predicate. Source metadata only uses completed physics frames.
- [x] After V22 terminal, follow the separate full-budget V23 numerical experiment to its real terminal16922: closePASS/full1s holdPASS/liftFAIL, right non-pad finger13 source interval bound .999886mm<1mm; all6pad regions pass. No new run/approach replay. Window wiring is executed, but this is not a successful expert or whole-wire-clearance source.
- [x] Update README, CHANGELOG and CODEX_RESUME with actual results and remaining work. Overall goal is notcomplete; currentlyblocked pending a separate palm/arm-posture design choice, not pending this already-implemented window repair.

## Delayed user confirmation received2026-10-22

User selected the older `call_ab79a295dd58419bad6911f33f44b6d9` window-feedback option. This confirms this plan only, not a new palm-posture change. Fresh inspection confirms `observe` at every actual close row and the existing100µs×100-frame/10ms decision window. V23decision300 recorded currentthumb .901272N but windowminimum .893021N and heldFalse; decision400 current .910818N/windowminimum .899197N and leftheldFalse/rightheldTrue. These are prevention-of-capture examples, not claims of a measuredlatched→released event. The existing real-interval-dip regression explicitly startslatched and asserts release.

Fresh targeted test on receipt:
```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_grip_load_reference.py -q
```
Actual14passed in.07s; no production source changes, physical parameter changes, goal restart, Isaac launch or oldV21 resurrection. Broader138/whole-suite results above remain historical, not newly executed.
