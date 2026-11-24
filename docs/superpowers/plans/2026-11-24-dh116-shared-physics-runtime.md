# DH116 shared physics runtime implementation plan

> Execute inline in this delegated work package. Parent explicitly approved the
> minimal inheritance design. Do not commit, run Isaac/GPU, or edit entry,
> geometry or lifecycle modules.

**Goal:** Teacher and Student action producers share one ordered physical owner,
source/guard/reward window and selected Reset path, without constructing a model
or PPO trainer in the shared runtime.

**Architecture:** Move the existing physical initialization and methods to
`tabletop_tensor_physics_runtime.py`. Split initialization into physical setup
and shared reward/reference setup so the Teacher adapter constructs its model
at the original point between those stages. Keep `sdk` as the only physical
substep chain. Capture is an optional observer, off by default, with owned
measured payloads, window-default cadence, bounded per-lane history and strict
clock/transaction/mutation fences. Hook exceptions and guard failures poison the
scope. Only a settled full eight-step window permits its done rows to Reset.

**Tech stack:** Python, PyTorch CPU tests, existing TensorTrainingReward and PPO.

- [x] Add failing `tests/test_dh116_tensor_physics_runtime.py` at the action-window
  API boundary: one ordered physical chain, policy-owned identities, no model,
  owned measured payloads, optional window capture, selected history clear,
  mid-window Reset rejection, original guard failure and hook violations.
- [x] Extract helper functions/physical methods without rewriting their numerical
  bodies. Keep compatibility imports in `tabletop_tensor_collector.py` and adapt
  the existing raw-clearance monkeypatch to the shared owner module.
- [x] Add `begin_action_window(actions, token, policy_uids, end)`,
  `advance_action_window(actions, end)`, `record_window_request()` and
  `complete_action_window(truncated=None, buffer_full=None)`; keep the original
  observe/begin/settle order and expose measured snapshots separately from any
  actor, model, reference context or critic API.
- [x] Add `attach_capture(capture, cadence="window", history_length=2)` and
  `capture_history(env_id)`. Require `on_measurement(event)` to return lane-keyed
  capture payloads. Call `reset_selected(ids,event)` only after physical Reset
  and peer proof, clear only those local histories, reject source/clock/owner/
  transaction/tensor mutation and reentrant calls, and propagate hook errors.
- [x] Reduce Teacher collector to original model/trainer construction,
  observation packing and window policy/value/record calls against shared APIs.
- [x] Run `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m pytest -q
  tests/test_dh116_tensor_physics_runtime.py tests/test_dh116_tensor_collector.py`
  then relevant source/control/reward suites. Record actual counts and limits in
  a new report. Parent owns source sealing and native equivalence verification.
