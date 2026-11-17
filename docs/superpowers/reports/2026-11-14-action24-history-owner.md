# New controller action24 history and issued-action binding

Base: 01a9ae25a16ee04b83709c77eaffa27fe5b73219. Independent opt-in package; no changes to the original controller, collector, reward/configuration, checkpoint or 371D validity mask. No native launch, checkpoint training or optimizer update.

The flexible DualHandPhysicalEpisode/FlexiblePhysicalExchange path has no TensorTask.actions/previous_actions buffer. Existing PolicyOwnerSource observes the real TensorTask and cannot legitimately observe this different controller. Historical reset callbacks therefore do not provide a previous normalized action; SDK target vectors cannot reconstruct that value. This package does not backfill any historical record.

## Implemented boundary

Action24HistoryOwner owns one new controller-memory buffer for one fresh reset. The caller must explicitly provide a finite zero float32 [1,24] seed and source, matching the original reset action semantics. The seed is copied separately from the caller, digest checked, and compared with the actual new buffer write/readback. One owner cannot be reused across resets. The nested reset receipt binds the full clock/epoch, owner UUID, source seed SHA and preserved original controller receipt. An original status of unavailable remains unavailable.

The memory is controller state, not measured pose, force, SDK state or native reset qualification. It adds no valid observation fields: the existing measured 66-of-371 projection is unchanged.

The exact integration order is:

1. Construct the owner before the new reset; wrap only the controller callback with wrap_controller_reset(original). Pass that wrapper in the original owner_resets map.
2. Wrap the existing durable physical recorder with wrap_recorder(original). Deliver the original setup, reset, initial_geometry and physical callbacks. The original recorder is called first, before validation can raise. A storage or validation exception poisons this owner; the original physical episode closes on the exception. Callback return values are not treated as stop signals. Durability remains the original recorder's responsibility.
3. At the original pre-action boundary, call capture_pre_action(pre), run the existing observation adapter and trainer.collect_step, then begin(pre, trainer). The pre-action bytes must still match. The original trainer pending sample is bound to token, session, policy version, complete pending-data hash, model/checkpoint namespace and current model-weight hash.
4. Preserve the existing physical journal begin/end calls (the recorder wrapper does not replace them). Run the original physical control. Each actual request.action24 must exactly equal the sampled float32 values represented as float64. Original _Sequence checks clock/episode/8-step order and partial terminal settlement; original _physical_source checks same hand owner, full q/qd continuity and actual backend target readback.
5. Call end(result), then issued_for_pending(trainer, settlement). Pass its returned actual-request-derived float32 tensor as executed_actions to the original trainer.record_step with the same collection token. Caller must independently supply admitted reward/termination/bootstrap semantics.

A terminal partial window keeps exactly its actual rows; no padding to eight or no-op replacement is permitted. All failure rows delivered to the wrapper reach the prior recorder before validation. The settlement is an owned copy; mutation, reuse, changed token/session/epoch/policy weights or namespace is rejected.

The normalized issued action and executed dimensional target are separate. Every row retains hashes of the existing full physical row, expected backend targets and wrist projection. No target is re-encoded into a PPO action. In the CPU boundary fixture, an actual negative requested Z is projected to +1.6689300497141346e-8 m while its normalized policy action remains negative. This small positive value comes from the original float32 readback/float64 seed arithmetic, not a changed 2.1 mm threshold.

The owner binds the immutable sampled observation payload but does not re-prove that the observation adapter packed that payload from the measured pre-action source. That remains the existing actual model adapter/collector's responsibility. It does not admit the new backend, prove mesh reset health, establish R0 reward or enable the denied actual collector. Model-weight hashing at begin/settlement is a CPU integrity check, not a claim of production-rate PPO performance.

## CPU verification

Initial red: 24 failed because the new module did not exist. Initial implementation found the existing model_identity is a property, not a callable; corrected to use that property. The later projection fixture initially expected exact zero, while the unchanged original function produced the measured positive rounding value above; the test now requires nonnegative projected Z and an actual difference from the negative request.

Final command, run inside this Ubuntu worktree:

    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_action24_history_owner.py tests/test_dh116_no_camera_collector.py tests/test_dh116_readonly_control_cost.py tests/test_deformable_r0_replay.py

Result: 132 passed in 67.60 seconds (30 new, 102 existing). No native or optimizer updates. CPU fixtures construct a temporary original trainer/Adam object, sample and store transitions, but never call update or optimizer.step. The two-window test verifies all temporary model weights remain unchanged. Synthetic stored reward zero is solely a storage fixture, never an R0 reward; all owner receipts retain training_reward=None.

Covered negatives: seed source/dtype/shape/finiteness/range/nonzero; caller seed aliasing and internal buffer mutation; changed pre-action clock/value; historical/foreign reset receipt and repeated reset; altered token/session/epoch/namespace/pending action; actual request mismatch, actual target mismatch, duplicate/wrong-clock physical row; incomplete or falsely terminated settlement, mutated settlement actions/targets; terminal overwrite/reuse. A separate test exercises the original safety projection at its floor.

Evidence directory: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/action24-history-owner-20261114

- red.log SHA256 c9e37379a021ac6f741288b0b2d5cbc9e877fab024d1ac69203261dea4f4ae06
- regression.log SHA256 5eb41a370289a81ab9777531316ba073d0c651e0a6c42131030dc7ee2682d569
- cpu-example.json SHA256 b640f8f3e575ab393ef6153c2e33ea24bd0e5abf7558b274deb5aa63587bb657

The example was independently generated with the same test fixture and begin/settle helpers: two original 8-substep CPU-double windows, 16 CPU samples, 2 original PPO storage transitions, zero optimizer updates, unchanged temporary model weights. It records reset memory, both histories, pending bindings, first actual request, issued actions and settlement hashes. It is explicitly a CPU fixture, not a native receipt or historical first-action repair.

Runtime source SHA256:
- tabletop_action24_history_owner.py: 83203b7d1418df5524860ba5cef4e90a32732716351ce06191fd10d6ed0cd376
- test_action24_history_owner.py: 12f10265e0cb9acc83bc6fd5eaa3e57bf8f9838eca0ed0b1067c6d0d0adb5968

Reused unchanged implementation SHA256:
- tabletop_deformable_dual_evidence.py: 79d3b9b2881df6ab4345410dab7f49e3f60e66445699d99d752d6b8b16bcbb16
- tabletop_readonly_control_cost.py: dd9f37e18770329e8635d9493153b65e6c0916bcb06f0e89c1bceb6b8b881616
- tabletop_deformable_privileged_ppo.py: 7e1f4a9fd36244d990391bcf42682b7d92fae54222c8fcc70798d338546a8614
- tabletop_privileged_ppo.py: f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854

Physics peer was unavailable; no peer review is claimed. Parent review/publication is separate. The next execution change is wiring these opt-in hooks into a separately reviewed controller/collector entry after its source/reset/reward admission is satisfied; no current native recipe or frozen profile is changed here.
