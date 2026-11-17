# R04-mask: masked 371D label ledger compatibility

Date: 2026-11-15. Base: `a6a4290407456b4ea2d707e27272885b74e95bc6`.
Worktree: `student-r04-mask-20261115`; branch: `codex/student-r04-mask-20261115`.
Status: CPU implementation tested; independent SPEC/QUALITY and Windows publication pending.

## Resulting behavior

This is the approved R04-mask compatibility task from the Student source-binding plan. It does not diagnose the previous full-finite label contract as a historical bug.

- `tabletop_stereo_supervision_dataset.validate_rows` requires a floating 371D label tensor and same-device boolean 371D validity. Only fields marked valid must be finite.
- `tabletop_stereo_teacher_collector.PreActionState.validate` applies the same label rule after the existing shape/dtype/device checks.
- No label is filled, normalized, replaced, or made valid. Mixed common66/unknown305 data retain all 305 NaNs and false bits through batching.
- `tabletop_stereo_student_cli._snapshot` and the original full371 Critic remain unchanged. Unknown labels can be stored, but this change does not feed NaNs into that Critic. Its existing finite unchanged native371 guard still rejects them.
- The actual sealed-source checks, independent holdouts, Teacher qualification, action ownership, and BC/DAgger admission are unchanged. This package does not implement R04-live producer witness.

Source changes are limited to seven added / three removed lines in the two validators. The other new files are the dedicated tests and this report. No physical parameters, action24 contract, model, controller, checkpoint, or success thresholds were changed.

## TDD and actual CPU evidence

Existing related baseline: **96 passed / 0 failed**, exit0, 22.25s. The original stdout is retained in the chat tool execution record; the private baseline summary is not a synthesized raw log.

New tests before implementation: **5 failed / 5 passed**, exit1, 2.69s. Failures demonstrated unknown labels rejected before their mask, the old Critic guard not yet reachable, mixed66/305 batching rejected, and mismatched label/mask devices accepted. The remaining five cases already protected valid nonfinite rejection.

After the minimal implementation: **106 passed / 0 failed**, exit0, 22.11s: 10 dedicated tests plus the same 96 existing tests. The command used:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -B -m pytest -q -p no:cacheprovider \
 tests/test_dh116_stereo_masked_label_compatibility.py \
 tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py \
 tests/test_dh116_stereo_student_cli.py tests/test_dh116_stereo_canonical_field_losses.py
```

Private evidence directory:
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/student-r04-mask-20261115-coordinator`.

| Actual evidence | SHA256 |
|---|---|
| red.log | 6cb531ccb2d34049667d5279eef42dbc07561c5762e8cb844bbaafb2da596078 |
| green.log | 7a7f32ecd065d75c82a9851da6db52e4e9d4e447f3318419ae52a1ca738ec37e |
| protection.log | 24dc08d4b877489d073352e38605a735c546c093da905e113e511e8f6994c39f |

The command JSON files record exact argv, cwd, exit, duration and raw stdout SHA. Tests run on CPU; the device mismatch test uses PyTorch's meta device, without CUDA execution.

`Repository.verify_protection()` verified all **528 protected files and two historical checkpoints**. Tracked diff was exactly the two validator paths; the legacy Critic file has no diff. `git diff --check` passed. Raw logs, fixture data, and checkpoints are not committed.

## Limits and next tasks

No Isaac/native execution, data collection, GPU work, model training, dependency changes, or success-pool promotion occurred. The fixtures are schema-only and keep native_qualified=false. Missing/invalid fields are not newly licensed for Actor input. Formal Student training remains gated by qualified Teacher and synchronized stereo data. R04-live remains independently pending; this task does not convert online DAgger into sealed/offline BC.

Next: independent reviews, integration CPU regression, Windows publication and remote SHA verification; continue the new Teacher context/collector/checkpoint chain in separate worktrees.
