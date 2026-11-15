# C05 read-only checkpoint audit: expected origin binding

Date: 2026-11-15. Scope: CPU fixture software only. Base: clean `51ee707ec5b36a5a02ef961f6b37e1cc9b0ca666` in `codex/teacher-c05-training-20261115`. The previous C05 report is unchanged. Only the existing checkpoint module, its existing test file and this new report are in scope.

A C06 artifact can otherwise claim a fresh run while carrying a structurally valid resume checkpoint. The read-only audit now accepts a keyword-only origin expectation:

```python
audit_checkpoint_bytes(raw, trainer, contract, *, expected_origin=None)
```

Default `None` keeps the previous behavior. A non-None expectation first passes the existing `_origin` format validator. The single immutable `BytesIO` CPU/weights-only deserialization and full existing `_validate_payload` still run, then payload origin must equal the expected origin or the audit raises `ValueError`. No second deserialization or duplicated payload validator was added. C06 A can pass `expected_origin={'kind': 'fresh'}`; a future resume caller can bind `{'kind': 'resume', 'checkpoint_sha256': actual_A_sha256}`. Origin matching itself does not authorize a resume experiment or replace C06's update/budget/source checks.

The implementation changes exactly three source lines: signature, expected-origin format check, and comparison after complete payload validation. All other nine function source segments remain exactly equal to the base, including `_origin`, `_validate_payload`, load and save. The checkpoint format/version/device and the nine returned metadata fields are unchanged. No origin field or Tensor/payload/model/optimizer is exposed. Audit still does not clean/restore/update, change pending or rollout, restore global RNG, open a checkpoint path, or launch an App. Existing source-identity SHA reads remain required; the current checkpoint module SHA naturally changes without rewriting any saved historical checkpoint.

## TDD and CPU verification

RED ran before the keyword was implemented on a real saved fixture. Its default call and unchanged-state check passed, then explicit `expected_origin=None` raised the expected unexpected-keyword `TypeError`: **1 failed in 10.08s**, exit 1. `origin-audit-red.log` SHA256: `46ace9b01d53e3f5b0db03b13bac9d9b15f623207962ae757068a8296b684968`.

One targeted GREEN command completed: **19 passed, 70 deselected in 99.36s (0:01:39), zero failed, exit 0**. Exact command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_checkpoint.py -k 'audit_expected_origin or test_audit_checkpoint_bytes_returns_owned_metadata_without_any_live_mutation or test_resume_origin_and_state_bind_single_immutable_file_snapshot or test_deserialize_uses_exactly_one_read_bytes_snapshot'
```

The twelve new cases cover fresh and syntactically valid resume payloads: default, explicit None and matching expectations return identical exact-nine metadata; a fourth positional argument is rejected; resume-as-fresh, fresh-as-resume and a different resume SHA are rejected only after complete payload validation; seven malformed expected-origin shapes fail before deserialization. Resume variants are locally reserialized test variants of the genuinely saved CPU fixture, not native/resume experiment evidence. Every call checks model/Adam, counters/version, populated markers, all three CPU RNGs, real pending contents/identity and nonempty rollout contents/identity. Expectations themselves remain unmodified, and each successful/valid-payload rejection uses exactly one load of the supplied immutable bytes.

The seven selected existing cases cover both present/absent-marker read-only metadata tests, all four delete/replace immutable-snapshot restore boundaries and the loader's single-read test. The previous 129-case result retains its earlier source provenance and was not repeated or relabeled here. No new failure appeared in this successor; root will run its required full affected integration suite.

Private evidence directory: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/c05-training-20261115`. Actual argv, environment, timestamps, exit status and tested source/test SHA are retained in `origin-audit-green.command.json` / `.result.json`. `origin-audit-green.log` SHA256: `68e6934e1b56d864c35afb878bef3c959c1dae07b7385f14ea6dcdddae983011`. Tested checkpoint source SHA256: `2b26625a62bb8cad74280d2c0ca07f29af418d3b0d4a3b20d67e4f564444c021`. `origin-audit-design.json` records unchanged function segments and load/save body hashes.

After this report was written, `Repository.verify_protection()` passed with the actual registry counts **528 protected source/config files + 2 historical checkpoints**, retained in `origin-audit-protection.log`. In-memory Python compilation and explicit three-path unstaged/staged diff checks passed; evidence is in `origin-audit-diffchecks.json`. Final status is a local CPU candidate for root review/integration/publication. Actual native/App/GPU/native-training/install/push counts are all **0**. Native qualification remains **false**; no actual native worker entry point, physical continuation or C06 worker lease is supplied. Historical models and `11-10` are untouched.
