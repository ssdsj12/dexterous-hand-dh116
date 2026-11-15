# R03 Teacher expert source binding

Base: `d1fa6630f76420177a93cda7c24de25d97bf20af`
Branch: `codex/repair-r03-source-20261115`
Scope: software/CPU source admission only. No native run, GPU, training experiment, push, or physical-success claim.

## Result

`validate_bound_expert_dataset` first runs the unchanged metadata/native-proof body, then replays supervised sources exactly once and returns `(payload, bindings)`. The existing public validator still returns the payload. `read_expert` preserves its tuple API and calls that bound validator once; each independent production `ActorBCTrainer.step` revalidates its mutable dataset through the public wrapper.

Replay retains SHA verification, `weights_only=True` loading, per-admission source caching, contiguous policy ticks and bindings. It additionally reopens qualification evidence, verifies the ledger path/SHA belongs to that qualification's actual process completion with `_process_output`, reads live raw physical samples, and reuses assembly `_episode_source_rows` for canonical actions and original issue/physics clocks. Observation/action source slices must match both dtype and values. Every canonical issued-action mask entry must remain true. Sources with 3844D observations or noncanonical action contracts are refused; no observation conversion is introduced.

`train_teacher` and its CLI accept `expert_dataset_sha256` / `--expert-dataset-sha256`. Expert path/SHA pairing and byte hashes are checked before native startup or model/optimizer construction. Full identity-bound admission runs once the bridge identity is available, before model construction. CLI expert loading uses `read_expert`, while exploratory PPO without BC still requires no expert proof. Existing 371D/24D/Tanh architecture, PPO algorithm and historical checkpoints are unchanged. Existing no-expert callers remain valid; expert callers must now supply the reviewed SHA alongside their path. The payload-only validator and tuple-returning reader keep their public return types.

## Test scope and evidence

The shared `CPU_only_gate` now mocks only LOWER native qualification. The process-completion path/SHA check, ledger deserialization, tensor comparisons, mask validation, and physical source join all run real production code. Synthetic artifact metadata cannot establish native qualification.

The initial RED run (new source tests plus CLI suite) reported **26 failed, 12 passed**. All six observation/action/mask/order/observation-dtype/action-dtype mutations specifically failed with `DID NOT RAISE`, demonstrating the old admission hole; no import error is used as its evidence. Other RED cases expose missing bound replay or the new CLI argument/early rejection.

First GREEN attempt reported **1 failed, 94 passed** because the new intact CLI test supplied a minimal assembly identity to the real checkpoint writer (`KeyError: source_sha256`). The test was corrected to assemble against the actual training identity; checkpoint behavior was not changed. The subsequent run passed 99 tests. Final expanded coverage includes raw issued-action/terminal/policy/physics-clock alterations, source contract reopening, explicit bound-validator tuple behavior, repeat admission counts, direct BC and CLI tampering, and intact BC/PPO/checkpoint integration.

Final command:

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_teacher_expert_source_binding.py tests/test_dh116_teacher_expert_assembly.py tests/test_dh116_privileged_cli.py tests/test_dh116_teacher_warm_start.py tests/test_dh116_privileged_teacher.py
```

Actual final result: **105 passed, 0 failed** in 6.20s. The original warm-start fixture needed an explicitly approved scope extension: its success path previously mocked the public metadata validator and supplied a ledger without qualification/process/contract evidence. It now uses the real assembly fixture and only the shared lower-native mock.

Evidence directory, outside Git: `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r03-source-20261115`.

| Evidence | SHA256 |
|---|---|
| `red.log` | `a1e8611c6b33bbf3d620c261bad422dd5e2512c803b9bfdd64077452382ffc7c` |
| `green-attempt1.log` | `ac995d90edd06954359067498adf1b4badd4b282906e741c1e45231a98be6426` |
| `green.log` | `aa7dab397f9a0e8ae1ea1f7e399be723279c7a358fb138da0004ca1239a4cc44` |
| `green-final.log` | `af91454c9a5b26dffa1fc48464941a1a438e5335087f1e5241c09496645c1913` |
| `protection.json` | `32c71b98cba1314df9d1d69a57adb4a9aa7169709f68bcc035ef661a9c647e39` |

`Repository(Path.cwd(), evidence_dir=Path(E)).verify_protection()` passed: **528 protected source/script/config files and 2 historical checkpoints**. `git diff --check` and `git diff --cached --check` passed before commit.

## Exact files

- `src/go2_pvcnn/control/dh116/tabletop_privileged_dataset.py`
- `src/go2_pvcnn/control/dh116/tabletop_privileged_warm_start.py`
- `src/go2_pvcnn/control/dh116/tabletop_privileged_cli.py`
- `tests/test_teacher_expert_source_binding.py`
- `tests/test_dh116_teacher_expert_assembly.py`
- `tests/test_dh116_privileged_cli.py`
- `tests/test_dh116_teacher_warm_start.py` (approved necessary fixture adaptation)
- `docs/superpowers/reports/2026-11-15-r03-teacher-source-binding.md`

## Limits and next gate

CPU tests prove admission/serialization/optimizer wiring only. No actual native qualification, physical grasp result, or new production model is established. The existing native verifier remains mandatory for real inputs. Independent specification and quality review precede integration and release; this task leaves a local commit only.
