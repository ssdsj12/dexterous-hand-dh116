# DH116 P0 independent proof for the corrected thumb budget

Base `2ef7f90ed9af9039dd32fb197063040ae11996f3`. Status `cpu_verified`. New native runs, PPO updates and evaluation episodes for this work package all equal zero. No native exit code or new model applies.

## Frozen budget and evidence

The two one-environment recipes each specify 2 initialization ticks, 0 warm-up ticks, 238 measurement ticks and 0 other physics-advancing ticks. The total is 240 per side, or 480 for a serial batch. The original command curve still runs phases 1–238. The planned maximum is 180 wall-clock seconds per side on GPU 3, with zero PPO updates. The exact recipe and code/config digests require new authorization before either launch. The right depends on the new successful left receipt. The old 484-step attempts and original receipt bytes are retained as historical diagnostic evidence with failed budget compliance.

The entrypoint and scheduler use `native_physics_budget()` to calculate all four phase counts and their sum. The entrypoint derives measured counts from the SDK counter and records the final SDK clock again while Kit physics is frozen and before clearing its context. The independent validator reconciles that event, SDK first/last and total counters, and every complete measured row. It verifies that the rod ledger advances 12 substeps per successful `RopeBatch.step_free()` call. The rod substep field is the cumulative issued count, supported by the CUDA solver's `step_free` loop, while `sdk_step` is the PhysX callback counter.

Each row must identify the recipe's side, actual DH116 thumb link, rod UID and registered mesh slot. Node contact indices are independently recalculated from the recorded world-force vectors for the 21-node rod. The raw teardown trace must show the frozen matching SDK clock, SDK view release on STOP and return from App close in that order. The App close marker must carry no pending callback exception. Existing scheduler checks require natural outer exit 0, the observed native flock, and a clean GDB inferior for these recipes. The right entrypoint also revalidates the left artifacts with the current left recipe before starting.

## Recalculate fingerprints on the final integration tree

Run from the Ubuntu integration worktree after all source and test changes have been merged. These commands only inspect recipes and protection hashes.

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh - <<'PY'
from pathlib import Path
from scripts.run_dh116_work_packages import Repository

repo = Repository(Path.cwd())
repo.verify_protection()
packages = repo.registry()['packages']
for side in ('left', 'right'):
    name = f'flex.f4-{side}-contact-native'
    recipe = repo.native_recipe(packages[name])
    print(name, {key: recipe[key] for key in ('command_spec_sha256', 'code_config_sha256')})
    print(recipe['spec']['budget'], recipe['spec']['validator']['physics_clock']['budget'])
PY
```

The command-spec digest hashes the exact JSON bytes. The code/config digest hashes the canonical combination of that digest and the pinned repository source digests. Recipe validation also checks the declared external input digests. A report-only or readiness change does not replace the recipe fingerprints. The execution scheduler must compare the approved digests with the final checkout and validate available resources before dispatch.

## CPU results

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_work_packages.py
```

78 passed in 16.17 seconds. The initial negative tests showed that nine tampered budget, rod, contact or teardown cases were accepted by the old validator. The revised validator rejects them, and accepts complete CPU fixtures for both sides. The SDK lifecycle, abbreviated legacy recipe format, stale approval, binding, outer exit and native lock regressions also pass. All 528 protected files and two historical checkpoints match.

A read-only audit verified that the historical raw ledgers remain compatible with the per-row clock and contact schema and still contain 68 contact-bearing rows per side. Those records span 242 actual SDK ticks per side and confer no new qualification. Planning with the old approval finds the new left awaiting approval and the new right awaiting the new left evidence. The actual corrected 480-step budget, native teardown and both sides' contact are pending a separately authorized native batch.
