# Stereo Phase 1 integration: independent privileged PPO Teacher

Software is implemented and CPU-tested. **Native Teacher qualification is pending.** This phase has not run Isaac, trained a real task model, collected a new successful demonstration or modified any existing controller/checkpoint.

See [core implementation report](../../privileged_teacher_phase1_report.md) for five modules, network, saved-latent PPO, independent Actor BC, reward passthrough and resume arithmetic. Added integration files are `tabletop_stereo_contract.py`, `tabletop_stereo_native.py`, `tabletop_privileged_cli.py`, three privileged entry scripts, independent native bootstrap config and integration tests.

## What runs

The new Teacher has 371D inputs, independent 256/256 Actor/Critic and canonical 24D Tanh Gaussian actions. The standalone loop collects its own policy samples, calls the unchanged native `TensorEnv.step`, uses the already-scaled reward exactly once, checks reward-term closure, preserves latent/log-prob identity and saves update-specific checkpoints exclusively. Resume is into a new output directory. Intermediate checkpoints are retained for later independent validation selection; final-update performance is not assumed best.

The native bridge uses the existing `scripts/python.sh` environment and original scene launcher. Only the final new-run training callback is replaced in-process; original Task/World/Env methods and every physical guard are unchanged. Linux single-instance lock and process detection reject another native job. The observer records live 480Hz samples before reset and retains the first terminal against later no-ops. Runtime capture alone is explicitly **unqualified**, with support/force qualification set to NA.

All native paths require both `--enable-experiment` and `--execute-native`; importing modules, showing help and checking config cannot launch Isaac. Candidate collection keeps sampled/mean/issued actions distinct and does not admit itself to successful expert BC. Formal BC additionally requires the raw-evidence verifier; a summary claiming native success is insufficient.

## Actual CPU verification

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_contract.py
```

Result: **49 passed in 6.10s, exit 0**. CPU integration performs two synthetic optimizer updates, exclusive checkpoint saves and a third resumed synthetic update. Those fixtures are not native episodes. Terminal observer fixtures test preservation, not PhysX behavior. No new native task tests were run. The separate core regression in its report has overlapping tests and must not be added to this count.

Actual entrypoint config inspection:

```bash
python scripts/train_dh116_privileged_teacher.py --config configs/dh116_training/privileged_teacher_ppo_v1.json --check-config
```

Native command template (not executed):

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python bash scripts/python.sh scripts/train_dh116_privileged_teacher.py --config configs/dh116_training/privileged_teacher_ppo_v1.json --output runs/NEW_EXCLUSIVE_TEACHER_RUN --seed 116 --num-envs 2 --updates 1 --enable-experiment --execute-native
```

## Remaining gates

- The native bootstrap, installed SDK/RSL dependencies and 480Hz observer path need actual server validation; CPU import tests do not prove installed Isaac compatibility.
- R0 is the only production CLI profile. R1/R2 source qualification is absent. R2 currently implements geometry plus support-deficit subset; contact recovery/balance/overload/target-change factors are not claimed implemented or qualified.
- Full first-episode Teacher success per grasp, native source attribution and held-out validation must pass before formal Student DAgger. No historical 261D/26D artifact is silently imported.
- CPU software readiness permits isolated Phase 2–7 implementation under the user's follow-up rule; it does not authorize promotion through these pending native gates.
