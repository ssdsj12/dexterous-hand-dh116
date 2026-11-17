# Stable-origin live R0 bounded entry — 2026-11-14

CPU preparation only; no new native result is claimed here.

## Actual changes
- `verify_dh116_origin_r0.py`: independent worker/environment and scope; binds original atomic contact reader plus opt-in `OriginLiveR0Observer`, restores both bindings even on exceptions.
- `validate_dh116_origin_r0.py`: post-exit recomputation of reset-bound origin and every retained sample identity. Restored original measurements are independently checked against the atomic raw capture with the existing complete-pair validator.
- `run_dh116_origin_work_packages.py`: original budget, single-instance, natural-exit and GDB checks remain; origin scope is independently named and cannot grant Teacher/grasp/full-R0 admission.
- `configs/dh116_execution/origin_r0_native_v1.json`: immutable distinct recipe; original assets/inputs/budget retained; protected assets are not added as publishable source edits.
- `tests/test_origin_r0_native.py`: recipe fingerprints, source protection, exception restoration, budget rejection, complete real Python owner using CPU doubles and post-exit validation.

## CPU evidence
9 missing-entry tests failed before implementation. Final regression:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_origin_r0_native.py tests/test_r0_measurement_origin.py tests/test_r0_origin_replay.py tests/test_complete_pair_native_worker.py tests/test_complete_pair_native_audit.py tests/test_complete_pair_recipe_scope.py -q --tb=short
```
54 passed, 146.39 s. CPU fixture covers all 232 measured samples / 29 control windows; it is **not** native physics evidence.

Experiment code/config SHA:
`a60f8bbd4746d9a81112c5300ebf63bf79b05b7f01ed61d2e421fa070d8e81d0`.

## Frozen next native scope
GPU3, single Isaac, one environment, 0 PPO, 234 total SDK steps (2 setup + 232 measurement), at most 180 s, one attempt. Old complete-pair run is consumed and failed budget; not rerun. Existing passed native parameter and byte-exact-codec scopes supply only their qualified prerequisite evidence.

Standing user authorization “批准，后续自动批准” applies to this new frozen bounded recipe after Windows publication, source/protection checks and a resource-lock check. It does not authorize long PPO, enlarged budgets or weakened success criteria. Record complete GDB/outer-exit evidence, including failure, without force termination.

## Limits
Exact CPU replay established numeric/validity/guard/action equivalence and smaller storage, **not native speedup**. The previous run took 196.212 s and failed budget despite natural exit0. This new run must earn its own budget/lifecycle evidence. Original actions, controls, actual contact definition, physics and limits remain unchanged. Effective opposing contact, approach, hold, slip, instability and full training-state/reward admission remain unresolved.
