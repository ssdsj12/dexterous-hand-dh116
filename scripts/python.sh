#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hand_python="${DH116_PYTHON:-/home/xk/miniconda3/envs/go2/bin/python}"
python_prefix="$("$hand_python" -c 'import sys;print(sys.prefix)')"
isaac_root="$python_prefix/lib/python3.11/site-packages/isaacsim"
usd_candidates=("$isaac_root"/extscache/omni.usd.libs-*.cp311)
test -d "${usd_candidates[0]}"
export PYTHONPATH="$project_root/src:$project_root:$project_root/vendor/DeformX:${usd_candidates[0]}"
export LD_LIBRARY_PATH="$python_prefix/lib:${usd_candidates[0]}/bin:$isaac_root/kit${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
numeric_threads="${DH116_NUM_THREADS:-}"
if [[ -n "$numeric_threads" ]]; then
  export OMP_NUM_THREADS="$numeric_threads"
  export OPENBLAS_NUM_THREADS="$numeric_threads"
  export MKL_NUM_THREADS="$numeric_threads"
  export NUMBA_NUM_THREADS="$numeric_threads"
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
fi
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export DH116_TABLE_SOURCE_DIR="$project_root/resources/manufacturer"
cpu_affinity="${DH116_CPU_AFFINITY:-0}"
exec taskset -c "$cpu_affinity" "$hand_python" "$@"
