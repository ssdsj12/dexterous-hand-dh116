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
# Default to available cores; explicit per-run overrides remain supported.
read -r default_numeric_threads default_cpu_affinity < <("$hand_python" -c '
import os
from pathlib import Path
cpus = sorted(os.sched_getaffinity(0))
cores = set()
for cpu in cpus:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        cores.add(((topology/"physical_package_id").read_text().strip(), (topology/"core_id").read_text().strip()))
    except OSError:
        cores.add(("cpu", cpu))
print(max(1, len(cores)), ",".join(map(str, cpus)))
')
numeric_threads="${DH116_NUM_THREADS:-$default_numeric_threads}"
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
cpu_affinity="${DH116_CPU_AFFINITY:-$default_cpu_affinity}"
exec taskset -c "$cpu_affinity" "$hand_python" "$@"
