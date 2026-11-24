#!/usr/bin/env bash
# Source this file to use the existing installed environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' 'Use: source activate.sh' >&2
  exit 1
fi
DH116_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /mnt/mydisk/xk/envs/dh116/bin/activate || return 1
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export DH116_NUM_THREADS="${DH116_NUM_THREADS:-4}"
export XDG_CACHE_HOME="$DH116_ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv/ComputeCache"
export TMPDIR="$DH116_ROOT/runs/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTEST_DEBUG_TEMPROOT="$TMPDIR"
mkdir -p -- "$XDG_CACHE_HOME" "$TMPDIR" || return 1
cd -- "$DH116_ROOT" || return 1
