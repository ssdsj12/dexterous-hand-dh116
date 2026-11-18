#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
count="${1:-10240}"
iterations="${2:-2048}"
run="runs/free_grasp_lift_${count}_${iterations}_$(date +%Y%m%d_%H%M%S)"
export DH116_RESUME="${DH116_RESUME:-}"
export DH116_OUTPUT="$run"
export DH116_MAX_ITERATIONS="$iterations"
export DH116_FREE_GRASP_LIFT=1
export DH116_PHASE_FORCE_REWARD=1
nohup bash scripts/train_dh116_tensor_ppo.sh --num-envs "$count" > "/tmp/$(basename "$run").log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$run" > /root/hand/current_training_run.txt
printf 'RUN=%s\nPID=%s\n' "$run" "$pid"
