#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
run="runs/phase_force_below_axis_10240_2048_$(date +%Y%m%d_%H%M%S)"
export DH116_RESUME=
export DH116_OUTPUT="$run"
export DH116_MAX_ITERATIONS=2048
export DH116_PHASE_FORCE_REWARD=1
nohup bash scripts/train_dh116_tensor_ppo.sh --num-envs 10240 > "/tmp/$(basename "$run").log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$run" > /root/hand/current_training_run.txt
printf 'RUN=%s\nPID=%s\n' "$run" "$pid"
