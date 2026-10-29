#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
export DH116_RESUME=
export DH116_OUTPUT=runs/below_axis_2mm_8env_smoke_v1
export DH116_MAX_ITERATIONS=1
nohup bash scripts/train_dh116_tensor_ppo.sh --num-envs 8 > /tmp/below_axis_2mm_8env_smoke_v1.log 2>&1 < /dev/null &
printf 'PID=%s\n' "$!"
