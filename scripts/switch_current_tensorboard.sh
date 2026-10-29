#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
run=$(cat current_training_run.txt)
case "$run" in runs/phase_force_below_axis_10240_2048_*|runs/free_grasp_lift_*) ;; *) exit 1;; esac
if kill -0 17414 2>/dev/null; then kill -TERM 17414; fi
nohup /opt/conda/envs/hand/bin/python -m tensorboard.main --logdir "/root/hand/$run/tensorboard" --host 127.0.0.1 --port 6006 --reload_interval 5 > /tmp/tensorboard_phase_force_10240.log 2>&1 < /dev/null &
printf 'TENSORBOARD_PID=%s\n' "$!"
