#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
nohup bash scripts/python.sh scripts/probe_fixed_pose_positions.py > /tmp/fixed_orientation_force_probe_v2.log 2>&1 < /dev/null &
printf 'PID=%s\n' "$!"
