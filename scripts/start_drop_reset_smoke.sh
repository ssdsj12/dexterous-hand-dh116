#!/usr/bin/env bash
set -euo pipefail
cd /root/hand
nohup bash scripts/python.sh scripts/smoke_object_drop_reset.py > /tmp/object_drop_reset_8env_smoke_v2.log 2>&1 < /dev/null &
printf 'PID=%s\n' "$!"
