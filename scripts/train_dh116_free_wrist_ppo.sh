#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if pgrep -f '[t]rain_dh116_tabletop_gpu_ppo.py|[p]robe_continuous|[d]iagnose_lift' >/dev/null; then
  echo 'An Isaac training or diagnostic instance is already running.' >&2
  exit 1
fi
export DH116_NUM_THREADS="${DH116_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
output="${DH116_OUTPUT:-runs/free_wrist_$(date +%Y%m%d_%H%M%S)}"
resume=()
if [[ -n "${DH116_RESUME:-}" ]]; then resume=(--resume "$DH116_RESUME"); fi
exec bash scripts/python.sh scripts/train_dh116_tabletop_gpu_ppo.py \
  --pose-summary configs/dh116_training/pose_reference.json \
  --adaptive-grasp --force-feedback --full-hand-contact \
  --grasp-initialization configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json \
  --load-profile configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json \
  --object-kind solid_cylinder --object-material extreme_grip --hand-material grip_compliant \
  --num-envs "${DH116_NUM_ENVS:-64}" --rollout-steps 64 --max-iterations "${DH116_MAX_ITERATIONS:-128}" \
  --physics-hz 480 --mimic-frequency 200 --position-iterations 64 --velocity-iterations 16 \
  --reset-height-jitter-m 0 --cpu-threads "$DH116_NUM_THREADS" --headless --device cuda:0 \
  --direct-training --tensor-pipeline --training-episode-resets --pressure-relief \
  --grasp-controller policy --palm-transport driven_carriage --palm-exploration-axis xyz \
  --passive-velocity-limit-rad-s 2.97 --contact-reward-window close_start --grasp-readiness measured_load \
  --skip-hold --close-downward-penalty --grasp-shaping --vertical-force-reward \
  --vertical-force-reward-mode continuous --force-reward-start downward_after_close \
  --force-region-control --phase-force-reward --free-grasp-lift --continuous-motion --free-wrist-grasp \
  --checkpoint-every 32 --output "$output" --kit_args=--allow-root "${resume[@]}" "$@"
