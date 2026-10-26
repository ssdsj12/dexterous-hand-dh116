#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if pgrep -f '[t]rain_dh116_tabletop_gpu_ppo.py|[r]ecovery_replay.py|[a]ction_capture.py|[t]ensor_fault_trace.py|[t]ensor_terminal_trace.py' >/dev/null; then
  echo 'An Isaac training or replay instance is already running.' >&2
  exit 1
fi
export DH116_NUM_THREADS="${DH116_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
output="${DH116_OUTPUT:-runs/tabletop_tensor_$(date +%Y%m%d_%H%M%S)}"
continuation=()
if [[ -n "${DH116_RESUME:-}" ]]; then
  if [[ "${DH116_FORCE_DIRECTION_REPOSITION:-0}" == 1 ]]; then
    echo 'DH116_FORCE_DIRECTION_REPOSITION=1 requires a fresh run without DH116_RESUME.' >&2
    exit 1
  fi
  continuation=(--resume "$DH116_RESUME")
fi
reposition_flag=(--release-reposition)
if [[ "${DH116_FORCE_DIRECTION_REPOSITION:-0}" == 1 ]]; then
  reposition_flag=(--force-direction-reposition)
fi
exec bash scripts/python.sh scripts/train_dh116_tabletop_gpu_ppo.py \
  --pose-summary configs/dh116_training/pose_reference.json \
  --adaptive-grasp --force-feedback --full-hand-contact \
  --joint-envelope configs/dh116_joint_envelopes/free_grasp_geometry_v1.json \
  --grasp-initialization configs/dh116_training/grasp_initialization_pos64.json \
  --load-profile configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json \
  --object-kind solid_cylinder --object-material extreme_grip --hand-material grip_compliant \
  --num-envs 4096 --rollout-steps 24 --physics-hz 480 --mimic-frequency 400 \
  --position-iterations 64 --velocity-iterations 16 --reset-height-jitter-m 0 --cpu-threads "$DH116_NUM_THREADS" \
  --headless --device cuda:0 --direct-training --tensor-pipeline --training-episode-resets --pressure-relief \
  --grasp-controller coordinated --palm-transport driven_carriage \
  "${reposition_flag[@]}" \
  --passive-velocity-limit-rad-s 2.97 --contact-reward-window close_start \
  --grasp-readiness measured_load --skip-hold --close-downward-penalty --grasp-shaping --vertical-force-reward --vertical-force-reward-mode continuous --force-reward-start downward_after_close \
  --episodes-per-env 1024 --output "$output" "${continuation[@]}" "$@"
