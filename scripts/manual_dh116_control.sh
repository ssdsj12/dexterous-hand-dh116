#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export DH116_NUM_THREADS="${DH116_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
output="${DH116_OUTPUT:-runs/manual_dh116_$(date +%Y%m%d_%H%M%S)}"
initialization="${DH116_GRASP_INITIALIZATION:-configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json}"
profile="${DH116_LOAD_PROFILE:-configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json}"
exec bash scripts/python.sh scripts/train_dh116_tabletop_gpu_ppo.py \
  --pose-summary configs/dh116_training/pose_reference.json \
  --adaptive-grasp --force-feedback --full-hand-contact \
  --joint-envelope configs/dh116_joint_envelopes/free_grasp_geometry_v1.json \
  --grasp-initialization "$initialization" --require-initial-below-axis \
  --load-profile "$profile" \
  --object-kind solid_cylinder --object-material extreme_grip --hand-material grip_compliant \
  --num-envs 1 --rollout-steps 24 --physics-hz 480 --mimic-frequency 200 \
  --position-iterations 64 --velocity-iterations 16 --reset-height-jitter-m 0 --cpu-threads "$DH116_NUM_THREADS" \
  --device cuda:0 --direct-training --tensor-pipeline --training-episode-resets --pressure-relief \
  --grasp-controller policy --palm-transport driven_carriage --palm-exploration-axis xyz \
  --passive-velocity-limit-rad-s 2.97 --contact-reward-window close_start --grasp-readiness measured_load \
  --skip-hold --close-downward-penalty --grasp-shaping --vertical-force-reward \
  --vertical-force-reward-mode continuous --force-reward-start downward_after_close \
  --grasp-acquisition-shaping --policy-lift-hold --stable-lift-reward --force-region-control --phase-force-reward \
  --manual-control --output "$output" --kit_args=--allow-root "$@"
