#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tag="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
exec bash scripts/python.sh scripts/train_dh116_tabletop_gpu_ppo.py \
 --pose-summary configs/dh116_training/pose_reference.json --adaptive-grasp --force-feedback --full-hand-contact \
 --grasp-initialization configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json \
 --load-profile configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json \
 --object-kind solid_cylinder --object-material extreme_grip --hand-material grip_compliant \
 --num-envs 2 --rollout-steps 16 --max-iterations 1 --physics-hz 480 --mimic-frequency 400 \
 --solver-type 0 --position-iterations 64 --velocity-iterations 16 --reset-height-jitter-m 0 --cpu-threads 4 \
 --headless --device cuda:0 --direct-training --tensor-pipeline --training-episode-resets --pressure-relief \
 --grasp-controller policy --palm-transport driven_carriage --palm-exploration-axis xyz \
 --passive-velocity-limit-rad-s 2.97 --contact-reward-window close_start --grasp-readiness measured_load \
 --skip-hold --close-downward-penalty --grasp-shaping --vertical-force-reward \
 --vertical-force-reward-mode continuous --force-reward-start downward_after_close \
 --force-region-control --phase-force-reward --free-grasp-lift --continuous-motion --free-wrist-grasp \
 --surface-force-sharing --end-support-clearance-m .06 --surface-diagnostic-steps 512 \
 --output "runs/surface_smoke_${tag}" --kit_args=--allow-root
