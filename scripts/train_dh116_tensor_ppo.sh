#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if pgrep -f '[t]rain_dh116_tabletop_gpu_ppo.py|[r]ecovery_replay.py|[a]ction_capture.py|[t]ensor_fault_trace.py|[t]ensor_terminal_trace.py' >/dev/null; then
  echo 'An Isaac training or replay instance is already running.' >&2
  exit 1
fi
export DH116_NUM_THREADS="${DH116_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
kit_flags=()
if [[ "$EUID" == 0 ]]; then
  kit_flags=(--kit_args=--allow-root)
fi
output="${DH116_OUTPUT:-runs/tabletop_tensor_$(date +%Y%m%d_%H%M%S)}"
initialization="${DH116_GRASP_INITIALIZATION:-configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json}"
continuation=()
if [[ -n "${DH116_RESUME:-}" ]]; then
  if [[ "${DH116_FORCE_DIRECTION_REPOSITION:-0}" == 1 ]]; then
    echo 'DH116_FORCE_DIRECTION_REPOSITION=1 requires a fresh run without DH116_RESUME.' >&2
    exit 1
  fi
  continuation=(--resume "$DH116_RESUME")
fi
controller="${DH116_GRASP_CONTROLLER:-policy}"
reposition_flag=()
palm_axis_flag=()
acquisition_flag=()
lift_hold_flag=()
stable_lift_flag=()
force_region_flag=()
phase_force_flag=()
free_grasp_flag=()
iteration_flag=()
if [[ -n "${DH116_MAX_ITERATIONS:-}" ]]; then
  iteration_flag=(--max-iterations "$DH116_MAX_ITERATIONS")
fi
if [[ "$controller" == policy ]]; then
  palm_axis_flag=(--palm-exploration-axis xyz)
  if [[ "${DH116_GRASP_ACQUISITION:-1}" == 1 ]]; then
    acquisition_flag=(--grasp-acquisition-shaping)
  fi
  if [[ "${DH116_POLICY_LIFT_HOLD:-1}" == 1 ]]; then
    lift_hold_flag=(--policy-lift-hold)
  fi
  if [[ "${DH116_FREE_GRASP_LIFT:-1}" == 1 ]]; then
    free_grasp_flag=(--free-grasp-lift)
    lift_hold_flag=()
  fi
  if [[ "${DH116_STABLE_LIFT_REWARD:-1}" == 1 ]]; then
    stable_lift_flag=(--stable-lift-reward)
  fi
  if [[ "${DH116_FORCE_REGION_CONTROL:-1}" == 1 ]]; then
    force_region_flag=(--force-region-control)
    if [[ "${DH116_PHASE_FORCE_REWARD:-1}" == 1 ]]; then
      phase_force_flag=(--phase-force-reward)
    fi
  fi
fi
if [[ "$controller" == coordinated ]]; then
  reposition_flag=(--release-reposition)
fi
if [[ "${DH116_FORCE_DIRECTION_REPOSITION:-0}" == 1 ]]; then
  reposition_flag=(--force-direction-reposition)
fi
# Retain nf200 while diagnosing contact/control faults. Both nf200 and nf400
# have recorded follower overshoots; neither guarantees the measured speed cap.
exec bash scripts/python.sh scripts/train_dh116_tabletop_gpu_ppo.py \
  --pose-summary configs/dh116_training/pose_reference.json \
  --adaptive-grasp --force-feedback --full-hand-contact \
  --joint-envelope configs/dh116_joint_envelopes/free_grasp_geometry_v1.json \
  --grasp-initialization "$initialization" --require-initial-below-axis \
  --load-profile configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json \
  --object-kind solid_cylinder --object-material extreme_grip --hand-material grip_compliant \
  --num-envs 4096 --rollout-steps 24 --physics-hz 480 --mimic-frequency 200 \
  --position-iterations 64 --velocity-iterations 16 --reset-height-jitter-m 0 --cpu-threads "$DH116_NUM_THREADS" \
  --headless --device cuda:0 --direct-training --tensor-pipeline --training-episode-resets --pressure-relief \
  --grasp-controller "$controller" --palm-transport driven_carriage \
  "${reposition_flag[@]}" "${palm_axis_flag[@]}" "${acquisition_flag[@]}" "${lift_hold_flag[@]}" "${stable_lift_flag[@]}" "${force_region_flag[@]}" "${phase_force_flag[@]}" "${free_grasp_flag[@]}" "${iteration_flag[@]}" \
  --passive-velocity-limit-rad-s 2.97 --contact-reward-window close_start \
  --grasp-readiness measured_load --skip-hold --close-downward-penalty --grasp-shaping --vertical-force-reward --vertical-force-reward-mode continuous --force-reward-start downward_after_close \
  --episodes-per-env 1024 --output "$output" "${kit_flags[@]}" "${continuation[@]}" "$@"
