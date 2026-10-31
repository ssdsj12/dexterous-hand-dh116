#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Select one existing run without changing its files or any training process.
if (( $# > 1 )); then printf 'Usage: %s [run-directory]\n' "$0" >&2; exit 2; fi
if (( $# == 1 )); then
  target="$(realpath -e -- "$1")"
  case "$target" in "$root"/runs/*) ;; *) printf 'Run must be inside %s/runs\n' "$root" >&2; exit 2 ;; esac
  if [[ ! -d "$target" ]]; then printf 'Run directory required\n' >&2; exit 2; fi
  if [[ -e "$root/runs/current" && ! -L "$root/runs/current" ]]; then
    printf 'Refusing to replace an existing real runs/current directory\n' >&2; exit 2
  fi
  ln -sfn -- "$target" "$root/runs/current"
fi
if [[ ! -d "$root/runs/current" ]]; then printf 'Select a current run first\n' >&2; exit 2; fi
# Use the dedicated supervisor socket; leave system supervisor and Isaac alone.
bash "$root/deploy/hand_services.sh" start
"/opt/conda/envs/hand/bin/supervisorctl" -c "$root/deploy/supervisord.conf" reread
"/opt/conda/envs/hand/bin/supervisorctl" -c "$root/deploy/supervisord.conf" update tensorboard
exec bash "$root/deploy/hand_services.sh" restart
