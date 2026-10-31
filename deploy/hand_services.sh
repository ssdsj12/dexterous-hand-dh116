#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="$root/deploy/supervisord.conf"
control=/opt/conda/envs/hand/bin/supervisorctl
daemon=/opt/conda/envs/hand/bin/supervisord
case "${1:-status}" in
  start)
    mkdir -p "$root/runs/services"
    if ! "$control" -c "$config" pid >/dev/null 2>&1; then
      "$daemon" -c "$config"
    fi
    for ((attempt=0; attempt<10; attempt++)); do
      if "$control" -c "$config" status; then exit 0; fi
      sleep 1
    done
    exit 1
    ;;
  status) exec "$control" -c "$config" status ;;
  restart) exec "$control" -c "$config" restart tensorboard ;;
  stop) exec "$control" -c "$config" shutdown ;;
  *) printf 'Usage: %s {start|status|restart|stop}\n' "$0" >&2; exit 2 ;;
esac
