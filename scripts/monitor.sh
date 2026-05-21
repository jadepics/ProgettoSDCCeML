#!/usr/bin/env bash
set -u

ROLE="${1:-node}"
HOST="$(hostname)"
TS="$(date +"%Y%m%d_%H%M%S")"

OUT_DIR="/mnt/efs/gp_artifacts/diagnostics"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/${ROLE}_${HOST}_${TS}.log"

echo "Logging to: $LOG_FILE"

(
  while true; do
    echo
    echo "=================================================="
    date
    echo "=================================================="

    echo
    echo "[UPTIME]"
    uptime

    echo
    echo "[MEMORY]"
    free -h

    echo
    echo "[DISK]"
    df -h

    echo
    echo "[DOCKER STATS]"
    docker stats --no-stream 2>/dev/null || true

    echo
    echo "[DOCKER SYSTEM DF]"
    docker system df 2>/dev/null || true

    echo
    echo "[TOP CPU]"
    top -b -n 1 | head -20

    echo
    echo "[PROCESS COUNT]"
    ps aux | wc -l

    echo
    echo "[LOAD AVG]"
    cat /proc/loadavg

    echo
    echo "[SIZE /var/lib/docker]"
    sudo du -sh /var/lib/docker 2>/dev/null || true

    echo
    echo "[SIZE /var/log]"
    sudo du -sh /var/log 2>/dev/null || true

    echo
    echo "[SIZE gp_artifacts]"
    du -sh /mnt/efs/gp_artifacts 2>/dev/null || true

    sleep 5
  done
) >> "$LOG_FILE" 2>&1 &