#!/bin/bash

# =========================================================
# GP DISTRIBUTED ML - MONITOR SCRIPT
# =========================================================
#
# Uso:
# chmod +x monitor.sh
#
# MASTER:
# ./monitor.sh master
#
# WORKER:
# ./monitor.sh worker
#
# Output:
# /mnt/efs/gp_artifacts/diagnostics/
#
# =========================================================

ROLE=${1:-unknown}

HOSTNAME=$(hostname)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR="/mnt/efs/gp_artifacts/diagnostics"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="${OUTPUT_DIR}/${ROLE}_${HOSTNAME}_${TIMESTAMP}.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo "START MONITORING" | tee -a "$LOG_FILE"
echo "ROLE: $ROLE" | tee -a "$LOG_FILE"
echo "HOSTNAME: $HOSTNAME" | tee -a "$LOG_FILE"
echo "LOG_FILE: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

while true
do
    echo "" >> "$LOG_FILE"
    echo "==================================================" >> "$LOG_FILE"
    date >> "$LOG_FILE"
    echo "==================================================" >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[UPTIME]" >> "$LOG_FILE"
    uptime >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[MEMORY]" >> "$LOG_FILE"
    free -h >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[DISK]" >> "$LOG_FILE"
    df -h >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[TOP CPU]" >> "$LOG_FILE"
    top -b -n 1 | head -20 >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[DOCKER STATS]" >> "$LOG_FILE"
    docker stats --no-stream >> "$LOG_FILE" 2>&1

    echo "" >> "$LOG_FILE"
    echo "[DOCKER PS]" >> "$LOG_FILE"
    docker ps >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[NETWORK SOCKETS]" >> "$LOG_FILE"
    ss -tulpn >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[LAST KERNEL MESSAGES]" >> "$LOG_FILE"
    journalctl -k -n 30 >> "$LOG_FILE" 2>/dev/null
    
    echo "" >> "$LOG_FILE"
    echo "[PROCESS COUNT]" >> "$LOG_FILE"
    ps aux | wc -l >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"
    echo "[LOAD AVG]" >> "$LOG_FILE"
    cat /proc/loadavg >> "$LOG_FILE"

    sleep 10
done