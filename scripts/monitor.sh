#!/bin/bash

ROLE=$1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HOST=$(hostname)

OUT_DIR="/mnt/efs/gp_artifacts/diagnostics"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/${ROLE}_${HOST}_${TIMESTAMP}.log"

echo "Monitoring -> $LOG_FILE"

(
while true
do
    echo ""
    echo "=================================================="
    date
    echo "=================================================="

    echo ""
    echo "----- UPTIME -----"
    uptime

    echo ""
    echo "----- MEMORY -----"
    free -m

    echo ""
    echo "----- DISK -----"
    df -h

    echo ""
    echo "----- TOP -----"
    top -b -n 1 | head -20

    echo ""
    echo "----- DOCKER STATS -----"
    docker stats --no-stream

    echo ""
    echo "----- PYTHON PROCESSES -----"
    ps aux | grep python

    echo ""
    echo "----- DOCKER PROCESSES -----"
    docker ps

    sleep 2

done
) >> "$LOG_FILE" 2>&1 &