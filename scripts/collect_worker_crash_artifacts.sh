#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-}"
PHASE="${2:-snapshot}"

PROJECT_ROOT="/home/ec2-user/gp"
EFS_ROOT="/mnt/efs/gp_artifacts"

if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id> <phase>"
  echo "Example: $0 job_xxx before_crash"
  exit 1
fi

RUN_DIR="$PROJECT_ROOT/logs/fault_tolerance/worker_crash/$JOB_ID"
SNAPSHOT_DIR="$RUN_DIR/$PHASE"

mkdir -p "$SNAPSHOT_DIR"

echo "[collect] job_id=$JOB_ID"
echo "[collect] phase=$PHASE"
echo "[collect] output=$SNAPSHOT_DIR"

date --iso-8601=seconds > "$SNAPSHOT_DIR/timestamp.txt"

echo "[collect] Saving docker state..."
docker ps > "$SNAPSHOT_DIR/docker_ps.txt" 2>&1 || true
docker ps -a > "$SNAPSHOT_DIR/docker_ps_all.txt" 2>&1 || true

echo "[collect] Saving container logs..."
mkdir -p "$SNAPSHOT_DIR/container_logs"

for container in $(docker ps -a --format '{{.Names}}' | grep -E 'master|worker' || true); do
  safe_name="$(echo "$container" | tr '/' '_')"
  docker logs --timestamps --tail 1000 "$container" \
    > "$SNAPSHOT_DIR/container_logs/${safe_name}.log" 2>&1 || true
done

JOB_DIR="$EFS_ROOT/jobs/$JOB_ID"

echo "[collect] Saving EFS job artifacts..."
mkdir -p "$SNAPSHOT_DIR/efs_job"

if [ -d "$JOB_DIR" ]; then
  if [ -f "$JOB_DIR/job_record.json" ]; then
    cp "$JOB_DIR/job_record.json" "$SNAPSHOT_DIR/efs_job/job_record.json"
  fi

  if [ -f "$JOB_DIR/task_ledger.json" ]; then
    cp "$JOB_DIR/task_ledger.json" "$SNAPSHOT_DIR/efs_job/task_ledger.json"
  fi

  find "$JOB_DIR" -type f | sort > "$SNAPSHOT_DIR/efs_job/job_files.txt" || true

  find "$JOB_DIR" -path "*/trees/*.joblib" -type f | sort \
    > "$SNAPSHOT_DIR/efs_job/tree_files.txt" || true

  wc -l "$SNAPSHOT_DIR/efs_job/tree_files.txt" \
    > "$SNAPSHOT_DIR/efs_job/tree_count.txt" || true
else
  echo "Job directory not found: $JOB_DIR" > "$SNAPSHOT_DIR/efs_job/MISSING_JOB_DIR.txt"
fi

echo "[collect] Searching model manifest for this job..."
mkdir -p "$SNAPSHOT_DIR/model"

if [ -d "$EFS_ROOT/models" ]; then
  grep -R "\"job_id\": \"$JOB_ID\"" "$EFS_ROOT/models" \
    --include="manifest.json" -l \
    > "$SNAPSHOT_DIR/model/matching_manifests.txt" 2>/dev/null || true

  while read -r manifest_path; do
    if [ -n "$manifest_path" ] && [ -f "$manifest_path" ]; then
      model_dir_name="$(basename "$(dirname "$manifest_path")")"
      cp "$manifest_path" "$SNAPSHOT_DIR/model/${model_dir_name}_manifest.json" || true
    fi
  done < "$SNAPSHOT_DIR/model/matching_manifests.txt"
fi

echo "[collect] Writing summary..."
{
  echo "fault_tolerance_test=worker_crash"
  echo "job_id=$JOB_ID"
  echo "phase=$PHASE"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo
  echo "docker_running_containers:"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  echo
  echo "job_dir=$JOB_DIR"
  if [ -f "$SNAPSHOT_DIR/efs_job/job_record.json" ]; then
    echo "job_record_present=true"
  else
    echo "job_record_present=false"
  fi
  if [ -f "$SNAPSHOT_DIR/efs_job/task_ledger.json" ]; then
    echo "task_ledger_present=true"
  else
    echo "task_ledger_present=false"
  fi
  if [ -f "$SNAPSHOT_DIR/efs_job/tree_count.txt" ]; then
    echo "tree_count=$(cat "$SNAPSHOT_DIR/efs_job/tree_count.txt")"
  fi
} > "$SNAPSHOT_DIR/summary.txt"

echo "[collect] Done."
echo "[collect] Snapshot saved in: $SNAPSHOT_DIR"