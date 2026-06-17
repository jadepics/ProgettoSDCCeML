#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-worker"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.worker"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

WORKERS=("worker1" "worker2" "worker3" "worker4" "worker5" "worker6")

#TODO togli questo dettaglio di worker già specifici
declare -A WORKER_PORTS=(
  ["worker1"]="50061"
  ["worker2"]="50062"
  ["worker3"]="50063"
  ["worker4"]="50064"
  ["worker5"]="50065"
  ["worker6"]="50066"
)

ACTION="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

WORKER_ID="worker1"
WORKER_PORT=""
EXTRA_ENV_VARS=()

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./scripts/worker.sh build"
  echo ""
  echo "Single worker:"
  echo "  ./scripts/worker.sh start <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh restart <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh rebuild <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh update <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh stop <worker_id>"
  echo "  ./scripts/worker.sh logs <worker_id>"
  echo "  ./scripts/worker.sh shell <worker_id>"
  echo ""
  echo "All workers:"
  echo "  ./scripts/worker.sh start-all [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh restart-all [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh rebuild-all [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh update-all [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh stop-all"
  echo "  ./scripts/worker.sh ps"
  echo ""
  echo "Example:"
  echo "  ./scripts/worker.sh rebuild-all MASTER_CLUSTER_HOST=172.31.37.47 WORKER_ADVERTISE_HOST=172.31.39.5"
  exit 1
}

is_known_worker() {
  local worker_id="$1"

  for candidate in "${WORKERS[@]}"; do
    if [[ "$candidate" == "$worker_id" ]]; then
      return 0
    fi
  done

  return 1
}

parse_worker_and_env() {
  [[ $# -ge 1 ]] || usage

  WORKER_ID="$1"
  shift

  if [[ $# -gt 0 && "$1" != *=* ]]; then
    WORKER_PORT="$1"
    shift
  else
    if is_known_worker "$WORKER_ID"; then
      WORKER_PORT="${WORKER_PORTS[$WORKER_ID]}"
    else
      echo "[ERROR] worker_port is required for unknown worker_id: $WORKER_ID"
      exit 1
    fi
  fi

  EXTRA_ENV_VARS=("$@")
}

parse_env_only() {
  EXTRA_ENV_VARS=("$@")
}

set_env_value() {
  local file="$1"
  local key="$2"
  local value="$3"

  if grep -qE "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    echo "${key}=${value}" >> "$file"
  fi
}

get_extra_env_value() {
  local key="$1"

  for kv in "${EXTRA_ENV_VARS[@]}"; do
    if [[ "${kv%%=*}" == "$key" ]]; then
      echo "${kv#*=}"
      return 0
    fi
  done

  return 1
}

get_template_env_value() {
  local key="$1"

  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    return 1
  fi

  grep -E "^${key}=" "$ENV_TEMPLATE" | tail -n 1 | cut -d "=" -f 2- || true
}

detect_private_ip() {
  hostname -I | awk '{print $1}'
}

master_cluster_host() {
  local value=""

  if value="$(get_extra_env_value "MASTER_CLUSTER_HOST" 2>/dev/null)"; then
    echo "$value"
    return
  fi

  if [[ -n "${MASTER_CLUSTER_HOST:-}" ]]; then
    echo "$MASTER_CLUSTER_HOST"
    return
  fi

  value="$(get_template_env_value "MASTER_HOST")"
  if [[ -n "$value" ]]; then
    echo "$value"
    return
  fi

  echo "172.31.37.47"
}

worker_advertise_host() {
  local value=""

  if value="$(get_extra_env_value "WORKER_ADVERTISE_HOST" 2>/dev/null)"; then
    echo "$value"
    return
  fi

  if [[ -n "${WORKER_ADVERTISE_HOST:-}" ]]; then
    echo "$WORKER_ADVERTISE_HOST"
    return
  fi

  value="$(get_template_env_value "WORKER_ADVERTISE_HOST")"
  if [[ -n "$value" ]]; then
    echo "$value"
    return
  fi

  detect_private_ip
}

master_seeds() {
  local host
  host="$(master_cluster_host)"

  echo "${host}:50051,${host}:50052,${host}:50053"
}

container_name_from_worker_id() {
  local worker_id="$1"
  echo "${worker_id}"
}

generate_env() {
  local worker_id="$1"
  local worker_port="$2"
  local runtime_env_file="${RUNTIME_ENV_DIR}/${worker_id}.runtime.env"

  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    echo "[ERROR] .env.worker not found at: $ENV_TEMPLATE"
    exit 1
  fi

  cp "$ENV_TEMPLATE" "$runtime_env_file"

  local master_host
  local advertise_host
  local seeds

  master_host="$(master_cluster_host)"
  advertise_host="$(worker_advertise_host)"
  seeds="$(master_seeds)"

  set_env_value "$runtime_env_file" "WORKER_ID" "$worker_id"
  set_env_value "$runtime_env_file" "WORKER_BIND_HOST" "0.0.0.0"
  set_env_value "$runtime_env_file" "WORKER_PORT" "$worker_port"
  set_env_value "$runtime_env_file" "WORKER_ADVERTISE_HOST" "$advertise_host"

  set_env_value "$runtime_env_file" "MASTER_HOST" "$master_host"
  set_env_value "$runtime_env_file" "MASTER_PORT" "50051"

  set_env_value "$runtime_env_file" "MASTER_LEADER_HOST" "$master_host"
  set_env_value "$runtime_env_file" "MASTER_LEADER_PORT" "50051"

  set_env_value "$runtime_env_file" "MASTER_SEEDS" "$seeds"

  set_env_value "$runtime_env_file" "SHARED_STORAGE_ROOT" "$ARTIFACT_MOUNT"
  set_env_value "$runtime_env_file" "MAX_CONCURRENT_TASKS" "1"

  for kv in "${EXTRA_ENV_VARS[@]}"; do
    [[ "$kv" == *=* ]] || {
      echo "[ERROR] invalid env override: $kv"
      exit 1
    }

    key="${kv%%=*}"
    value="${kv#*=}"

    if [[ "$key" == "MASTER_CLUSTER_HOST" ]]; then
      continue
    fi

    set_env_value "$runtime_env_file" "$key" "$value"
  done

  echo "$runtime_env_file"
}

stop_container() {
  local container_name="$1"
  docker stop "$container_name" >/dev/null 2>&1 || true
  docker rm "$container_name" >/dev/null 2>&1 || true
}

stop_all_containers() {
  for worker_id in "${WORKERS[@]}"; do
    stop_container "$(container_name_from_worker_id "$worker_id")"
  done
}

build_image() {
  cd "$PROJECT_ROOT"
  docker build -t "$IMAGE_NAME" -f Dockerfile.worker .
}

run_worker() {
  local worker_id="$1"
  local worker_port="$2"
  local container_name
  local env_file

  container_name="$(container_name_from_worker_id "$worker_id")"
  env_file="$(generate_env "$worker_id" "$worker_port")"

  docker run -d \
    --name "$container_name" \
    --restart unless-stopped \
    --network host \
    --env-file "$env_file" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"

  echo "[WORKER] started $worker_id as container $container_name"
  echo "[WORKER] port: $worker_port"
  echo "[WORKER] env file: $env_file"
}

start_one() {
  parse_worker_and_env "$@"
  stop_container "$(container_name_from_worker_id "$WORKER_ID")"
  run_worker "$WORKER_ID" "$WORKER_PORT"
}

restart_one() {
  parse_worker_and_env "$@"
  stop_container "$(container_name_from_worker_id "$WORKER_ID")"
  run_worker "$WORKER_ID" "$WORKER_PORT"
}

rebuild_one() {
  parse_worker_and_env "$@"
  stop_container "$(container_name_from_worker_id "$WORKER_ID")"
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image
  run_worker "$WORKER_ID" "$WORKER_PORT"
}

update_one() {
  parse_worker_and_env "$@"
  cd "$PROJECT_ROOT"
  git pull
  stop_container "$(container_name_from_worker_id "$WORKER_ID")"
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image
  run_worker "$WORKER_ID" "$WORKER_PORT"
}

start_all() {
  parse_env_only "$@"
  wait_master_leader
  stop_all_containers

  for worker_id in "${WORKERS[@]}"; do
    run_worker "$worker_id" "${WORKER_PORTS[$worker_id]}"
  done
}

restart_all() {
  parse_env_only "$@"
  wait_master_leader
  stop_all_containers

  for worker_id in "${WORKERS[@]}"; do
    run_worker "$worker_id" "${WORKER_PORTS[$worker_id]}"
  done
}
wait_master_leader() {
  local host
  host="$(master_cluster_host)"

  local timeout_seconds="${WAIT_LEADER_TIMEOUT_SECONDS:-60}"
  local deadline=$((SECONDS + timeout_seconds))

  echo "[WORKER] waiting for Raft leader on master cluster ${host}..."

  while (( SECONDS < deadline )); do
    for port in 50151 50152 50153; do
      response="$(curl -s --max-time 1 "http://${host}:${port}/status" || true)"

      if echo "$response" | grep -q '"role": "LEADER"'; then
        echo "[WORKER] leader detected on Raft port ${port}"
        echo "$response"
        return 0
      fi
    done

    sleep 1
  done

  echo "[WARN] no Raft leader detected within ${timeout_seconds}s; starting workers anyway"
  return 0
}

rebuild_all() {
  parse_env_only "$@"
  wait_master_leader
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image

  for worker_id in "${WORKERS[@]}"; do
    run_worker "$worker_id" "${WORKER_PORTS[$worker_id]}"
  done
}

update_all() {
  parse_env_only "$@"
  wait_master_leader
  cd "$PROJECT_ROOT"
  git pull
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image

  for worker_id in "${WORKERS[@]}"; do
    run_worker "$worker_id" "${WORKER_PORTS[$worker_id]}"
  done
}

show_logs() {
  parse_worker_and_env "$@"
  docker logs -f "$(container_name_from_worker_id "$WORKER_ID")"
}

open_shell() {
  parse_worker_and_env "$@"
  docker exec -it "$(container_name_from_worker_id "$WORKER_ID")" bash
}

show_ps() {
  docker ps --filter "name=worker" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}"
}

case "$ACTION" in
  build)
    build_image
    ;;

  start)
    start_one "$@"
    ;;

  restart)
    restart_one "$@"
    ;;

  rebuild)
    rebuild_one "$@"
    ;;

  update)
    update_one "$@"
    ;;

  stop)
    [[ $# -ge 1 ]] || usage
    stop_container "$(container_name_from_worker_id "$1")"
    ;;

  logs)
    show_logs "$@"
    ;;

  shell)
    open_shell "$@"
    ;;

  start-all)
    start_all "$@"
    ;;

  restart-all)
    restart_all "$@"
    ;;

  rebuild-all)
    rebuild_all "$@"
    ;;

  update-all)
    update_all "$@"
    ;;

  stop-all)
    stop_all_containers
    ;;

  ps)
    show_ps
    ;;

  "")
    usage
    ;;

  *)
    usage
    ;;
esac