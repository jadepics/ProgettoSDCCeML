#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-worker"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.worker"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

ACTION="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

WORKER_ID=""
WORKER_PORT=""
EXTRA_ENV_VARS=()

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./worker.sh build"
  echo ""
  echo "Single worker:"
  echo "  ./worker.sh start <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./worker.sh restart <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./worker.sh rebuild <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./worker.sh update <worker_id> [worker_port] [KEY=VALUE ...]"
  echo "  ./worker.sh stop <worker_id>"
  echo "  ./worker.sh logs <worker_id>"
  echo "  ./worker.sh shell <worker_id>"
  echo ""
  echo "All workers on this instance:"
  echo "  ./worker.sh start-all [KEY=VALUE ...]"
  echo "  ./worker.sh restart-all [KEY=VALUE ...]"
  echo "  ./worker.sh rebuild-all [KEY=VALUE ...]"
  echo "  ./worker.sh update-all [KEY=VALUE ...]"
  echo "  ./worker.sh stop-all"
  echo "  ./worker.sh ps"
  echo ""
  echo "Dynamic worker configuration:"
  echo "  WORKER_COUNT=4"
  echo "  WORKER_ID_PREFIX=worker"
  echo "  WORKER_INDEX_START=1"
  echo "  WORKER_BASE_PORT=50061"
  echo ""
  echo "Examples:"
  echo "  ./worker.sh rebuild-all WORKER_COUNT=4 MASTER_CLUSTER_HOST=172.31.37.47"
  echo "  ./worker.sh rebuild-all WORKER_COUNT=4 WORKER_ID_PREFIX=worker WORKER_INDEX_START=5 MASTER_CLUSTER_HOST=172.31.37.47"
  echo "  ./worker.sh start worker7 50067 MASTER_CLUSTER_HOST=172.31.37.47"
  exit 1
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

get_runtime_config_value() {
  local key="$1"
  local default="$2"
  local value=""

  if value="$(get_extra_env_value "$key" 2>/dev/null)"; then
    echo "$value"
    return
  fi

  if [[ -n "${!key:-}" ]]; then
    echo "${!key}"
    return
  fi

  value="$(get_template_env_value "$key")"
  if [[ -n "$value" ]]; then
    echo "$value"
    return
  fi

  echo "$default"
}

sanitize_for_worker_id() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

default_worker_id_prefix() {
  local raw
  raw="$(hostname -s 2>/dev/null || hostname)"
  raw="$(sanitize_for_worker_id "$raw")"
  echo "${raw}-worker"
}

worker_id_prefix() {
  get_runtime_config_value "WORKER_ID_PREFIX" "$(default_worker_id_prefix)"
}

worker_count() {
  get_runtime_config_value "WORKER_COUNT" "1"
}

worker_index_start() {
  get_runtime_config_value "WORKER_INDEX_START" "1"
}

worker_base_port() {
  get_runtime_config_value "WORKER_BASE_PORT" "50061"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${name} must be a positive integer, got: ${value}"
    exit 1
  fi

  if [[ "$value" -lt 1 ]]; then
    echo "[ERROR] ${name} must be >= 1, got: ${value}"
    exit 1
  fi
}

is_control_env_key() {
  case "$1" in
    MASTER_CLUSTER_HOST|MASTER_DEPLOYMENT_MODE|MASTER1_PRIVATE_IP|MASTER2_PRIVATE_IP|MASTER3_PRIVATE_IP|MASTER1_PORT|MASTER2_PORT|MASTER3_PORT|MASTER_SEEDS|WORKER_COUNT|WORKER_ID_PREFIX|WORKER_INDEX_START|WORKER_BASE_PORT|WAIT_LEADER_TIMEOUT_SECONDS)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

detect_private_ip() {
  hostname -I | awk '{print $1}'
}

master_deployment_mode() {
  get_runtime_config_value "MASTER_DEPLOYMENT_MODE" "single-host"
}

master_node_port() {
  local node="$1"

  case "$node" in
    master1)
      get_runtime_config_value "MASTER1_PORT" "50051"
      ;;
    master2)
      get_runtime_config_value "MASTER2_PORT" "50052"
      ;;
    master3)
      get_runtime_config_value "MASTER3_PORT" "50053"
      ;;
    *)
      echo "[ERROR] unknown master node: $node" >&2
      exit 1
      ;;
  esac
}

master_node_private_ip() {
  local node="$1"

  case "$node" in
    master1)
      get_runtime_config_value "MASTER1_PRIVATE_IP" "127.0.0.1"
      ;;
    master2)
      get_runtime_config_value "MASTER2_PRIVATE_IP" "127.0.0.1"
      ;;
    master3)
      get_runtime_config_value "MASTER3_PRIVATE_IP" "127.0.0.1"
      ;;
    *)
      echo "[ERROR] unknown master node: $node" >&2
      exit 1
      ;;
  esac
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

  master_node_private_ip "master1"
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

  detect_private_ip
}

master_seeds() {
  local override
  override="$(get_runtime_config_value "MASTER_SEEDS" "")"

  if [[ -n "$override" ]]; then
    echo "$override"
    return
  fi

  local mode
  mode="$(master_deployment_mode)"

  local master1_ip
  local master2_ip
  local master3_ip

  local master1_port
  local master2_port
  local master3_port

  master1_ip="$(master_node_private_ip master1)"
  master2_ip="$(master_node_private_ip master2)"
  master3_ip="$(master_node_private_ip master3)"

  master1_port="$(master_node_port master1)"
  master2_port="$(master_node_port master2)"
  master3_port="$(master_node_port master3)"

  if [[ "$mode" == "single-host" ]]; then
    echo "${master1_ip}:${master1_port},${master1_ip}:${master2_port},${master1_ip}:${master3_port}"
    return
  fi

  if [[ "$mode" == "multi-host" ]]; then
    echo "${master1_ip}:${master1_port},${master2_ip}:${master2_port},${master3_ip}:${master3_port}"
    return
  fi

  echo "[ERROR] invalid MASTER_DEPLOYMENT_MODE: $mode" >&2
  echo "[ERROR] allowed values: single-host, multi-host" >&2
  exit 1
}

container_name_from_worker_id() {
  local worker_id="$1"
  echo "${worker_id}"
}

infer_worker_port() {
  local worker_id="$1"
  local numeric_suffix
  local base
  local start

  numeric_suffix="${worker_id##*[!0-9]}"

  if [[ -z "$numeric_suffix" ]]; then
    echo "[ERROR] worker_port is required when worker_id has no numeric suffix: $worker_id" >&2
    exit 1
  fi

  base="$(worker_base_port)"
  start="$(worker_index_start)"

  validate_positive_integer "WORKER_BASE_PORT" "$base"
  validate_positive_integer "WORKER_INDEX_START" "$start"

  echo $((base + numeric_suffix - start))
}

parse_worker_and_env() {
  [[ $# -ge 1 ]] || usage

  WORKER_ID="$1"
  shift

  if [[ $# -gt 0 && "$1" != *=* ]]; then
    WORKER_PORT="$1"
    shift
  else
    WORKER_PORT=""
  fi

  EXTRA_ENV_VARS=("$@")

  if [[ -z "$WORKER_PORT" ]]; then
    WORKER_PORT="$(infer_worker_port "$WORKER_ID")"
  fi
}

for_configured_workers() {
  local callback="$1"
  local prefix
  local count
  local start
  local base
  local offset
  local index
  local worker_id
  local worker_port

  prefix="$(worker_id_prefix)"
  count="$(worker_count)"
  start="$(worker_index_start)"
  base="$(worker_base_port)"

  validate_positive_integer "WORKER_COUNT" "$count"
  validate_positive_integer "WORKER_INDEX_START" "$start"
  validate_positive_integer "WORKER_BASE_PORT" "$base"

  for ((offset = 0; offset < count; offset++)); do
    index=$((start + offset))
    worker_port=$((base + offset))
    worker_id="${prefix}${index}"

    "$callback" "$worker_id" "$worker_port"
  done
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
  set_env_value "$runtime_env_file" "MASTER_PORT" "$(master_node_port master1)"

  set_env_value "$runtime_env_file" "MASTER_LEADER_HOST" "$master_host"
  set_env_value "$runtime_env_file" "MASTER_LEADER_PORT" "$(master_node_port master1)"

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

    if is_control_env_key "$key"; then
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
  local ids
  local legacy_names

  ids="$(docker ps -aq --filter "label=gp.role=worker" || true)"

  if [[ -n "$ids" ]]; then
    docker rm -f $ids >/dev/null 2>&1 || true
  fi

  legacy_names="$(docker ps -a --format '{{.Names}}' | grep -E '^worker[0-9]+$' || true)"

  if [[ -n "$legacy_names" ]]; then
    docker rm -f $legacy_names >/dev/null 2>&1 || true
  fi
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
    --label "gp.role=worker" \
    --label "gp.worker_id=${worker_id}" \
    --restart unless-stopped \
    --network host \
    --env-file "$env_file" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"

  echo "[WORKER] started $worker_id as container $container_name"
  echo "[WORKER] port: $worker_port"
  echo "[WORKER] env file: $env_file"
}

wait_master_leader() {
  local timeout_seconds
  local deadline
  local response
  local mode

  timeout_seconds="${WAIT_LEADER_TIMEOUT_SECONDS:-60}"
  deadline=$((SECONDS + timeout_seconds))
  mode="$(master_deployment_mode)"

  echo "[WORKER] waiting for Raft leader..."
  echo "[WORKER] master deployment mode: ${mode}"

  while (( SECONDS < deadline )); do
    if [[ "$mode" == "multi-host" ]]; then
      for node in master1 master2 master3; do
        local host
        host="$(master_node_private_ip "$node")"

        response="$(curl -s --max-time 1 -X POST "http://${host}:50151/status" || true)"

        if echo "$response" | grep -q '"role": "LEADER"'; then
          echo "[WORKER] leader detected on ${node} ${host}:50151"
          echo "$response"
          return 0
        fi
      done
    else
      local host
      host="$(master_cluster_host)"

      for port in 50151 50152 50153; do
        response="$(curl -s --max-time 1 -X POST "http://${host}:${port}/status" || true)"

        if echo "$response" | grep -q '"role": "LEADER"'; then
          echo "[WORKER] leader detected on Raft port ${port}"
          echo "$response"
          return 0
        fi
      done
    fi

    sleep 1
  done

  echo "[WARN] no Raft leader detected within ${timeout_seconds}s; starting workers anyway"
  return 0
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
  for_configured_workers run_worker
}

restart_all() {
  parse_env_only "$@"
  wait_master_leader
  stop_all_containers
  for_configured_workers run_worker
}

rebuild_all() {
  parse_env_only "$@"
  wait_master_leader
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image
  for_configured_workers run_worker
}

update_all() {
  parse_env_only "$@"
  wait_master_leader
  cd "$PROJECT_ROOT"
  git pull
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image
  for_configured_workers run_worker
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
  docker ps \
    --filter "label=gp.role=worker" \
    --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}"
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