#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-master"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.master"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

NODES=("master1" "master2" "master3")

declare -A MASTER_PORTS=(
  ["master1"]="50051"
  ["master2"]="50052"
  ["master3"]="50053"
)

declare -A RAFT_PORTS=(
  ["master1"]="50151"
  ["master2"]="50152"
  ["master3"]="50153"
)

ACTION="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

NODE="master1"
EXTRA_ENV_VARS=()

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./master.sh build"
  echo ""
  echo "Single master node:"
  echo "  ./master.sh start [master1|master2|master3] [KEY=VALUE ...]"
  echo "  ./master.sh restart [master1|master2|master3] [KEY=VALUE ...]"
  echo "  ./master.sh rebuild [master1|master2|master3] [KEY=VALUE ...]"
  echo "  ./master.sh update [master1|master2|master3] [KEY=VALUE ...]"
  echo "  ./master.sh stop [master1|master2|master3]"
  echo "  ./master.sh logs [master1|master2|master3]"
  echo "  ./master.sh shell [master1|master2|master3]"
  echo ""
  echo "Cluster:"
  echo "  ./master.sh start-all [KEY=VALUE ...]"
  echo "  ./master.sh restart-all [KEY=VALUE ...]"
  echo "  ./master.sh rebuild-all [KEY=VALUE ...]"
  echo "  ./master.sh update-all [KEY=VALUE ...]"
  echo "  ./master.sh stop-all"
  echo "  ./master.sh ps"
  echo "  ./master.sh reset-raft"
  echo ""
  echo "Example:"
  echo "  ./master.sh rebuild-all MASTER_CLUSTER_HOST=172.31.37.47"
  exit 1
}

is_valid_node() {
  local node="$1"
  for candidate in "${NODES[@]}"; do
    if [[ "$candidate" == "$node" ]]; then
      return 0
    fi
  done
  return 1
}

validate_node() {
  local node="$1"
  if ! is_valid_node "$node"; then
    echo "[ERROR] invalid node: $node"
    echo "Allowed nodes: ${NODES[*]}"
    exit 1
  fi
}

parse_node_and_env() {
  NODE="master1"
  EXTRA_ENV_VARS=()

  if [[ $# -gt 0 && "$1" != *=* ]]; then
    NODE="$1"
    shift
  fi

  validate_node "$NODE"
  EXTRA_ENV_VARS=("$@")
}

parse_env_only() {
  EXTRA_ENV_VARS=("$@")
}

container_name() {
  local node="$1"
  echo "gp-master-${node}"
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

get_config_value() {
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

detect_private_ip() {
  hostname -I | awk '{print $1}'
}

cluster_host() {
  local value=""

  if value="$(get_extra_env_value "MASTER_CLUSTER_HOST" 2>/dev/null)"; then
    echo "$value"
    return
  fi

  if [[ -n "${MASTER_CLUSTER_HOST:-}" ]]; then
    echo "$MASTER_CLUSTER_HOST"
    return
  fi

  if value="$(get_extra_env_value "MASTER_ADVERTISE_HOST" 2>/dev/null)"; then
    echo "$value"
    return
  fi

  if [[ -n "${MASTER_ADVERTISE_HOST:-}" ]]; then
    echo "$MASTER_ADVERTISE_HOST"
    return
  fi

  detect_private_ip
}

master_deployment_mode() {
  get_config_value "MASTER_DEPLOYMENT_MODE" "single-host"
}

master_grpc_port() {
  local node="$1"
  local mode
  mode="$(master_deployment_mode)"

echo "${MASTER_PORTS[$node]}"
}

master_raft_port() {
  local node="$1"
  local mode
  mode="$(master_deployment_mode)"

  if [[ "$mode" == "multi-host" ]]; then
    echo "50151"
    return
  fi

  echo "${RAFT_PORTS[$node]}"
}

master_node_private_ip() {
  local node="$1"

  case "$node" in
    master1)
      get_config_value "MASTER1_PRIVATE_IP" "172.31.37.47"
      ;;
    master2)
      get_config_value "MASTER2_PRIVATE_IP" "172.31.33.26"
      ;;
    master3)
      get_config_value "MASTER3_PRIVATE_IP" "172.31.35.187"
      ;;
    *)
      echo "[ERROR] unknown master node: $node" >&2
      exit 1
      ;;
  esac
}

ensure_single_host_cluster_command_allowed() {
  local command_name="$1"
  local mode
  mode="$(master_deployment_mode)"

  if [[ "$mode" == "multi-host" ]]; then
    echo "[ERROR] '$command_name' is not allowed when MASTER_DEPLOYMENT_MODE=multi-host"
    echo ""
    echo "In multi-host mode, each EC2 instance must run only its own master node:"
    echo "  master1 EC2 -> ./master.sh rebuild master1"
    echo "  master2 EC2 -> ./master.sh rebuild master2"
    echo "  master3 EC2 -> ./master.sh rebuild master3"
    echo ""
    echo "Do not use start-all/restart-all/rebuild-all/update-all in multi-host mode."
    exit 1
  fi
}

default_raft_peers() {
  local mode
  mode="$(master_deployment_mode)"

  if [[ "$mode" == "multi-host" ]]; then
    echo "master1:$(master_node_private_ip master1):50151,master2:$(master_node_private_ip master2):50151,master3:$(master_node_private_ip master3):50151"
    return
  fi

  local host
  host="$(cluster_host)"

  echo "master1:${host}:50151,master2:${host}:50152,master3:${host}:50153"
}

generate_env() {
  local node="$1"
  local runtime_env_file="${RUNTIME_ENV_DIR}/${node}.runtime.env"

  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    echo "[ERROR] .env.master not found at: $ENV_TEMPLATE"
    exit 1
  fi

  cp "$ENV_TEMPLATE" "$runtime_env_file"

  set_env_value "$runtime_env_file" "CONSENSUS_BACKEND" "raft"
  set_env_value "$runtime_env_file" "MASTER_NODE_ID" "$node"

  set_env_value "$runtime_env_file" "MASTER_HOST" "0.0.0.0"
  set_env_value "$runtime_env_file" "MASTER_PORT" "$(master_grpc_port "$node")"

  set_env_value "$runtime_env_file" "RAFT_HOST" "0.0.0.0"
  set_env_value "$runtime_env_file" "RAFT_PORT" "$(master_raft_port "$node")"
  set_env_value "$runtime_env_file" "RAFT_PEERS" "$(default_raft_peers)"
  set_env_value "$runtime_env_file" "RAFT_LOG_DIR" "${ARTIFACT_MOUNT}/raft/${node}"

  set_env_value "$runtime_env_file" "ARTIFACT_ROOT" "$ARTIFACT_MOUNT"

  set_env_value "$runtime_env_file" "RAFT_ELECTION_TIMEOUT_MS" "3000"
  set_env_value "$runtime_env_file" "RAFT_HEARTBEAT_INTERVAL_MS" "500"

  set_env_value "$runtime_env_file" "RECOVER_INCOMPLETE_JOBS_ON_STARTUP" "true"
  set_env_value "$runtime_env_file" "RECOVER_FAILED_JOBS_ON_STARTUP" "false"
  set_env_value "$runtime_env_file" "RECOVERY_STARTUP_DELAY_SECONDS" "10"
  set_env_value "$runtime_env_file" "RECOVERY_WAIT_WORKERS_TIMEOUT_SECONDS" "60"
  set_env_value "$runtime_env_file" "RECOVERY_WAIT_WORKERS_POLL_SECONDS" "2"
  set_env_value "$runtime_env_file" "RECOVERY_LEADER_POLL_SECONDS" "2"

  set_env_value "$runtime_env_file" "TRAINING_MAX_RECOVERY_ROUNDS" "60"
  set_env_value "$runtime_env_file" "TRAINING_MAX_IDLE_RECOVERY_ROUNDS" "5"
  set_env_value "$runtime_env_file" "TRAINING_RECOVERY_DEFERRED_SLEEP_SECONDS" "5"
  set_env_value "$runtime_env_file" "TRAINING_FORCE_RECLAIM_DEFERRED_AFTER_ROUNDS" "3"

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

stop_legacy_container() {
  docker stop gp-master >/dev/null 2>&1 || true
  docker rm gp-master >/dev/null 2>&1 || true
}

stop_container() {
  local node="$1"
  local cname
  cname="$(container_name "$node")"

  docker stop "$cname" >/dev/null 2>&1 || true
  docker rm "$cname" >/dev/null 2>&1 || true

  if [[ "$node" == "master1" ]]; then
    stop_legacy_container
  fi
}

stop_all_containers() {
  stop_legacy_container

  for node in "${NODES[@]}"; do
    stop_container "$node"
  done
}

build_image() {
  cd "$PROJECT_ROOT"
  docker build -t "$IMAGE_NAME" -f Dockerfile.master .
}

run_container() {
  local node="$1"
  local cname
  local env_file

  cname="$(container_name "$node")"
  env_file="$(generate_env "$node")"

  docker run -d \
    --name "$cname" \
    --restart unless-stopped \
    --network host \
    --env-file "$env_file" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"

  echo "[MASTER] started $node as container $cname"
  echo "[MASTER] deployment mode: $(master_deployment_mode)"
  echo "[MASTER] grpc port: $(master_grpc_port "$node")"
  echo "[MASTER] raft port: $(master_raft_port "$node")"
  echo "[MASTER] env file: $env_file"
}

start_one() {
  parse_node_and_env "$@"
  stop_container "$NODE"
  run_container "$NODE"
}

restart_one() {
  parse_node_and_env "$@"
  stop_container "$NODE"
  run_container "$NODE"
}

rebuild_one() {
  parse_node_and_env "$@"
  stop_container "$NODE"
  build_image
  run_container "$NODE"
}

update_one() {
  parse_node_and_env "$@"
  cd "$PROJECT_ROOT"
  git pull
  stop_container "$NODE"
  build_image
  run_container "$NODE"
}

start_all() {
  parse_env_only "$@"
  ensure_single_host_cluster_command_allowed "start-all"
  stop_all_containers

  for node in "${NODES[@]}"; do
    run_container "$node"
  done
}

restart_all() {
  parse_env_only "$@"
  ensure_single_host_cluster_command_allowed "restart-all"
  stop_all_containers

  for node in "${NODES[@]}"; do
    run_container "$node"
  done
}

rebuild_all() {
  parse_env_only "$@"
  ensure_single_host_cluster_command_allowed "rebuild-all"
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image

  for node in "${NODES[@]}"; do
    run_container "$node"
  done
}

update_all() {
  parse_env_only "$@"
  ensure_single_host_cluster_command_allowed "update-all"
  cd "$PROJECT_ROOT"
  git pull
  stop_all_containers
  docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
  build_image

  for node in "${NODES[@]}"; do
    run_container "$node"
  done
}

show_logs() {
  parse_node_and_env "$@"
  docker logs -f "$(container_name "$NODE")"
}

open_shell() {
  parse_node_and_env "$@"
  docker exec -it "$(container_name "$NODE")" bash
}

show_ps() {
  docker ps --filter "name=gp-master" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}"
}

reset_raft() {
  echo "[WARN] stopping all master containers before deleting Raft state..."
  stop_all_containers

  rm -rf "${ARTIFACT_MOUNT}/raft/master1"
  rm -rf "${ARTIFACT_MOUNT}/raft/master2"
  rm -rf "${ARTIFACT_MOUNT}/raft/master3"

  echo "[OK] Raft state deleted under ${ARTIFACT_MOUNT}/raft"
}

wait_leader() {
  local timeout_seconds="${WAIT_LEADER_TIMEOUT_SECONDS:-60}"
  local deadline=$((SECONDS + timeout_seconds))
  local mode
  local response

  mode="$(master_deployment_mode)"

  echo "[MASTER] waiting for Raft leader..."
  echo "[MASTER] deployment mode: ${mode}"

  while (( SECONDS < deadline )); do
    if [[ "$mode" == "multi-host" ]]; then
      for node in "${NODES[@]}"; do
        local host
        local port

        host="$(master_node_private_ip "$node")"
        port="$(master_raft_port "$node")"

        response="$(curl -s --max-time 1 -X POST "http://${host}:${port}/status" || true)"

        if echo "$response" | grep -q '"role": "LEADER"'; then
          echo "[MASTER] leader detected on ${node} ${host}:${port}"
          echo "$response"
          return 0
        fi
      done
    else
      local host
      host="$(cluster_host)"

      for port in 50151 50152 50153; do
        response="$(curl -s --max-time 1 -X POST "http://${host}:${port}/status" || true)"

        if echo "$response" | grep -q '"role": "LEADER"'; then
          echo "[MASTER] leader detected on Raft port ${port}"
          echo "$response"
          return 0
        fi
      done
    fi

    sleep 1
  done

  echo "[ERROR] no Raft leader detected within ${timeout_seconds}s"
  return 1
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
    parse_node_and_env "$@"
    stop_container "$NODE"
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

    wait-leader)
    parse_env_only "$@"
    wait_leader
    ;;

  stop-all)
    stop_all_containers
    ;;

  ps)
    show_ps
    ;;

  reset-raft)
    reset_raft
    ;;

  "")
    usage
    ;;

  *)
    usage
    ;;
esac