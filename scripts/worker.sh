#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-worker"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.worker"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

ACTION="${1:-}"
WORKER_ID="${2:-}"
WORKER_PORT="${3:-}"
EXTRA_ENV_VARS=()
if (( $# > 3 )); then
  EXTRA_ENV_VARS=("${@:4}")
fi

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./scripts/worker.sh build"
  echo "  ./scripts/worker.sh start <worker_id> <worker_port> [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh restart <worker_id> <worker_port> [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh rebuild <worker_id> <worker_port> [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh update <worker_id> <worker_port> [KEY=VALUE ...]"
  echo "  ./scripts/worker.sh stop <worker_id>"
  echo "  ./scripts/worker.sh logs <worker_id>"
  echo "  ./scripts/worker.sh shell <worker_id>"
  exit 1
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

  set_env_value "$runtime_env_file" "WORKER_ID" "$worker_id"
  set_env_value "$runtime_env_file" "WORKER_PORT" "$worker_port"

  for kv in "${EXTRA_ENV_VARS[@]}"; do
    [[ "$kv" == *=* ]] || { echo "[ERROR] invalid env override: $kv"; exit 1; }
    key="${kv%%=*}"
    value="${kv#*=}"
    set_env_value "$runtime_env_file" "$key" "$value"
  done

  echo "$runtime_env_file"
}

stop_container() {
  local container_name="$1"
  docker stop "$container_name" >/dev/null 2>&1 || true
  docker rm "$container_name" >/dev/null 2>&1 || true
}

build_image() {
  cd "$PROJECT_ROOT"
  docker build -t "$IMAGE_NAME" -f Dockerfile.worker .
}

run_worker() {
  local worker_id="$1"
  local worker_port="$2"
  local container_name
  container_name="$(container_name_from_worker_id "$worker_id")"

  local env_file
  env_file="$(generate_env "$worker_id" "$worker_port")"

  docker run -d \
    --name "$container_name" \
    --restart unless-stopped \
    --network host \
    --env-file "$env_file" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"
}

case "$ACTION" in
  build)
    build_image
    ;;

  start)
    [[ -n "$WORKER_ID" && -n "$WORKER_PORT" ]] || usage
    stop_container "$(container_name_from_worker_id "$WORKER_ID")"
    run_worker "$WORKER_ID" "$WORKER_PORT"
    ;;

  restart)
    [[ -n "$WORKER_ID" && -n "$WORKER_PORT" ]] || usage
    stop_container "$(container_name_from_worker_id "$WORKER_ID")"
    run_worker "$WORKER_ID" "$WORKER_PORT"
    ;;

  rebuild)
    [[ -n "$WORKER_ID" && -n "$WORKER_PORT" ]] || usage
    stop_container "$(container_name_from_worker_id "$WORKER_ID")"
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_worker "$WORKER_ID" "$WORKER_PORT"
    ;;

  update)
    [[ -n "$WORKER_ID" && -n "$WORKER_PORT" ]] || usage
    cd "$PROJECT_ROOT"
    git pull
    stop_container "$(container_name_from_worker_id "$WORKER_ID")"
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_worker "$WORKER_ID" "$WORKER_PORT"
    ;;

  stop)
    [[ -n "$WORKER_ID" ]] || usage
    stop_container "$(container_name_from_worker_id "$WORKER_ID")"
    ;;

  logs)
    [[ -n "$WORKER_ID" ]] || usage
    docker logs -f "$(container_name_from_worker_id "$WORKER_ID")"
    ;;

  shell)
    [[ -n "$WORKER_ID" ]] || usage
    docker exec -it "$(container_name_from_worker_id "$WORKER_ID")" bash
    ;;

  "")
    usage
    ;;

  *)
    usage
    ;;
esac