#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-master"
CONTAINER_NAME="gp-master"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.master"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

ACTION="${1:-}"
EXTRA_ENV_VARS=()
if (( $# > 1 )); then
  EXTRA_ENV_VARS=("${@:2}")
fi

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./scripts/master.sh build"
  echo "  ./scripts/master.sh start [KEY=VALUE ...]"
  echo "  ./scripts/master.sh restart [KEY=VALUE ...]"
  echo "  ./scripts/master.sh rebuild [KEY=VALUE ...]"
  echo "  ./scripts/master.sh update [KEY=VALUE ...]"
  echo "  ./scripts/master.sh stop"
  echo "  ./scripts/master.sh logs"
  echo "  ./scripts/master.sh shell"
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

generate_env() {
  local runtime_env_file="${RUNTIME_ENV_DIR}/master.runtime.env"

  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    echo "[ERROR] .env.master not found at: $ENV_TEMPLATE"
    exit 1
  fi

  cp "$ENV_TEMPLATE" "$runtime_env_file"

  for kv in "${EXTRA_ENV_VARS[@]}"; do
    [[ "$kv" == *=* ]] || { echo "[ERROR] invalid env override: $kv"; exit 1; }
    key="${kv%%=*}"
    value="${kv#*=}"
    set_env_value "$runtime_env_file" "$key" "$value"
  done

  echo "$runtime_env_file"
}

stop_container() {
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

build_image() {
  cd "$PROJECT_ROOT"
  docker build -t "$IMAGE_NAME" -f Dockerfile.master .
}

run_container() {
  local env_file
  env_file="$(generate_env)"

  docker run -d \
    --name "$CONTAINER_NAME" \
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
    stop_container
    run_container
    ;;

  restart)
    stop_container
    run_container
    ;;

  rebuild)
    stop_container
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_container
    ;;

  update)
    cd "$PROJECT_ROOT"
    git pull
    stop_container
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_container
    ;;

  stop)
    stop_container
    ;;

  logs)
    docker logs -f "$CONTAINER_NAME"
    ;;

  shell)
    docker exec -it "$CONTAINER_NAME" bash
    ;;

  "")
    usage
    ;;

  *)
    usage
    ;;
esac