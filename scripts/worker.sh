#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-worker"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.worker"
RUNTIME_ENV_DIR="${PROJECT_ROOT}/runtime-env"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

mkdir -p "$RUNTIME_ENV_DIR"

usage() {
  echo "Usage:"
  echo "  ./scripts/worker.sh build"
  echo "  ./scripts/worker.sh start <worker_id> <worker_port>"
  echo "  ./scripts/worker.sh restart <worker_id> <worker_port>"
  echo "  ./scripts/worker.sh rebuild <worker_id> <worker_port>"
  echo "  ./scripts/worker.sh update <worker_id> <worker_port>"
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

make_runtime_env() {
  local worker_id="$1"
  local worker_port="$2"
  local runtime_env_file="${RUNTIME_ENV_DIR}/${worker_id}.env"

  cp "$ENV_TEMPLATE" "$runtime_env_file"
  set_env_value "$runtime_env_file" "WORKER_ID" "$worker_id"
  set_env_value "$runtime_env_file" "WORKER_PORT" "$worker_port"

  echo "$runtime_env_file"
}

container_name_from_worker_id() {
  local worker_id="$1"
  echo "gp-${worker_id}"
}

stop_container() {
  local container_name="$1"
  docker stop "$container_name" >/dev/null 2>&1 || true
  docker rm "$container_name" >/dev/null 2>&1 || true
}

build_image() {
  docker build -t "$IMAGE_NAME" -f Dockerfile.worker .
}

run_worker() {
  local worker_id="$1"
  local worker_port="$2"

  local container_name
  container_name="$(container_name_from_worker_id "$worker_id")"

  local runtime_env_file
  runtime_env_file="$(make_runtime_env "$worker_id" "$worker_port")"

  docker run -d \
    --name "$container_name" \
    --restart unless-stopped \
    --network host \
    --env-file "$runtime_env_file" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"
}

case "${1:-}" in
  build)
    build_image
    ;;

  start)
    [[ $# -eq 3 ]] || usage
    worker_id="$2"
    worker_port="$3"
    container_name="$(container_name_from_worker_id "$worker_id")"
    stop_container "$container_name"
    run_worker "$worker_id" "$worker_port"
    ;;

  restart)
    [[ $# -eq 3 ]] || usage
    worker_id="$2"
    worker_port="$3"
    container_name="$(container_name_from_worker_id "$worker_id")"
    stop_container "$container_name"
    run_worker "$worker_id" "$worker_port"
    ;;

  rebuild)
    [[ $# -eq 3 ]] || usage
    worker_id="$2"
    worker_port="$3"
    container_name="$(container_name_from_worker_id "$worker_id")"
    stop_container "$container_name"
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_worker "$worker_id" "$worker_port"
    ;;

  update)
    [[ $# -eq 3 ]] || usage
    worker_id="$2"
    worker_port="$3"
    container_name="$(container_name_from_worker_id "$worker_id")"
    git pull
    stop_container "$container_name"
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_worker "$worker_id" "$worker_port"
    ;;

  stop)
    [[ $# -eq 2 ]] || usage
    worker_id="$2"
    container_name="$(container_name_from_worker_id "$worker_id")"
    stop_container "$container_name"
    ;;

  logs)
    [[ $# -eq 2 ]] || usage
    worker_id="$2"
    docker logs -f "$(container_name_from_worker_id "$worker_id")"
    ;;

  shell)
    [[ $# -eq 2 ]] || usage
    worker_id="$2"
    docker exec -it "$(container_name_from_worker_id "$worker_id")" bash
    ;;

  *)
    usage
    ;;
esac