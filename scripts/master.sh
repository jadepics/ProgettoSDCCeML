#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="gp-master"
CONTAINER_NAME="gp-master"
ENV_FILE="${PROJECT_ROOT}/.env.master"
ARTIFACT_MOUNT="/mnt/efs/gp_artifacts"

cd "$PROJECT_ROOT"

stop_container() {
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

build_image() {
  docker build -t "$IMAGE_NAME" -f Dockerfile.master .
}

run_container() {
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --network host \
    --env-file "$ENV_FILE" \
    -v "$ARTIFACT_MOUNT:$ARTIFACT_MOUNT" \
    "$IMAGE_NAME"
}

case "${1:-}" in
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
    git pull
    stop_container
    docker rmi "$IMAGE_NAME" >/dev/null 2>&1 || true
    build_image
    run_container
    ;;

  logs)
    docker logs -f "$CONTAINER_NAME"
    ;;

  shell)
    docker exec -it "$CONTAINER_NAME" bash
    ;;

  *)
    echo "Usage:"
    echo "  ./scripts/master.sh build"
    echo "  ./scripts/master.sh start"
    echo "  ./scripts/master.sh restart"
    echo "  ./scripts/master.sh rebuild"
    echo "  ./scripts/master.sh update"
    echo "  ./scripts/master.sh logs"
    echo "  ./scripts/master.sh shell"
    exit 1
    ;;
esac