#!/bin/bash

ACTION=$1
WORKER_ID=$2
WORKER_PORT=$3

BASE_DIR="$HOME/gp"

IMAGE_NAME="gp-worker"
CONTAINER_NAME="$WORKER_ID"

ENV_TEMPLATE="$BASE_DIR/.env.worker"
RUNTIME_ENV_DIR="$BASE_DIR/runtime-env"

mkdir -p "$RUNTIME_ENV_DIR"

if [ -z "$ACTION" ]; then
  echo "Usage:"
  echo "./worker.sh build"
  echo "./worker.sh start worker1 50061"
  echo "./worker.sh restart worker1 50061"
  echo "./worker.sh stop worker1"
  exit 1
fi

build_image () {

  cd "$BASE_DIR" || exit 1

  docker build \
    -t $IMAGE_NAME \
    -f Dockerfile.worker \
    .
}

generate_env () {

  if [ ! -f "$ENV_TEMPLATE" ]; then
    echo ".env.worker not found in $ENV_TEMPLATE"
    exit 1
  fi

  ENV_FILE="$RUNTIME_ENV_DIR/$WORKER_ID.env"

  cp "$ENV_TEMPLATE" "$ENV_FILE"

  sed -i "s/^WORKER_ID=.*/WORKER_ID=$WORKER_ID/" "$ENV_FILE"

  sed -i "s/^WORKER_PORT=.*/WORKER_PORT=$WORKER_PORT/" "$ENV_FILE"

  echo "$ENV_FILE"
}

start_worker () {

  if [ -z "$WORKER_ID" ] || [ -z "$WORKER_PORT" ]; then
    echo "Missing worker_id or worker_port"
    exit 1
  fi

  ENV_FILE=$(generate_env)

  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --network host \
    --env-file "$ENV_FILE" \
    -v /mnt/efs/gp_artifacts:/mnt/efs/gp_artifacts \
    $IMAGE_NAME
}

stop_worker () {

  docker stop "$CONTAINER_NAME" || true

  docker rm "$CONTAINER_NAME" || true
}

restart_worker () {

  stop_worker

  start_worker
}

case $ACTION in

  build)
    build_image
    ;;

  start)
    start_worker
    ;;

  stop)
    stop_worker
    ;;

  restart)
    restart_worker
    ;;

  *)
    echo "Invalid action"
    ;;
esac