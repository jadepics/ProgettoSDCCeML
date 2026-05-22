#!/usr/bin/env bash
set -e

docker system prune -af
docker builder prune -af
docker image prune -af