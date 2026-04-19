#!/usr/bin/env bash
# Build + push the swordfish-autoresearch image.
#
# Usage:
#   REGISTRY=ghcr.io/chokevin TAG=$(git rev-parse --short HEAD) ./build.sh
#   REGISTRY=ghcr.io/chokevin TAG=latest ./build.sh push
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/chokevin}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
IMAGE="${REGISTRY}/swordfish-autoresearch:${TAG}"

cd "$(dirname "$0")"

echo "building ${IMAGE} ..."
docker build \
  --platform linux/amd64 \
  -t "${IMAGE}" \
  -t "${REGISTRY}/swordfish-autoresearch:latest" \
  .

if [[ "${1:-}" == "push" ]]; then
  echo "pushing ${IMAGE} ..."
  docker push "${IMAGE}"
  docker push "${REGISTRY}/swordfish-autoresearch:latest"
fi

echo "done: ${IMAGE}"
