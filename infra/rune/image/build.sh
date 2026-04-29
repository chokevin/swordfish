#!/usr/bin/env bash
# infra/rune/image/build.sh — local build helper for swordfish-bench.
#
# CI is the canonical builder (.github/workflows/build-swordfish-image.yml);
# this script is for local smoke tests on a host with podman or docker.
#
# Defaults to podman because (a) the swordfish dev box has the libkrun
# podman-machine running and (b) macOS Docker Desktop typically gives the
# VM only ~3.5GB RAM which is tight for the nvcr base image. Override with
# CONTAINER_CMD=docker if you want.
#
# The cluster runs amd64 GPUs, so CI builds amd64. Locally we build the
# host arch (arm64 on Apple silicon) for fast iteration; arm64 of the nvcr
# pytorch base exists. Override with PLATFORM=linux/amd64 to cross-build.
#
# Usage:
#   ./build.sh                              build :latest for host arch
#   TAG=v0.1.0 ./build.sh                   build :v0.1.0
#   PLATFORM=linux/amd64 ./build.sh         force amd64 (slow on arm)
#   PUSH=1 ./build.sh                       build + push to GHCR (gh auth required)
#   LIGER_VERSION=0.5.11 ./build.sh         override liger-kernel pin
#   LIGER_REF=main ./build.sh               build against the Liger main branch
#   CONTAINER_CMD=docker ./build.sh         use docker instead of podman

set -euo pipefail

CONTAINER_CMD="${CONTAINER_CMD:-podman}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-chokevin/swordfish-bench}"
TAG="${TAG:-latest}"
LIGER_VERSION="${LIGER_VERSION:-0.5.10}"
LIGER_REF="${LIGER_REF:-}"
PLATFORM="${PLATFORM:-}"
PUSH="${PUSH:-0}"

if ! command -v "${CONTAINER_CMD}" >/dev/null 2>&1; then
    echo "${CONTAINER_CMD} not on PATH; install it or set CONTAINER_CMD=docker" >&2
    exit 2
fi

cd "$(dirname "$0")"

full_tag="${REGISTRY}/${IMAGE_NAME}:${TAG}"
short_tag="${IMAGE_NAME}:${TAG}"

build_args=(
    --build-arg "LIGER_VERSION=${LIGER_VERSION}"
    --build-arg "LIGER_REF=${LIGER_REF}"
    --tag "${full_tag}"
    --tag "${short_tag}"
)
if [[ -n "${PLATFORM}" ]]; then
    build_args+=(--platform "${PLATFORM}")
fi

# podman uses --log-level=info instead of --progress=plain
log_flag=()
case "${CONTAINER_CMD}" in
    podman) log_flag=(--log-level=info) ;;
    docker) log_flag=(--progress=plain) ;;
esac

echo "== building ${full_tag} via ${CONTAINER_CMD} (platform=${PLATFORM:-host}) =="
"${CONTAINER_CMD}" build "${log_flag[@]}" "${build_args[@]}" .

echo "== smoke =="
"${CONTAINER_CMD}" run --rm "${short_tag}" bash -c '
    set -euo pipefail
    python -c "import torch, triton, liger_kernel; \
print(f\"torch={torch.__version__} triton={triton.__version__}\"); \
print(\"liger OK\")"
    nvcc --version | head -3
    ncu --version | head -1
    nsys --version | head -1
'

if [[ "${PUSH}" == "1" ]]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "gh CLI not on PATH; install gh or push manually with ${CONTAINER_CMD} push" >&2
        exit 2
    fi
    echo "== logging in to ${REGISTRY} via gh =="
    gh auth token | "${CONTAINER_CMD}" login "${REGISTRY}" -u "$(gh api user -q .login)" --password-stdin
    echo "== pushing ${full_tag} =="
    "${CONTAINER_CMD}" push "${full_tag}"
fi
