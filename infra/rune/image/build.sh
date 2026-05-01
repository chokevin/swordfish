#!/usr/bin/env bash
# infra/rune/image/build.sh — local build helper for swordfish-bench.
#
# Two modes:
#   1. Stable build (default): builds and tags as `swordfish-bench:<sha>` plus
#      `swordfish-bench:dev-<sha>[-dirty]` for fast local iteration.
#   2. Push (PUSH=1): also pushes to GHCR. Requires `gh auth status` for the
#      `chokevin` account.
#
# CI is the canonical builder for `:latest` and `:main` tags; this script is
# the local-iteration loop. With layer caching, an iteration that only edits
# swordfish/ Python rebuilds only the final COPY+pip-install layer (~10s on
# arm64, ~5s on amd64).
#
# Usage:
#   ./build.sh                      build with auto-tag dev-<sha>[-dirty]
#   PUSH=1 ./build.sh               build + push (gh auth required)
#   TAG=foo ./build.sh              override the auto tag
#   PLATFORM=linux/amd64 ./build.sh force amd64 (slow on arm via QEMU)
#   LIGER_VERSION=0.5.11 ./build.sh override liger-kernel pin
#   CONTAINER_CMD=docker ./build.sh use docker instead of podman

set -euo pipefail

CONTAINER_CMD="${CONTAINER_CMD:-podman}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-chokevin/swordfish-bench}"
LIGER_VERSION="${LIGER_VERSION:-0.5.10}"
LIGER_REF="${LIGER_REF:-}"
PLATFORM="${PLATFORM:-}"
PUSH="${PUSH:-0}"

if ! command -v "${CONTAINER_CMD}" >/dev/null 2>&1; then
    echo "${CONTAINER_CMD} not on PATH; install it or set CONTAINER_CMD=docker" >&2
    exit 2
fi

# Walk up to repo root so `git describe` and the COPY context line up.
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd "$repo_root"

sha="$(git rev-parse --short HEAD)"
dirty=""
if ! git diff --quiet || ! git diff --cached --quiet; then
    dirty="-dirty"
fi
auto_tag="dev-${sha}${dirty}"
TAG="${TAG:-${auto_tag}}"

full_tag="${REGISTRY}/${IMAGE_NAME}:${TAG}"
sha_tag="${REGISTRY}/${IMAGE_NAME}:${sha}"
short_tag="${IMAGE_NAME}:${TAG}"

build_args=(
    --build-arg "LIGER_VERSION=${LIGER_VERSION}"
    --build-arg "LIGER_REF=${LIGER_REF}"
    --build-arg "SWORDFISH_SHA=${sha}${dirty}"
    --file infra/rune/image/Dockerfile
    --tag "${full_tag}"
    --tag "${sha_tag}"
    --tag "${short_tag}"
)
if [[ -n "${PLATFORM}" ]]; then
    build_args+=(--platform "${PLATFORM}")
fi

# podman uses --log-level=info; docker uses --progress=plain.
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
    python -c "import torch, triton, liger_kernel, swordfish.runner; \
print(f\"torch={torch.__version__} triton={triton.__version__}\"); \
print(\"liger + swordfish OK\")"
    ncu --version | head -1
    nsys --version | head -1
    echo "SWORDFISH_SHA=$SWORDFISH_SHA"
'

if [[ "${PUSH}" == "1" ]]; then
    if ! command -v gh >/dev/null 2>&1; then
        echo "gh CLI not on PATH; install gh or push manually with ${CONTAINER_CMD} push" >&2
        exit 2
    fi
    echo "== logging in to ${REGISTRY} via gh =="
    gh auth token | "${CONTAINER_CMD}" login "${REGISTRY}" -u "$(gh api user -q .login)" --password-stdin
    echo "== pushing ${full_tag} + ${sha_tag} =="
    "${CONTAINER_CMD}" push "${full_tag}"
    "${CONTAINER_CMD}" push "${sha_tag}"
    echo "${full_tag}"
fi
