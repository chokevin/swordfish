#!/usr/bin/env bash
# infra/rune/image/build-acr.sh — remote build via Azure ACR Tasks.
#
# Builds the swordfish-bench image on ACR's build agents, pushing the result
# to voiceagentcr.azurecr.io/airun/swordfish-bench. This is the canonical
# (and fastest) build path:
#
#   - Base image (autoresearch-pytorch-ray:dev) is local to ACR -> the layer
#     pull during build is intra-region and quick.
#   - The cluster pulls the result from the same in-region registry, with
#     base-layer dedup against the prewarmed daemonset.
#   - No ~12GB local docker pull/push round-trip.
#   - No GitHub Actions runner with 14GB of preinstalled tooling to fight.
#
# Tags: every build publishes :<short-sha> + :dev. For sweep pinning, use the
# :<sha> tag (the :dev tag is moving).
#
# Usage:
#   az acr login -n voiceagentcr        # one-time per laptop session
#   ./build-acr.sh                      # build + push :<sha> and :dev
#   ./build-acr.sh --no-push            # az acr build dry run (validates)
#
# Env overrides:
#   ACR_REGISTRY=voiceagentcr           # registry name (no .azurecr.io)
#   ACR_REPO=airun/swordfish-bench      # repo name within registry
#   TRANSFORMERS_VERSION=4.56.2         # Liger Llama patch-compatible pin
#   LIGER_VERSION=0.5.10                # bake a different liger pin
#   LIGER_REF=                          # or a git ref against linkedin/Liger-Kernel
#   BASE_IMAGE=...                      # pin a digest for sweep reproducibility
#   EXTRA_TAG=v0.2.0                    # publish an additional human tag
#
# Verify the publish:
#   az acr repository show-tags -n voiceagentcr --repository airun/swordfish-bench --orderby time_desc --top 5

set -euo pipefail

ACR_REGISTRY="${ACR_REGISTRY:-voiceagentcr}"
ACR_REPO="${ACR_REPO:-airun/swordfish-bench}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.56.2}"
LIGER_VERSION="${LIGER_VERSION:-0.5.10}"
LIGER_REF="${LIGER_REF:-}"
BASE_IMAGE="${BASE_IMAGE:-}"
EXTRA_TAG="${EXTRA_TAG:-}"

if ! command -v az >/dev/null 2>&1; then
    echo "az CLI not on PATH; install Azure CLI: https://aka.ms/azure-cli" >&2
    exit 2
fi

# Walk up to repo root so the build context lines up with the COPY paths.
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd "$repo_root"

sha="$(git rev-parse --short HEAD)"
dirty=""
if ! git diff --quiet || ! git diff --cached --quiet; then
    dirty="-dirty"
fi
sha_tag="${sha}${dirty}"

build_args=(
    -r "$ACR_REGISTRY"
    -t "${ACR_REPO}:dev"
    -t "${ACR_REPO}:${sha_tag}"
    --build-arg "TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}"
    --build-arg "LIGER_VERSION=${LIGER_VERSION}"
    --build-arg "SWORDFISH_SHA=${sha_tag}"
    -f infra/rune/image/Dockerfile
)

if [[ -n "$LIGER_REF" ]]; then
    build_args+=(--build-arg "LIGER_REF=${LIGER_REF}")
fi
if [[ -n "$BASE_IMAGE" ]]; then
    build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi
if [[ -n "$EXTRA_TAG" ]]; then
    build_args+=(-t "${ACR_REPO}:${EXTRA_TAG}")
fi
if [[ "${1:-}" == "--no-push" ]]; then
    build_args+=(--no-push)
fi

echo "== az acr build -> ${ACR_REGISTRY}.azurecr.io/${ACR_REPO}:{${sha_tag},dev${EXTRA_TAG:+,${EXTRA_TAG}}} =="
time az acr build "${build_args[@]}" .

echo
echo "== published =="
echo "  voiceagentcr.azurecr.io/${ACR_REPO}:${sha_tag}"
echo "  voiceagentcr.azurecr.io/${ACR_REPO}:dev"
[[ -n "$EXTRA_TAG" ]] && echo "  voiceagentcr.azurecr.io/${ACR_REPO}:${EXTRA_TAG}"
echo
echo "Cluster profile pack (swordfish-pack.yaml) tracks :dev. Pin :${sha_tag}"
echo "into a sweep config when reproducibility matters."
