#!/usr/bin/env bash
# Image-baked bootstrap. Tiny on purpose: anything that lives here can only be
# changed by rebuilding the image (~25 min CI). The real work lives in the
# repo's deploy/image/entrypoint.sh, which we exec after cloning.
#
# Why split it: iterating on the profiling loop should not require a Docker
# rebuild. Image bakes torch/marlin/nsys/ncu/gh/git (slow, stable). Logic
# (pip install flags, summary rendering, PR title format) lives in the repo.
#
# Inputs (env): REF (default main), REPO (default chokevin/swordfish), GH_TOKEN.
# Everything else is forwarded to the inner entrypoint untouched.

set -euo pipefail

REF="${REF:-main}"
REPO="${REPO:-chokevin/swordfish}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-autoresearch@swordfish.bot}"
GIT_USER_NAME="${GIT_USER_NAME:-swordfish-autoresearch}"

if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "FATAL: GH_TOKEN not set. Mount the gh-token secret." >&2
  exit 1
fi

export GH_TOKEN
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"
git config --global credential.helper '!f() { echo "username=x-access-token"; echo "password=${GH_TOKEN}"; }; f'

cd /work
echo "=== bootstrap: clone ${REPO}@${REF} ==="
git clone --depth 50 "https://github.com/${REPO}.git" swordfish
cd swordfish
git fetch --depth 50 origin "${REF}"
git checkout "${REF}"
echo "bootstrap source SHA: $(git rev-parse --short=7 HEAD)"

INNER="deploy/image/run.sh"
if [[ ! -x "${INNER}" ]]; then
  chmod +x "${INNER}" 2>/dev/null || {
    echo "FATAL: ${INNER} missing or not executable in cloned repo." >&2
    exit 1
  }
fi

echo "=== exec inner entrypoint: ${INNER} ==="
exec "./${INNER}"
