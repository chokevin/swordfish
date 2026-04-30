# swordfish-bench image
#
# Cross-arch (A100 / H100 NVL / H200) GPU runtime for the swordfish benchmark
# lab. Used by airun + rune to dispatch jobs against the voice-agent-flex AKS
# cluster.
#
# DESIGN
#
# Base: NVIDIA's PyTorch container ships torch 2.7+nv25.03, CUDA 12.8, Triton,
# Nsight Compute, and Nsight Systems built against each other. We do not
# re-pin torch on top because it would break the Triton/NCU ABI alignment.
# The Ray-flavored cluster images do not ship this stack in a known-good
# combination, so we build on the NVIDIA pytorch base directly.
#
# What this image adds on top of the base, in increasing rebuild frequency:
#   1. (rare)  liger-kernel — the Triton-fused kernel library; the W1 first
#              upstream touchpoint. Baked so cold-start does not pay pip
#              install for every job.
#   2. (rare)  uv, gh, git, jq, ca-certificates — research toolbox.
#   3. (often) the swordfish source tree itself, installed editable via
#              `pip install -e . --no-deps`. Researchers iterate on Python
#              files in experiments/ without rebuilding this layer; only when
#              the swordfish/ package itself changes does this layer rebuild.
#
# Build context is the swordfish repo root so swordfish/ + pyproject.toml +
# uv.lock are reachable. .dockerignore at the repo root keeps the context
# small.

FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# OS deps + gh CLI in one apt update window.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        jq \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
         | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
         https://cli.github.com/packages stable main" \
         > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# uv for swordfish dependency management.
ARG UV_VERSION=0.5.7
RUN curl -fsSL https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh

# liger-kernel: pinned via build arg so reproducible. Default tracks the
# stable PyPI release. Bump LIGER_VERSION (or override LIGER_REF to a git ref)
# to refresh.
ARG LIGER_VERSION=0.5.10
ARG LIGER_REF=
RUN if [ -n "${LIGER_REF}" ]; then \
        pip install --no-cache-dir "git+https://github.com/linkedin/Liger-Kernel.git@${LIGER_REF}" --no-deps \
            && pip install --no-cache-dir "liger-kernel==${LIGER_VERSION}" --no-deps; \
    else \
        pip install --no-cache-dir "liger-kernel==${LIGER_VERSION}"; \
    fi \
 && python -c "import liger_kernel; print('liger_kernel OK', liger_kernel.__version__ if hasattr(liger_kernel, '__version__') else 'no __version__ attr')" \
 && python -c "from liger_kernel.transformers.rms_norm import LigerRMSNorm; print('LigerRMSNorm OK')" \
 && python -c "from liger_kernel.transformers.swiglu import LigerSwiGLUMLP; print('LigerSwiGLUMLP OK')"

# Bake the swordfish source. We copy the package + project files only (not the
# whole repo — see .dockerignore at repo root) and install editable so import
# paths resolve cleanly. --no-deps because torch/triton/numpy are already in
# the base image and we do not want pip to substitute its own versions.
WORKDIR /work/swordfish
COPY pyproject.toml uv.lock README.md ./
COPY swordfish/ ./swordfish/
RUN pip install --no-cache-dir --no-deps -e . \
 && python -c "import swordfish.runner; print('swordfish.runner OK')"

# Provenance: the SHA the source was built from. Stamped into the image at
# build time and surfaced inside the pod so result JSONs can record it.
ARG SWORDFISH_SHA=local
ENV SWORDFISH_SHA=${SWORDFISH_SHA}

# Sanity surface for `kubectl exec ... -- env`-style smoke tests.
RUN nvcc --version \
 && nsys --version \
 && ncu --version \
 && gh --version \
 && uv --version \
 && python -c "from swordfish.runner.liger_perkernel import KERNEL_NAMES; print('liger-perkernel kernels:', KERNEL_NAMES)"

# Default working dir is the baked source tree. The in-pod entrypoint script
# does not need to cd anywhere; rune submit already inherits this.
WORKDIR /work/swordfish

# No ENTRYPOINT — rune submit + airun render-airun-gemm both pass an explicit
# command. Leaving the image entrypoint open keeps the image reusable for
# `rune shell`-style interactive work too.
CMD ["bash"]
