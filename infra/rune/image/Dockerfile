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
# What this image adds on top of the base:
#   * liger-kernel — the Triton-fused kernel library that is the W1 first
#     upstream touchpoint. Baked in so cold-start does not pay pip install
#     for every job.
#   * uv — the project manager swordfish uses for `uv run`.
#   * gh — used by future Friday-publish flows that file a Discussion or PR
#     to linkedin/Liger-Kernel from inside the runner.
#   * git, jq, ca-certificates — the usual cluster-job toolbox.
#
# What it does NOT add:
#   * The swordfish source tree itself. By convention the swordfish working
#     copy lives on the training-nfs PVC at /data-nfs/swordfish/src/current
#     so live edits do not need a rebuild. The in-pod entrypoint
#     infra/rune/scripts/swordfish-bench.sh `cd`s there and runs
#     `python -m swordfish.runner`.

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

# Sanity surface for `kubectl exec ... -- env`-style smoke tests.
RUN nvcc --version \
 && nsys --version \
 && ncu --version \
 && gh --version \
 && uv --version

# Default working dir mirrors the airun convention; the in-pod entrypoint cd's
# here at startup. The actual source is mounted from training-nfs at runtime.
WORKDIR /data-nfs/swordfish/src/current

# No ENTRYPOINT — rune submit + airun render-airun-gemm both pass an explicit
# command (bash infra/rune/scripts/swordfish-bench.sh ...). Leaving the image
# entrypoint open keeps the image reusable for `rune shell`-style interactive
# work too.
CMD ["bash"]
