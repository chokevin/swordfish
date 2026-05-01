# swordfish-bench image
#
# Cross-arch (A100 / H100 NVL / H200) GPU runtime for the swordfish benchmark
# lab. Used by airun + rune to dispatch jobs against the voice-agent-flex AKS
# cluster.
#
# DESIGN
#
# Base: voiceagentcr.azurecr.io/airun/autoresearch-pytorch-ray:dev
#
# This is the in-region ACR image the airun-zero training stack uses. It is
# already prewarmed on every GPU node by the cluster-wide
# `baked-image-prewarm-v2` DaemonSet (see ai2:applications/airun-zero/deploy/
# umbrella/airun-core/values.yaml). Switching to it gets us:
#
#   1. ~10x faster cold-start pulls (in-region ACR vs ghcr.io egress).
#   2. ~250x layer-dedup win on prewarmed nodes — kubelet skips the ~10GB
#      base-layer download because containerd already has those digests
#      cached from the prewarm pull. Only our thin liger+swordfish diff
#      (~200MB) is fetched from ghcr at submit time.
#   3. Free Ray + transformers + trl + peft + datasets + accelerate +
#      bitsandbytes for any future training-style work without an extra pip
#      pass.
#
# Inventory of the base (verified 2026-04-30 against tag :dev):
#   torch         2.7.0+cu128   (matches our previous nvcr.io 25.03 base)
#   triton        3.3.0
#   ncu           2025.1.0.0    (for --profile-mode ncu)
#   ray           2.40.0
#   transformers  5.5.4         (liger 0.5.10 import verified compatible)
#   bitsandbytes, peft, trl, accelerate, datasets, evaluate, hf_transfer
#
# What we add via apt on top:
#   cuda-nsight-systems-12-8 (nsys 2024.6.2) — for --profile-mode nsys.
#   The base ships ncu but not nsys; this layer is ~250MB compressed.
#
# What this image adds on top of the base, in increasing rebuild frequency:
#   1. (rare)  liger-kernel — the Triton-fused kernel library; the W1 first
#              upstream touchpoint. Baked so cold-start does not pay pip
#              install for every job.
#   2. (often) the swordfish source tree itself, installed editable via
#              `pip install -e . --no-deps`. Researchers iterate on Python
#              files in experiments/ without rebuilding this layer; only when
#              the swordfish/ package itself changes does this layer rebuild.
#
# Build context is the swordfish repo root so swordfish/ + pyproject.toml +
# uv.lock are reachable. .dockerignore at the repo root keeps the context
# small.
#
# NOTE on the moving :dev tag: pinning to a digest is preferable for long
# sweeps (so a mid-sweep base bump doesn't change behavior). For now we
# track :dev and rely on CI to rebuild whenever the base shifts. If a sweep
# needs locked behavior, build with --build-arg BASE_IMAGE=<...@sha256:...>.

ARG BASE_IMAGE=voiceagentcr.azurecr.io/airun/autoresearch-pytorch-ray:dev
FROM ${BASE_IMAGE}

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Nsight Systems CLI for `--profile-mode nsys` (the autoresearch-pytorch-ray
# base ships Nsight Compute but not Nsight Systems). The cuda-nsight-systems
# meta-package is in the NVIDIA CUDA apt repo that the base already configures
# at /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list. update-alternatives
# wires `/usr/local/bin/nsys` automatically.
#
# Pinned to 12-8 to track the base's CUDA 12.8 toolchain — bump this in
# lockstep if the base ever moves to a different CUDA major.
RUN apt-get update \
 && apt-get install -y --no-install-recommends cuda-nsight-systems-12-8 \
 && rm -rf /var/lib/apt/lists/* \
 && nsys --version

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

# Sanity surface for `kubectl exec ... -- env`-style smoke tests. nvcc is in
# the autoresearch-pytorch-ray base via the CUDA toolkit; skip if missing in
# a future base bump.
RUN ncu --version | head -1 \
 && nsys --version | head -1 \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
 && python -c "from swordfish.runner.liger_perkernel import KERNEL_NAMES; print('liger-perkernel kernels:', KERNEL_NAMES)"

# Default working dir is the baked source tree. The in-pod entrypoint script
# does not need to cd anywhere; rune submit already inherits this.
WORKDIR /work/swordfish

# No ENTRYPOINT — rune submit + airun render-airun-gemm both pass an explicit
# command. Leaving the image entrypoint open keeps the image reusable for
# `rune shell`-style interactive work too.
CMD ["bash"]
