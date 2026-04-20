# swordfish-autoresearch image
#
# Runs bench/profile_marlin.sh on an A100, commits artifacts back to the
# swordfish repo on a per-run branch, and opens a draft PR for review.
#
# Base: NVIDIA's PyTorch container ships CUDA 12.4 + nsys + ncu + torch + triton,
# all built against each other and ABI-compatible. Don't reinvent this stack.
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Single layer for OS deps; gh CLI added below from its own apt source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        gnupg \
        jq \
    && rm -rf /var/lib/apt/lists/*

# gh CLI — we use this to open the draft PR back to chokevin/swordfish.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
      https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# Bake Marlin into the image at the pinned SHA so cold-start doesn't pay the
# build cost every run. Build flag matches docs/profiling/RUN_ME_ON_A100.md.
ARG MARLIN_SHA=1f25790bdd49fba53106164a24666dade68d7c90
ENV MARLIN_SHA=${MARLIN_SHA}
RUN TORCH_CUDA_ARCH_LIST="8.0" \
    pip install --no-cache-dir \
      "git+https://github.com/IST-DASLab/marlin.git@${MARLIN_SHA}" \
 && python -c "import marlin; print('marlin OK at', '${MARLIN_SHA}')"

WORKDIR /work
COPY bootstrap.sh /usr/local/bin/swordfish-autoresearch
RUN chmod +x /usr/local/bin/swordfish-autoresearch

# Sanity surface for `kubectl exec ... -- env`-style smoke tests.
RUN nvcc --version && nsys --version && ncu --version && gh --version

ENTRYPOINT ["/usr/local/bin/swordfish-autoresearch"]
