FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /workspace/swordfish

COPY pyproject.toml uv.lock README.md ./
COPY swordfish ./swordfish

RUN python -m pip install --no-cache-dir uv \
    && uv pip install --system --no-deps -e .

ENV PYTHONUNBUFFERED=1
