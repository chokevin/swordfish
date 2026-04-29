# Marlin-style INT4 x FP16 reproduction

This directory is the first correctness-first artifact for the INT4
weight-only baseline lane. It is not Marlin parity yet.

What exists:

- signed INT4 row-major packing, two values per byte,
- symmetric per-column, per-K-group quantization,
- a PyTorch `A @ dequant(W)` reference for correctness,
- a CUDA-only Triton kernel slot that dequantizes packed weights inside the
  tiled matmul loop,
- `python -m swordfish.runner bench-w4a16` for schema-valid latency and
  correctness JSON around the reference/Triton artifact.

What is still missing:

- a benchmark against real Marlin/Machete,
- layout packing that matches Marlin's production memory order,
- NCU evidence,
- the 90%-of-Marlin performance gate.

The point of this artifact is to make the data representation and correctness
contract explicit before chasing performance.

Local CPU smoke:

```bash
uv run python -m swordfish.runner bench-w4a16 \
  --backend reference \
  --m 8 --n 8 --k 16 \
  --group-size 8 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --out /tmp/swordfish-w4a16-smoke.json
```
