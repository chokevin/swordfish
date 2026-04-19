# Target shapes — voice-agent decode

Every kernel optimization is shape-dependent. This doc lists the shapes swordfish prioritizes, in order.

## Why these shapes

Voice agents at production scale hit:
- Small batch sizes (1–16) because audio streaming is user-bound, not batch-bound
- Seq len 1 in decode (KV cache handles history)
- Hidden sizes of common OSS models (Llama-3 family, Qwen, Kimi)
- Group sizes set by the quantization recipe, typically 64 or 128

## Shape catalog

### Llama-3-8B (hidden=4096)

| shape name | M | N | K | group_size | priority |
|------------|---|---|---|------------|----------|
| 8b-b1 | 1 | 4096 | 4096 | 128 | P0 |
| 8b-b4 | 4 | 4096 | 4096 | 128 | P0 |
| 8b-b8 | 8 | 4096 | 4096 | 128 | P0 |
| 8b-b16 | 16 | 4096 | 4096 | 128 | P1 |
| 8b-b1-g64 | 1 | 4096 | 4096 | 64 | P1 |
| 8b-b4-g64 | 4 | 4096 | 4096 | 64 | P1 |

### Llama-3-70B (hidden=8192, with TP=2: 4096 per GPU)

| shape name | M | N | K | group_size | priority |
|------------|---|---|---|------------|----------|
| 70b-tp2-b1 | 1 | 8192 | 4096 | 128 | P0 |
| 70b-tp2-b4 | 4 | 8192 | 4096 | 128 | P0 |
| 70b-tp2-b8 | 8 | 8192 | 4096 | 128 | P0 |
| 70b-tp2-mlp | 4..16 | 14336 | 4096 | 128 | P1 |

### Qwen / hybrid models (hidden=3584 and 5120)

| shape name | M | N | K | group_size | priority |
|------------|---|---|---|------------|----------|
| qwen-3584-b4 | 4 | 3584 | 3584 | 128 | P1 |
| qwen-5120-b4 | 4 | 5120 | 5120 | 128 | P1 |

## What we're NOT optimizing for (yet)

- Very long context prefill shapes (M >> 16) — that's covered by FP16 FlashAttention + unquantized GEMM in most production configs
- Non-standard hidden sizes below 2048 — low priority
- SM75 (T4) or SM86 (A10, RTX 3090) — Ampere consumer; close but not identical to A100

## How to add a shape

Edit `bench/shapes.py`:

```python
SHAPES = [
    Shape("8b-b4", M=4, N=4096, K=4096, group_size=128, priority=0),
    # ...
    Shape("your-shape", M=..., N=..., K=..., group_size=..., priority=...),
]
```

Tests and benchmarks pick up new shapes automatically.
