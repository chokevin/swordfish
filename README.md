# 🗡️ swordfish

> Ampere-native INT4 × FP16 decode kernels — a faster cousin of [Marlin](https://github.com/IST-DASLab/marlin), tuned for voice-agent batch sizes (1–16) on A100.

## Why

Marlin is excellent but was tuned for a different era of deployments:

- Batch size 1 speculative decode workloads
- group_size = 128 only
- Pre-CUDA-graphs, pre-Triton-3.x

Real 2026 A100 inference looks different:

- Voice agents at **batch 4–16** is the dominant shape
- Per-tenant LoRA requires **arbitrary group sizes**
- CUDA graphs capture demands kernel-shape stability
- Multi-replica PD-disaggregated servers need **predictable tail latency**, not just peak throughput

`swordfish` targets these shapes specifically. It ships:

- A Triton INT4 × FP16 decode kernel optimized for batch 1–16 on A100
- A CUTLASS 3.x path for larger batches (experimental)
- A benchmark harness comparing against Marlin, CUTLASS, vLLM-builtin, FP16 baseline
- A vLLM backend plugin (PR-bound)

## Non-goals

- Not a training kernel (use Apex or TE)
- Not an H100/Blackwell kernel — Hopper has FP8 and WGMMA, different design space
- Not a drop-in Marlin replacement — Marlin is still faster at its sweet spot (batch=1, group=128)

## Status

🚧 Week 1 of 8 — profiling baseline, benchmark harness

| Milestone | Status |
|-----------|--------|
| Marlin profiling baseline | ⏳ in progress |
| Benchmark harness | ⏳ in progress |
| Triton baseline (match Marlin ±30%) | ⬜ |
| Triton tuned (beat Marlin @ batch 4-16) | ⬜ |
| CUTLASS path for batch >16 | ⬜ |
| Arbitrary group_size support | ⬜ |
| vLLM backend integration | ⬜ |
| Upstream PR | ⬜ |

See [`docs/roadmap.md`](docs/roadmap.md) for the 8-week plan.

## Quick start

```bash
uv sync
uv run pytest tests/
uv run python -m bench.run_bench --shapes voice
```

## Target shapes (voice-agent decode)

See [`docs/shapes.md`](docs/shapes.md) for the full catalog. Headline shapes:

| name | M (batch) | N (hidden) | K (hidden) | group_size | notes |
|------|-----------|------------|------------|------------|-------|
| voice-tiny | 1 | 4096 | 4096 | 128 | baseline, Marlin's sweet spot |
| voice-4 | 4 | 4096 | 4096 | 128 | typical voice concurrency |
| voice-8 | 8 | 4096 | 4096 | 128 | busy voice replica |
| voice-16 | 16 | 4096 | 4096 | 128 | ceiling for one A100 |
| voice-70b | 1..16 | 8192 | 8192 | 64/128 | 70B model decode |

## Repository layout

```
swordfish/
├── swordfish/          # the package
│   ├── kernels/        # Triton and CUTLASS kernels
│   ├── reference.py    # naïve PyTorch reference for correctness
│   └── pack.py         # weight quantization + packing
├── bench/              # benchmark harness
├── tests/              # correctness tests
├── docs/               # design, shapes, roadmap
└── .github/workflows/  # CI
```

## License

MIT
