# swordfish

`swordfish` is a public kernel-shipping lab. The current mission is to build
reproducible A100/H100/H200 evidence for inference-kernel work, then turn that
evidence into upstream contributions across vLLM, ONNX Runtime, Triton,
PyTorch/Inductor, CUTLASS/CuTe, JAX/Pallas, TileLang, and pyptx.

## Week 1

| Day | Task |
| --- | --- |
| Mon | Freeze 4096³ fp16 torch/cuBLAS GEMM spec; ship strict JSON schema, runner, and A100/H100/H200 matrix with NCU. |
| Tue | Convert Monday evidence to a clean handoff: 10-line result note, dashboard status check, first upstream touchpoint candidate. |
| Wed | Run single-GPU Liger per-kernel sweep (RMSNorm/RoPE/SwiGLU/FusedLinearCE) vs HF reference on A100, H100 NVL, H200; capture latency, peak memory, correctness, NCU SOL. |
| Thu | Stretch: reproduce Liger's Llama-3-8B FSDP1 step on 8×A100; extend to 8×H100 NVL / 8×H200 if capacity allows. |
| Fri | Publish cross-arch Liger writeup, generate upstream packet, open GitHub Discussion on `linkedin/Liger-Kernel`, close out the contributions ledger row. |

### Week 1 detail

- **Monday baseline.** `torch.mm`, M=N=K=4096, fp16, 10 warmup + 50 timed iterations × 5 repeats. One full A100, one full H100 NVL, one full H200 once H200 capacity/preflight is healthy. Non-goals: no custom kernel, no tuning, no SOTA claims.
- **Tuesday handoff.** Detailed in [`docs/notes/week1-tuesday-handoff.md`](docs/notes/week1-tuesday-handoff.md). The chosen first upstream touchpoint is Liger Kernel cross-fleet training profile (LinkedIn / Microsoft); rationale and contribution shape in [`docs/notes/liger-first-touch.md`](docs/notes/liger-first-touch.md).
- **Wednesday measurement.** bf16 to match Liger defaults; baseline is the unmodified Hugging Face reference; rows land under `runs/rune/week1/liger-perkernel/`.
- **Friday artifact.** Writeup at `docs/profiling/liger-fleet-2026-w1.md`; maintainer packet via `swordfish.runner render-upstream-packet --target liger`; public artifact is a Discussion (not Issue, not PR) sharing reproducible JSON.

### Week 1 preconditions tracked separately

- `training_result.v1` sibling result schema for training-side metrics (`tokens_per_second`, `peak_gpu_mem_gb`, `iter_time_ms`, optimizer config, `liger_patch` block).
- `--target liger` packet template for `render-upstream-packet`.
- `GPU_PEAKS` SKU split (H100 NVL vs SXM5) before any H100 SOL row is published externally — loose end carried over from Monday's GEMM smoke.

## Quick start

```bash
uv sync
uv run pytest
uv run python -m swordfish.runner run-gemm \
  --backend torch \
  --m 32 --n 32 --k 32 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --arch-label a100 \
  --out /tmp/swordfish-gemm-smoke.json
```

The transformer reference also has a tiny forward benchmark smoke:

```bash
uv run python -m swordfish.runner bench-transformer \
  --scope block \
  --preset tiny \
  --batch-size 2 --seq-len 4 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --out /tmp/swordfish-transformer-smoke.json
```

To time a full training step instead of inference-only forward, use
`--mode train-step`. This runs forward, loss, backward, and an AdamW optimizer
step on the same tiny GPT-style reference:

```bash
uv run python -m swordfish.runner bench-transformer \
  --mode train-step \
  --scope model \
  --preset tiny \
  --batch-size 1 --seq-len 3 \
  --dtype fp32 \
  --repeats 1 --warmup 0 --iters 1 \
  --device cpu --allow-cpu \
  --out /tmp/swordfish-transformer-train-step-smoke.json
```

On a CUDA host, drop `--allow-cpu`, use `--device auto`, and run the Week 1
shape:

```bash
uv run python -m swordfish.runner run-gemm \
  --backend torch \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --repeats 5 --warmup 10 --iters 50 \
  --device auto \
  --out runs/week1/torch-gemm-a100.json
```

To dispatch a Kueue Job through `rune` (the day-to-day path), preview the
manifest first:

```bash
make rune-install-profiles                              # one-time symlink

rune submit my-bench \
  --profile swordfish-bench-h100 \
  --script infra/rune/scripts/swordfish-bench.sh \
  --output /data/swordfish/week1/torch-gemm-h100.json \
  --dry-run=client \
  -- run-gemm --backend torch --m 4096 --n 4096 --k 4096 \
     --dtype fp16 --device auto \
     --out /data/swordfish/week1/torch-gemm-h100.json
```

Drop `--dry-run=client` to actually submit, then fetch the JSON back with
`rune submit get my-bench -n ray --output raw > my-bench.json`. The Python SDK
`swordfish.dispatch.LigerPerkernelRun` wraps the same flow as a typed dataclass.

A100 + Nsight Compute is a known limitation: rune profiles can't currently
expose container `SYS_ADMIN`, which NCU needs to read A100 perf counters. The
fix lives in rune (let profiles request `securityContext.capabilities.add:
SYS_ADMIN` under a kueue-gated allowlist). H100 NVL and H200 NCU run cleanly.

After all three jobs have produced final JSON files, use the strict completion
gate:

```bash
make validate-results
```

It fails until A100, H100, and H200 each have a schema-valid `torch-gemm-*.json`
with matching arch provenance, passing correctness, and complete NCU metrics.
The Make target searches recursively under the result directory so timestamped
run subdirectories are accepted as long as each arch has exactly one matching
final JSON.

## Repository layout

```text
swordfish/
├── swordfish/runner/        # GEMM smoke runner and result schema
├── swordfish/transformer/   # PyTorch GPT-style reference model and benchmark
├── swordfish/dispatch/      # Python SDK wrapping `rune submit`
├── infra/rune/              # rune profiles, image, and in-pod bench script
├── docs/dashboard/          # 35-week roadmap dashboard
├── docs/research/           # research notes for upstream contribution lanes
└── tests/                   # local schema/CLI tests
```

## Result contract

Every benchmark result should record the benchmark config, git SHA, dirty flag,
host, torch/CUDA versions, GPU name/class/compute capability, latency samples,
summary latency stats, achieved TFLOP/s, estimated bandwidth, rough SOL
percentages, finite-output status, checksum, and torch-reference correctness
error fields for non-torch backends. The common cross-GPU contract is documented
in [`docs/benchmarking.md`](docs/benchmarking.md).

## Backend contract

`run-gemm` now has a `--backend` switch. `torch` is the correctness and cuBLAS
baseline; `triton` is a deliberately small educational matmul kernel for the
first custom-kernel comparison. `cutlass` is wired as the native CuTe/CUTLASS
extension slot and fails with explicit build instructions until that optional
extension exists. The raw-PTX lane starts with a handwritten vector-add PTX
artifact under `swordfish/kernels/ptx/`; it is intentionally blocked on a CUDA
driver loader instead of pretending `torch.add` is raw PTX. Future raw-PTX
benchmarks should plug into the same backend interface so timing, correctness,
NCU, and JSON output do not fork per kernel.

## License

MIT
