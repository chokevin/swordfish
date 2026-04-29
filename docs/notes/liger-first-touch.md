# Liger Kernel first touch candidate

## Candidate

Profile Liger Kernel's training improvements across the AKS GPU fleet
(A100 SXM4-80GB, H100 NVL, H200) and publish the cross-architecture
numbers, with the profiling harness landing in `swordfish` for reuse.

The contribution shape is a **GitHub Discussion on `linkedin/Liger-Kernel`**
sharing the cross-arch reproduction data, plus a `swordfish` writeup and
public result JSONs. Discussion not Issue, because the goal is sharing
data, not implying a bug.

## Why this is a useful first touch

- **The cross-arch gap is genuinely unpublished.** Liger's headline is
  20% throughput / 60% memory on `Llama-3-8B, BS=8, bf16, AdamW,
  gradient checkpointing, FSDP1, 8×A100`. Per-kernel benchmarks in
  `Liger-Kernel/benchmark/scripts/` typically run on a single A100 or
  H100. Side-by-side **A100 vs H100 NVL vs H200** at consistent settings
  with NCU SOL fields and reproducible JSON has not been published by
  the maintainers, by Azure, or by the broader community as of Tuesday's
  scan.
- **Microsoft visibility is direct.** Liger ships from LinkedIn
  (Microsoft). AKS GPU SKUs map 1:1 to Azure ND-series. The artifact is
  citable from both the LinkedIn maintainer side and the Azure
  field-marketing side.
- **Maintainer-aligned.** Liger has an AMD CI matrix. Cross-NVIDIA-SKU
  is the obvious next axis of reproduction. Discussion-shaped artifacts
  with reproducible JSON have low maintainer cost.
- **Trust-builder before quant-lane PRs.** A clean cross-arch
  reproduction proves the swordfish AKS Kueue + NCU + result-protocol
  pipeline on a real workload. ORT and vLLM PRs benefit from that
  reputation.

## Why this is consistent with the lane decision

The `swordfish` lane is FP8 / FP4 / INT4 quant inference GEMM
(`docs/lane-quantization.md`). Liger is training-side, not quant
inference. Including it in the contributions ledger does **not**
redirect the lane:

- The infra it exercises (AKS Kueue, NCU profiling, schema-versioned
  result JSON, dashboard, packet generator) is the same infra the
  inference quant lane needs.
- The training-side result schema work is reusable for any future
  training-related contribution but optional for the inference lane.
- The artifact is one cross-arch reproduction Discussion, not an ongoing
  Liger maintenance commitment.

If maintainers reply with follow-up requests (a specific kernel, a
specific arch, a specific dtype), respond with one more measurement
batch and stop. The next contribution should return to the quant
inference lane.

## Contribution preconditions

The first touch is gated on these landing in `swordfish` first:

1. **Sibling result schema** `swordfish.training_result.v1` covering
   `tokens_per_second`, `peak_gpu_mem_gb`, `iter_time_ms`, optimizer
   config, gradient checkpointing flag, dtype, distributed strategy,
   and a `liger_patch` block with the Liger commit, patched modules,
   and on/off mode. Tracked as `swordfish-result-schema-training`.
2. **Per-kernel bench wrapper** that runs the four headline Liger
   kernels (`LigerRMSNorm`, `LigerRoPE`, `LigerSwiGLU`,
   `LigerFusedLinearCrossEntropy`) on identical input as the HF
   reference, captures forward+backward latency, peak memory, and
   correctness deltas (`max_abs`, `max_rel`, `cosine`). Lands as
   `swordfish.runner liger-perkernel`.
3. **Kueue manifest template** mirroring `airun-gemm` for the
   per-kernel sweep across A100, H100, H200. Single-GPU per-kernel
   first; multi-GPU FSDP step-time as Thursday stretch.
4. **Packet template** `--target liger` for
   `swordfish.runner render-upstream-packet`.

## Evidence command (sketch, single-GPU per-kernel)

```bash
# Once liger-perkernel runner exists:
uv run python -m swordfish.runner liger-perkernel \
    --kernels rmsnorm,rope,swiglu,fused_linear_ce \
    --dtype bf16 \
    --device cuda \
    --hidden 4096 --vocab 128256 --seq 2048 \
    --repeats 5 --warmup 10 --iters 50 \
    --liger-commit "$(git -C $LIGER_REPO rev-parse HEAD)" \
    --out runs/airun/week1/liger-perkernel/<arch>-<kernel>.json
```

Cross-arch wrapping reuses `make airun-render` / `make airun-apply`
with a new `AIRUN_CONFIG` for the Liger per-kernel sweep.

## Honest constraints

- **Multi-GPU capacity.** 8×H200 reservation on the AKS fleet is
  uncertain (single-H200 was already a capacity blocker, see
  `docs/airun/h200-blocker-handoff.md`). Plan single-GPU per-kernel as
  the publishable floor; multi-GPU e2e is stretch.
- **NCU on A100.** DCGM exporter exclusion window is required, same as
  the GEMM smoke; reuse `airun-a100-ncu-preflight`.
- **Fair baseline.** Liger's claim is vs an unmodified HF reference. The
  cross-arch run must publish both sides at identical config (no
  Flash-Attention swap, no custom optimizer, no extra fused kernels) so
  the speedup ratio is the published one. A second row may add
  Flash-Attention to both sides for a "modern stack" comparison, but
  must not be conflated with the headline number.
- **Numerical tolerance.** Liger advertises exact computation, so
  `max_abs_error` and `max_rel_error` should be zero or
  floating-point-noise-level. Any non-trivial delta is itself a finding
  worth surfacing privately to maintainers before publishing.

## Out of scope for the first touch

- Kernel modifications to Liger.
- Adding new fused losses or post-training kernels.
- AMD ROCm reproduction (Liger already has AMD CI; not the gap we are
  filling).
- Mamba/SSM or MoE kernel contributions to Liger (those are a separate
  conversation, not Week 1 work).
