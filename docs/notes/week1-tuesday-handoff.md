# Week 1 Tuesday handoff

> Converts Monday evidence into a clean handoff per the workweek checklist:
> a 10-line result note, a dashboard status check, and one chosen first
> upstream touchpoint candidate. No new measurements today.

## Result note (10 lines)

1. Benchmark: `torch.matmul`, M=N=K=4096, fp16, container `nvcr.io/nvidia/pytorch:25.03-py3` (torch 2.7.0a0+nv25.03, CUDA 12.8), 50 iters × 5 repeats, p50 reported.
2. A100 SXM4-80GB: **227.2 TFLOPS**, p50 0.6144 ms, compute_SOL **72.8%**, dram_SOL 73.4%, sm_throughput 13.3%, NCU complete, correctness OK (max_abs_err = 0).
3. H100 NVL: **494.6 TFLOPS**, p50 0.2722 ms, compute_SOL **50.0%** (vs 989 TF table peak), dram_SOL 53.0%, sm_throughput 19.5%, NCU complete, correctness OK.
4. H200: **752.0 TFLOPS**, p50 0.1793 ms, compute_SOL **76.0%**, dram_SOL 58.4%, sm_throughput 18.7%, NCU complete, correctness OK.
5. Marlin w4a16 smoke (Triton backend) at M=N=K=64 fp16: A100 0.062 ms, H100 0.032 ms — too small to publish, kept only as backend-plumbing smoke.
6. **Side-finding (loose end, not the touchpoint):** H100 NVL underperforms peers on the same kernel call; 22.8 pp below A100 and 26 pp below H200 on table-peak compute_SOL.
7. **Methodology caveat:** `swordfish/runner/schema.py:23` uses `989 TF` as the H100 fp16 peak (SXM5 spec). H100 NVL's published dense fp16 peak is ~835 TF; corrected NVL SOL is **~59.2%**. Gap to H200 narrows from 26 pp to 17 pp but does not close.
8. Bottleneck shape on H100 NVL: dram_SOL 53% and sm_throughput 19% — neither memory- nor compute-bound at the kernel level.
9. All three rows reproducible from `runs/airun/week1/torch-gemm-{a100,h100,h200}.json` plus matching `*.ncu.csv`; raw `.profile.raw.json` and `.raw.json` indexed-out as `raw intermediate result`.
10. No external performance claim is published yet. Friday's first artifact comes from the Liger-fleet-profile work scoped below, not from this fp16-GEMM smoke.

## Dashboard status check

Refreshed via `make dashboard-index completion-report airun-validate-results`:

- `docs/dashboard/results-index.json` — **5 result rows indexed**, 7 raw/intermediate files correctly skipped, schema `swordfish.result_index.v1`.
- `docs/dashboard/completion-report.md` — **Status: READY**, gate passes for `a100, h100, h200`, all three rows show `matches_reference=True` and `ncu_complete=True`.
- `make airun-validate-results` — **PASSES**: "GEMM result matrix is complete" with `--require-ncu` enforced across A100/H100/H200.

No drift between Monday's evidence and the dashboard. The strict completion gate (every requested arch has a valid matching result with NCU) is green.

## First upstream touchpoint candidate (chosen from measured gaps)

**Candidate:** **Liger Kernel cross-fleet training profile (LinkedIn / Microsoft).**
Profile Liger Kernel's training improvements across the AKS GPU fleet
(A100 SXM4 / H100 NVL / H200) and publish the cross-arch numbers, with the
profiling harness landing in `swordfish` for reuse.

### Why this and not the existing ledger rows

The existing ledger (Triton, PyTorch/Inductor, CUTLASS/CuTe, vLLM, ONNX
Runtime, pyptx) was built around the quant-inference lane. Each row's
"first touch" preconditions require data swordfish does not yet have at
publishable shape (real-shape quant GEMM, FP8 example, contiguous fp16
Triton GEMM, etc.). Forcing one of those rows now would either be a
manufactured pretext (the H100 NVL anomaly routed through Inductor) or
underweight (a 64³ marlin smoke routed through Triton).

Liger Kernel is a better first touch because:

- **The cross-arch gap is genuinely unpublished.** Liger's headline benchmark
  is `Llama-3-8B, BS=8, bf16, AdamW, gradient checkpointing, FSDP1, 8×A100`.
  Per-kernel benchmarks in the Liger repo (RMSNorm, RoPE, SwiGLU,
  FusedLinearCrossEntropy, post-training losses) typically run on a single
  A100 or H100. Side-by-side **A100 vs H100 NVL vs H200** at consistent
  shape/batch/dtype, with NCU SOL/roofline fields, has not been published
  by the Liger maintainers, by Azure, or by the broader community as far
  as Tuesday's quick search shows.
- **Microsoft visibility is direct, not indirect.** Liger Kernel ships from
  LinkedIn (Microsoft). The AKS GPU SKUs we measure on map 1:1 to Azure
  ND-series (NDm_A100_v4 = 8×A100 SXM4, ND_H100_v5 = 8×H100 NVL,
  ND_H200_v5 = 8×H200). "Liger speedups across the Azure GPU lineup"
  fits both the LinkedIn maintainer story and the Azure field-marketing
  story.
- **It exercises swordfish end-to-end on a real workload.** A 4096³ GEMM
  smoke proves the harness mechanics. A Liger-instrumented training
  microbenchmark sweep proves the harness on a workload practitioners
  actually run. That trust-builder is the right thing to land before any
  ORT/vLLM PR.
- **Maintainer risk is low.** Liger already has an AMD CI matrix
  (cross-vendor reproduction is in their DNA). Extending the published
  numbers to cross-NVIDIA-SKU is the obvious next axis. The contribution
  shape is a discussion or issue with reproducible numbers, not a kernel
  PR.
- **Lane consistency, honestly described.** This is *training-side* work,
  not quant inference. It does not redirect the FP8/FP4/INT4 lane. It is
  "Liger profile in swordfish, infra and provenance reused by the quant
  lane later." That distinction is recorded in
  `docs/notes/liger-first-touch.md`.

### Concrete week scope (Wed → Fri)

**Wednesday (W1) — single-GPU per-kernel sweep:**

1. Stand up a `swordfish.runner` bench wrapper around Liger's per-kernel
   benchmarks for: `LigerRMSNorm`, `LigerRoPE`, `LigerSwiGLU`,
   `LigerFusedLinearCrossEntropy`. Each run reports HF-baseline vs
   Liger-patched on the **same** input.
2. Run that sweep on 1×A100, 1×H100 NVL, 1×H200 via the existing AKS Kueue
   path (`make airun-apply` semantics). bf16 to match Liger's defaults.
3. Capture: per-kernel forward+backward time (mean, p50, p95), peak
   reserved GPU memory (`torch.cuda.max_memory_reserved`), correctness
   delta (`max_abs`, `max_rel`, `cosine`) of HF vs Liger output, NCU SOL/dram
   metrics on both paths, and full `swordfish` env block.
4. Land result rows under `runs/airun/week1/liger-perkernel/` so
   `make dashboard-index` picks them up.

**Thursday (W1) — end-to-end stretch, capacity-permitting:**

1. Reproduce Liger's Llama-3-8B step-time + peak-memory headline on
   8×A100 (their published config). This is the parity row that lets the
   cross-arch numbers be cited next to Liger's own reference.
2. If 8×H100 NVL or 8×H200 capacity is available on the AKS fleet,
   repeat. If not, document the capacity gap as a known constraint and
   ship single-GPU per-kernel only.
3. Update the dashboard with end-to-end rows alongside the per-kernel
   rows.

**Friday (W1) — publish:**

1. Writeup as `docs/profiling/liger-fleet-2026-w1.md` with the cross-arch
   tables, methodology (config, container image, driver, torch, Liger
   commit), correctness deltas, and one short conclusion per kernel.
2. Maintainer-ready packet via `swordfish.runner render-upstream-packet
   --target liger`. (New `--target liger` packet template needed; small.)
3. **Public artifact:** open a GitHub Discussion (not Issue, not PR) on
   `linkedin/Liger-Kernel` titled approximately "Cross-arch reproduction:
   Liger per-kernel and Llama-3-8B numbers on A100 / H100 NVL / H200",
   linking the swordfish writeup and result JSONs. Discussion not Issue
   so we are sharing data, not implying a bug.
4. Add the Liger Kernel row to `docs/contributions.md` reflecting the
   published artifact.

### Honest constraints flagged for Wednesday-onward

- **Result schema extension required.** Current `swordfish.runner`
  result protocol is shaped for inference (`tflops`, `latency.{mean,p50,p95}_ms`,
  `compute_sol_pct`, `estimated_hbm_sol_pct`). Liger benchmarks need
  training metrics (`tokens_per_second`, `peak_gpu_mem_gb`,
  `iter_time_ms`, optimizer/grad-checkpointing config). Plan: add a
  sibling schema `swordfish.training_result.v1` rather than overloading
  the inference one. Tracked as `swordfish-result-schema-training`.
- **Multi-GPU capacity risk.** The H200 single-GPU capacity blocker
  recorded in `docs/airun/h200-blocker-handoff.md` was for 1×H200; an
  8×H200 reservation is materially harder. Plan: scope Wednesday to
  single-GPU per-kernel benchmarks (which need only 1× of each arch),
  and treat 8×A100 / 8×H100 NVL / 8×H200 e2e as Thursday stretch.
- **A100 NCU permission constraint.** Documented in
  `docs/airun/a100-ncu-blocker.md` (DCGM exporter exclusion window).
  Same constraint applies to the Liger sweep on A100; the runner
  preflight already handles it via `airun-a100-ncu-preflight`.
- **"Improvement" baseline definition.** Liger claims ~20%
  throughput / 60% memory vs HF reference at fixed config. We must
  publish the **same** HF reference path (no Flash-Attention swap, no
  custom optimizer) to make the comparison fair, then optionally show a
  second row that adds Flash-Attention to both sides.

### Loose ends from the GEMM smoke (kept, not the touchpoint)

- `peak-table-h100-nvl` (todo): split `GPU_PEAKS["h100"]` into NVL vs SXM
  before any H100 SOL row is published externally.
- H100 NVL fp16 4096-cube SOL anomaly: still real (17 pp gap to H200 after
  peak correction). Worth a follow-up `torch.compile` sweep at some point,
  but not the first upstream touchpoint and not on Wednesday's critical
  path.

## Status of contributions ledger after Tuesday

`docs/contributions.md` row updates that this handoff implies:

| Upstream | Was | Now |
| --- | --- | --- |
| **Liger Kernel** | (not in ledger) | **planned, candidate confirmed; cross-arch profile scoped Wed–Fri** |
| Triton | planned, Not started | unchanged — still needs a contiguous fp16 GEMM Triton run |
| PyTorch/Inductor | planned, Not started | unchanged (deferred from Tuesday — H100 NVL `torch.compile` sweep is now follow-up, not first touch) |
| vLLM | planned, Not started | unchanged — w4a16 smoke too small; revisit when a real shape lands |
| CUTLASS/CuTe | planned, Not started | unchanged — needs an FP8/FP4 example repro |
| ONNX Runtime | planned, Not started | unchanged — no data on file |
| pyptx | planned, Not started | unchanged — no data on file |

Ledger row update is left for the user to apply alongside the Monday
repo-reset commit so the working tree stays a single coherent
transaction.
