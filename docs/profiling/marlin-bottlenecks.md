# Marlin bottlenecks — A100, voice-decode shapes

> **STATUS:** Filled in from autoresearch run `20260420T010050Z` (PR #1).
> **Important methodology correction:** the first pass of this analysis read
> only the wall-clock CSV and concluded "Marlin is launch-overhead bound on
> the GPU." The Perfetto trace shows that's wrong. Wall-clock measured by
> `cuda.Event` over an iter loop captures both kernel time AND host-side
> idle gaps when Python dispatch can't keep up. The trace separates them.
> ncu per-kernel counters were unavailable (DCGM holds them exclusively on
> the voice-agent-flex cluster — see "Constraint" below), so the bottleneck
> classification here is based on **wall-clock + Perfetto kernel durations
> + first-principles roofline arithmetic** rather than ncu metrics.

## Method

For each P0 voice-decode shape, we capture:

1. **bench/run_bench.py** — wall-clock TFLOPS via cuda.Event timing, NVTX
   ranges around each impl call.
2. **nsys** — timeline trace, NVTX-annotated, exported to SQLite for
   Perfetto and to `.nsys-rep` for the Nsight Systems UI.
3. **ncu** — per-kernel SOL, MemoryWorkloadAnalysis, SchedulerStats,
   WarpStateStats, ComputeWorkloadAnalysis, plus a flat CSV consumed by
   `bench/roofline.py` to place each kernel on the A100 roofline.
4. **torch.profiler** — Chrome trace (`trace.json`) loaded in
   https://ui.perfetto.dev for the unified Python+CUDA view.

Marlin is pinned at upstream SHA `1f25790bdd49fba53106164a24666dade68d7c90`.
A100 hardware peaks used in the roofline analysis: **2.039 TB/s HBM**,
**312 TFLOPS FP16 tensor-core** (80GB SXM SKU).

## Per-shape findings

Measured on A100-SXM4-80GB (cc 8.0), torch 2.4.0a0+nv24.05 / CUDA 12.4,
Marlin pinned at `1f25790`. Wall-clock from `bench/run_bench.py` (50 iters
per impl × 5 repeats, p50 reported), source `docs/profiling/marlin/20260420T010050Z/results.csv`.
Per-kernel GPU times read from `trace.json` (Perfetto), grouped by the
chunk corresponding to each shape's iter loop.

| shape | M×N×K (g) | fp16 wall (kernel) | marlin wall (kernel) | host overhead (marlin) | bottleneck |
|---|---|---|---|---|---|
| 8b-b1       | 1×4096×4096 (128) | 31 µs (23 µs cublas 64×64) | 49 µs (**17.5 µs marlin**) | **31.5 µs** | **wrapper / Python dispatch** |
| 8b-b4       | 4×4096×4096 (128) | 31 µs (23 µs cublas 64×64) | 49 µs (17.5 µs marlin) | 31.5 µs | wrapper / Python dispatch |
| 8b-b8       | 8×4096×4096 (128) | 31 µs (23 µs cublas 64×64) | 49 µs (17.5 µs marlin) | 31.5 µs | wrapper / Python dispatch |
| 70b-tp2-b1  | 1×8192×4096 (128) | 49 µs (46 µs cublas 128×64) | 51 µs (19.1 µs marlin) | ~32 µs | wrapper / Python dispatch |
| 70b-tp2-b4  | 4×8192×4096 (128) | 49 µs (46 µs cublas 128×64) | 49 µs (19.1 µs marlin) | ~30 µs | wrapper / Python dispatch |
| 70b-tp2-b8  | 8×8192×4096 (128) | 49 µs (46 µs cublas 128×64) | 49 µs (19.1 µs marlin) | ~30 µs | wrapper / Python dispatch |

### What the data actually says (corrected)

A first pass of this analysis read only the wall-clock CSV and concluded
that Marlin loses to fp16 because Marlin is "launch-overhead bound on the
GPU." That conclusion was wrong. The Perfetto trace shows the GPU-side
picture is the **opposite** of what the wall-clock suggests:

- **Marlin's GPU kernel is fast.** 17.5–19.1 µs per call, essentially
  flat across batch (which is correct behavior at small M — the kernel
  is paying the same fixed cp.async + tile-setup cost regardless of
  whether the M dimension is 1 or 8). This is **faster** than cuBLAS
  fp16 at the 8b shapes (cuBLAS picks `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2`
  at 23 µs) and competitive at the 70b-tp2 shapes (cuBLAS uses
  `ampere_fp16_s16816gemm_fp16_128x64_ldg8` at 46 µs).
- **Our wrapper loses the win.** Wall-clock for Marlin is 49 µs but the
  GPU kernel is 17.5 µs. The remaining **~30 µs is host-side overhead
  per call**: workspace alloc, scale-tensor manipulation, input
  validation, and the Python→ATen→Marlin dispatch chain in
  `swordfish/marlin_compat.py:73-92`. cuBLAS via `torch.matmul` only
  pays ~8 µs of host overhead because it's gone through years of
  PyTorch-team optimization (cached workspace, dispatch shortcuts,
  fused launch).
- **At small shapes, the wrapper IS the loss.** 31 µs (fp16 wall) vs
  49 µs (marlin wall) is a 0.62× speedup. Subtract the host overhead
  delta and at the kernel level Marlin wins ~1.3× at 8b shapes.
- **At larger shapes, the kernels dominate** and the speedup ratio
  rounds to 1.0× because both kernels are big enough to bury host
  overhead.

### Bottleneck taxonomy (pick from this list when filling cells)

- **HBM-BW bound** — `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  > 80%. Achieved TFLOPS sits *on* the slanted memory roof in the roofline
  plot. Adding compute won't help; we need fewer HBM reads or better reuse.
- **Tensor-core feed bound** — `sm__pipe_tensor_op_hmma_cycles_active.avg.pct`
  < 50% **and** kernel sits well below the flat compute roof. Tensor cores
  are starved for operands; the kernel is moving FP16 from SMEM/regs too
  slowly, often due to bank conflicts or a too-shallow cp.async pipeline.
- **Scheduler stall** — high `smsp__average_warp_latency_per_inst_issued`,
  with breakdown dominated by `stall_long_scoreboard` (waiting on
  global/shared) or `stall_mio_throttle`. Means warps are blocked on memory
  ops the scheduler can't overlap.
- **Tile-shape mismatch** — kernel achieves <30% of the relevant roof
  *and* `sm__warps_active.avg.pct` is low *and* M is small. Tiles are
  oversized for the batch; we're paying for warps we can't fill.
- **GPU-side launch / fixed-overhead bound** — kernel wall time on the
  GPU is independent of M and sits at the cp.async+tile-setup floor
  (~15–20 µs on A100 for a small INT4 GEMM). The kernel itself can't
  win further until the floor is reduced. Attack: persistent kernel,
  shorter pipeline, smaller tile.
- **Wrapper / Python dispatch bound** _(new — added from this run)_ —
  wall-clock per call is significantly larger than GPU kernel time
  (`wall − kernel > 10 µs`), AND wall-clock is independent of M while
  GPU kernel is independent of M. The GPU is idle waiting for the next
  Python launch. Attack: CUDA Graph capture of the call, pre-allocated
  workspace, eliminate per-call setup, or call from C++.

## Ridge-point analysis

A100 ridge AI (FP16-TC peak / HBM bw) = `312e12 / 2.039e12 ≈ 153 FLOPs/byte`.

For our shapes, the per-element cost of W4A16 decode is:
- Weight: 0.5 byte/element from HBM (INT4)
- Activation: 2 bytes/element (FP16, but reused M times across N)
- FLOPs: `2 * M * N * K`
- Bytes: `~ K*N/2 + M*K*2 + M*N*2` (weight + activation + output)

Per-shape arithmetic intensity (back-of-envelope, dominated by weight load
at small M):

| shape | AI (FLOPs/HBM-byte) | bound |
|---|---|---|
| M=1,  N=K=4096       | ~3.99   | **memory** (way left of ridge) |
| M=4,  N=K=4096       | ~15.9   | **memory** |
| M=8,  N=K=4096       | ~31.6   | **memory** |
| M=16, N=K=4096       | ~62.6   | **memory** (still left of ridge) |
| M=8,  N=8192, K=4096 | ~31.6   | **memory** |

**Implication:** at every voice-decode shape we care about, A100 sits
left of the ridge — we are memory-bound. Marlin's job is to keep HBM
saturated; swordfish's job is to keep HBM saturated *while* using the
spare compute headroom to fold scales/zeros and epilogue ops in for free.

(These are theoretical AIs; the roofline plot from `bench/roofline.py`
shows where Marlin actually lands per kernel. Fill in `roofline.png`
reference once produced.)

## Conclusion: what swordfish will attack

> **Marlin's GPU kernel is good — it beats cuBLAS fp16 at the 8b shapes
> (17.5 µs vs 23 µs) and ties at the 70b-tp2 shapes (19.1 µs vs 46 µs).
> The wall-clock loss to fp16 (x0.62 at 8b-b1) is entirely caused by
> ~30 µs of host-side per-call overhead in our `swordfish/marlin_compat.py`
> wrapper, not by anything wrong with the kernel.**
>
> This rewrites the W2+ plan. The original assumption — that we'd have
> to write a better kernel to beat Marlin — turns out to be backwards
> at our shapes: we have to **integrate Marlin better** before kernel
> work even matters. Concretely, in priority order:
>
> 1. **Pre-allocate the workspace tensor.** `marlin_compat.py:90` calls
>    `torch.zeros(N // 128 * 16, dtype=torch.int32, device=...)` per
>    matmul. Caching this per (N, device) saves one allocator round-trip
>    per call (~5–10 µs).
> 2. **Eliminate per-call dtype/contiguity checks.** The wrapper does
>    several `assert` and `.contiguous()` paths that are no-ops for our
>    pre-packed weights but still cost dispatch time.
> 3. **CUDA Graph capture the entire decode call.** Once the wrapper is
>    side-effect free per call, capturing the matmul + dequant_scales
>    sequence into a graph erases the remaining Python+ATen dispatch
>    overhead. For a 50 tok/s decode, that's ~30 µs × 32 layers ×
>    50 tok/s = ~50 ms/sec saved per stream.
> 4. **Only then** look at kernel-level wins. The 17.5 µs floor is the
>    cp.async pipeline + tile-setup cost; reducing it requires Triton
>    or CUTLASS reimplementation (W2+).
>
> A larger-batch sweep (M=32–128) is still useful as a sanity anchor —
> at those shapes both kernels grow proportionally to work and we can
> measure HBM-bound behavior properly. That's deferred to a separate
> W3 pass; W1's job was to identify what to attack and the answer is
> **integration overhead first, kernel optimization second**.

## Roofline

The roofline plot (`./marlin/roofline.png`) is unavailable for this run
because per-kernel ncu data was blocked by DCGM (see Constraint below).
We don't actually need a visual roofline at this point — the bottleneck
is host-side dispatch, which doesn't appear on a GPU roofline at all.
A future run on a profile-permitted node will fill in the visual when
we move to W3 kernel-level work.

## Constraint: ncu vs DCGM on production clusters

NVIDIA Nsight Compute (`ncu`) needs exclusive access to the GPU
performance-counter hardware. On voice-agent-flex (and most production
multi-tenant GPU clusters) NVIDIA Data Center GPU Manager (DCGM) is
running as a node daemon for fleet-wide monitoring and **holds these
counters exclusively**. Any `ncu` invocation in a workload pod fails with:

```
==ERROR== Profiling failed because a driver resource was unavailable.
Ensure that no other tool (like DCGM) is concurrently collecting
profiling data.
```

We tried two of the three available mitigations:

| mitigation | result |
|---|---|
| `securityContext.capabilities.add: [SYS_ADMIN]` | ✓ unblocks `ERR_NVGPUCTRPERM` (kernel-module gate) but not DCGM contention |
| schedule on a node with DCGM disabled | not feasible — cluster-wide |
| `dcgmi profile --pause` for the duration of the run | requires cluster-ops cooperation; tracked in #TODO |

For W1 we didn't actually need ncu — the Perfetto trace gave us per-kernel
GPU timing, which combined with wall-clock and first-principles
roofline math was sufficient to land the bottleneck classification
(host wrapper dominates, not GPU kernel). For W2+ deep-dives inside
the kernel itself we will need ncu and will need ops to pause DCGM
for the run window.

## Source artifacts

- Raw results: `docs/profiling/marlin/20260420T010050Z/results.csv`
- Perfetto trace: `docs/profiling/marlin/20260420T010050Z/trace.json`
  (open at https://ui.perfetto.dev — sort kernels by name to see Marlin
  vs `ampere_fp16_s16816gemm_*` per-call durations)
- Run summary: `docs/profiling/marlin/20260420T010050Z/SUMMARY.md`
- Cluster PR: https://github.com/chokevin/swordfish/pull/1
