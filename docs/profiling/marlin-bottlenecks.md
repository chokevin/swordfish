# Marlin bottlenecks — A100, voice-decode shapes

> **STATUS:** Filled in from autoresearch run `20260420T010050Z` (PR #1).
> ncu per-kernel counters were unavailable (DCGM holds them exclusively on
> the voice-agent-flex cluster; see "Constraint" below), so the bottleneck
> classification here is based on **wall-clock timing + first-principles
> roofline arithmetic + Perfetto trace** rather than ncu metrics. The
> classification still resolves cleanly because the dominant bottleneck
> at these shapes turns out to be visible from timing alone.

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

| shape | M×N×K (g) | fp16 ms (p50) | marlin ms (p50) | marlin TFLOPS | speedup vs fp16 | dominant bottleneck | proof |
|---|---|---|---|---|---|---|---|
| 8b-b1       | 1×4096×4096 (128)  | 0.031 | 0.049 |  0.67 | **x0.62** | **launch / fixed overhead** | marlin time flat across batch (b1=49µs, b8=49µs) — independent of work |
| 8b-b4       | 4×4096×4096 (128)  | 0.031 | 0.049 |  2.7  | **x0.63** | launch / fixed overhead | same flat 49µs |
| 8b-b8       | 8×4096×4096 (128)  | 0.031 | 0.049 |  5.5  | **x0.63** | launch / fixed overhead | same flat 49µs |
| 70b-tp2-b1  | 1×8192×4096 (128)  | 0.049 | 0.051 |  1.3  | x0.99 | launch / fixed overhead (both impls) | both impls hit ~49–51µs floor |
| 70b-tp2-b4  | 4×8192×4096 (128)  | 0.049 | 0.049 |  5.4  | x1.00 | launch / fixed overhead (both impls) | both at floor |
| 70b-tp2-b8  | 8×8192×4096 (128)  | 0.049 | 0.049 | 10.9  | x1.00 | launch / fixed overhead (both impls) | both at floor |

### What we expected vs what we measured

The taxonomy in the next section listed HBM-BW / TC-feed / scheduler-stall
/ tile-mismatch as the candidates. The data eliminates all four:

- **Not HBM-bound.** First-principles HBM time for marlin int4 weights at
  `N=K=4096` is `4096*4096*0.5 / 2.039e12 ≈ 4.1 µs`. Measured: 49 µs.
  Marlin is at **8% of HBM peak** at this shape — nowhere near the memory
  roof.
- **Not compute-bound.** At M=8 the FLOP count is `2*8*4096*4096 = 268M`.
  At A100 FP16-TC peak (312 TF) that is `0.86 µs` — two orders of
  magnitude under the measured 49 µs.
- **Not tile-mismatch under-occupancy.** That hypothesis predicts
  marlin's time *grows* as M shrinks (because we waste warps on padding).
  It doesn't grow — it's flat.
- **Not scheduler stalls** in the usual sense. With kernels this short the
  GPU front-end is effectively idle between launches; warp stalls don't
  matter when the kernel barely runs.

The 5th category — added below — wins: **launch / fixed-overhead bound.**

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
- **Launch / fixed-overhead bound** _(new, added from this run)_ — kernel
  wall time is **independent of M** across the small-batch range, sitting
  at a flat floor (~30–50 µs on A100 for small CUDA kernels under cuBLAS
  / Marlin), and the floor is at least 5× the HBM-time lower bound for
  the workload. The kernel never gets a chance to be memory- or
  compute-bound because launch/setup/sync dominates. Attack surface:
  reduce per-call overhead — kernel launch, workspace alloc, scale-tensor
  setup, host↔device sync.

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

> **On voice-decode shapes (M=1–8, N=4096–8192, K=4096), Marlin sits at a
> flat ~49 µs floor that is independent of batch size and ~12× the HBM-BW
> lower bound for INT4 weight loads.** This is launch / fixed-overhead
> bound, not memory- or compute-bound. fp16 cuBLAS hits a similar floor
> (~31 µs at 8b shapes, ~49 µs at 70b-tp2 shapes) but Marlin's floor is
> higher, so Marlin **loses** at the smallest shapes (x0.62 vs fp16 at
> 8b-b1) and ties on the larger ones.
>
> **Swordfish wins by reducing fixed per-call overhead, not by being
> faster in the hot loop.** The hot loop barely runs at these sizes.
> Concretely:
>
> 1. **Persistent kernel** that takes a queue of M=1..16 calls and amortizes
>    launch + setup once. This alone could halve the floor.
> 2. **Pre-baked workspace** — Marlin allocates an int32 workspace per call
>    (`marlin_compat.py:90`); pre-allocate once at session start.
> 3. **Fuse scale/zero handling into the prologue** instead of the
>    separate dequant kernel Marlin emits. Saves one launch out of N.
> 4. **CUDA Graph capture** the whole decode call. With the kernel itself
>    being short, graph replay can erase 10–20 µs of host-side overhead
>    per token, which at 50 tokens/sec is 0.5–1 ms/sec saved per layer.
>
> A larger-batch baseline (M=32–128) is still useful as a sanity anchor —
> at those shapes we expect to enter the HBM-bound regime where Marlin's
> in-loop dequant cleverness actually starts to matter. That's deferred
> to a separate W2/W3 sweep; W1's job was to identify the regime, and
> the regime here is **launch-overhead, not bandwidth**.

## Roofline

The roofline plot (`./marlin/roofline.png`) is unavailable for this run
because per-kernel ncu data was blocked by DCGM (see Constraint below).
The first-principles roofline above (8% of HBM peak at 8b shapes) is
sufficient to land the bottleneck classification. A future run on a
profile-permitted node will fill in the visual.

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

The bottleneck classification above doesn't depend on ncu data because
the dominant signal (flat-floor wall time vs. 5× HBM-roof headroom) is
unambiguous from timing alone. For deeper questions in W2+ (e.g., which
specific stall reason inside Marlin's hot loop), we need ncu and will
need ops to pause DCGM for the run window. See PR #1 description for
the open ask.

## Source artifacts

- Raw results: `docs/profiling/marlin/20260420T010050Z/results.csv`
- Perfetto trace: `docs/profiling/marlin/20260420T010050Z/trace.json`
  (open at https://ui.perfetto.dev)
- Run summary: `docs/profiling/marlin/20260420T010050Z/SUMMARY.md`
- Cluster PR: https://github.com/chokevin/swordfish/pull/1
