# swordfish — 8-week roadmap

## Goal

Ship an INT4×FP16 decode kernel for A100 that **beats Marlin by 10%+ at batch 4–16 on voice-decode shapes**, is **merge-ready for vLLM**, and has **clean support for arbitrary group sizes**.

## Success criteria (in priority order)

1. Correctness: numerically matches FP16 reference within 1e-2 relative error for all target shapes, all group_sizes in {-1, 32, 64, 128}.
2. Speed: beats Marlin by ≥10% at batch 4–16 for the voice-decode shape catalog.
3. Coverage: supports group_size in {-1, 32, 64, 128}, asymmetric + symmetric quant.
4. Integration: works as a vLLM backend behind a `--quantization swordfish` flag.
5. Documentation: blog post + merge-ready PR.

## Weekly plan

### Week 1 — Baseline and profiling
- [x] Reproduce Marlin's published numbers on our A100 box _(see PR #1 / docs/profiling/marlin-bottlenecks.md)_
- [x] Profile Marlin with `nsys` and `ncu` across all target shapes _(nsys + perfetto: yes; ncu blocked by DCGM on voice-agent-flex — documented constraint, ncu not needed for bottleneck classification at these shapes)_
- [x] Identify per-shape bottlenecks _(launch / fixed-overhead bound at all voice-decode shapes — surprise finding, **not** HBM-bound as roadmap originally assumed)_
- [x] Build the benchmark harness with clean CSV output _(`bench/run_bench.py`, `results.csv`)_
- [x] Naïve reference (PyTorch) for correctness validation _(`swordfish/reference.py`)_
- [x] **L0 correctness in bench:** every impl's output is `allclose`-checked against `reference_w4a16_matmul` on identical packed weights before timing — kernels with arithmetic bugs flunk loudly with a `max_relerr` field instead of producing misleading "fast but wrong" speed numbers. _(`bench/run_bench.py:bench_shape`)_
- [x] **L2 perplexity harness:** `bench/eval_ppl.py` swaps every `nn.Linear` in a HF causal LM with a `QuantLinear[impl]` (sharing one packed/scales tensor across impls so quantization is held constant) and computes WikiText-2 PPL. Exits non-zero if swordfish drifts > 0.001 PPL from marlin on the same packed weights — that's a kernel-arithmetic bug masked by averaging. Bands: marlin vs fp16 ≤ 0.15 (the quant tax we accept), swordfish vs marlin ≤ 0.001 (we accept ~0).
- **Exit:** ✓ `bench/run_bench.py` produces a per-shape comparison table; we know *why* Marlin appears slow at our shapes — its **GPU kernel is fast** (17.5 µs vs cuBLAS 23 µs at 8b shapes) but our `swordfish/marlin_compat.py` wrapper adds ~30 µs of host-side overhead per call vs cuBLAS's ~8 µs. **Wall-clock loss is in the wrapper, not the kernel.** Swordfish's W2+ attack is **integration-overhead reduction first** (pre-allocated workspace, CUDA Graph capture), kernel rewriting second.

### Week 2 — Wrapper overhead, then Triton baseline
**Reordered after W1 finding + rvLLM lessons** (`docs/lessons-rvllm.md`): the
wrapper, not the kernel, is the W2 attack surface. Graph capture alone moved
rvLLM's stack 27× before any kernel work; we expect it to flip Marlin from
x0.62 to x>1 vs fp16 at 8b shapes WITHOUT touching CUDA.

- [x] Workspace cache in `marlin_compat.py` (eliminates per-call int32 alloc) _(done this commit)_
- [x] L0 correctness gate (`allclose AND cosine ≥ 0.999`) so wrong kernels don't ship speed numbers _(done this commit)_
- [x] `--capture` flag in `bench/run_bench.py` — measures wrapper overhead per impl as `eager_ms - captured_ms`, so we can see the win before writing it _(done this commit)_
- [ ] **Captured marlin entry point** — `marlin_matmul_captured(a, B, s, *, out)` that records into a CUDA graph at first call, replays thereafter. Pre-allocated `out` required (rvLLM §5.2).
- [ ] Eliminate per-call dtype/contiguity asserts in `marlin_matmul` (~5 µs win, free).
- [ ] Write a clean Triton INT4×FP16 decode kernel
- [ ] Packing utility compatible with Marlin's layout (for apples-to-apples)
- [ ] Correctness passes for all voice-decode shapes (auto via L0 gate)
- [ ] Match Marlin within 30% at batch=1 (don't optimize yet)
- **Exit:** Triton kernel is correct and in the benchmark table; captured-marlin row is in the bench output; we have a measured `wrapper_overhead_pct` per shape.

### Week 3 — Triton tuning for batch 1–4
- [ ] Tile size sweep, num_warps, num_stages
- [ ] Fuse zero-point handling, bias add, SiLU if caller wants
- [ ] Use `tl.multiple_of` and `tl.max_contiguous` for load hints
- [ ] Understand why Triton loses at batch=4–16 (we expect it will)
- **Exit:** Triton matches Marlin at batch=1; we have a clear hypothesis for batch 4–16.

### Week 4 — CUTLASS path for batch 4–16
- [ ] Decide: deeper Triton tuning vs CUTLASS collective?
- [ ] If CUTLASS: start from `examples/55_hopper_mixed_dtype_gemm` and back-port to SM80
- [ ] Implement dequant epilogue (or prologue, depending on layout)
- [ ] First correctness + benchmark
- **Exit:** CUTLASS kernel matches Marlin at batch=4–16.

### Week 5 — Beat the baseline
- [ ] The actual hard work: tune tile/warp/stage until we beat Marlin by 10%+
- [ ] Swizzled weight layout for bank-conflict avoidance
- [ ] Prefetch distance tuning (cp.async stages)
- [ ] Consider persistent-thread-block pattern
- **Exit:** ≥10% faster than Marlin at batch 4–16 on voice-decode shapes.

### Week 6 — Coverage expansion
- [ ] Arbitrary group_size: 32, 64, 128, -1 (channel-wise)
- [ ] Asymmetric quantization (non-zero zero-points)
- [ ] Fused bias, GELU/SiLU epilogues
- [ ] CUDA graph capture compatibility
- **Exit:** coverage matrix documented, benchmarks across all combinations.

### Week 7 — vLLM integration
- [ ] Write the vLLM backend plugin
- [ ] End-to-end benchmark: Llama-3-70B-INT4 decode on 8×A100
- [ ] Voice-agent workload simulation
- [ ] Measure real-world throughput lift, not just microbenchmarks
- **Exit:** vLLM serves a 70B model with swordfish; measurable end-to-end gain.

### Week 8 — Upstream and publish
- [ ] Open vLLM PR
- [ ] Engineering blog post
- [ ] Demo prep (Ignite / internal MS venue)
- [ ] Address PR feedback
- **Exit:** PR merged or in serious review; blog published.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Triton limits too restrictive for SM80 tensor-core feeding | Escape hatch to CUTLASS at week 4 |
| Marlin is genuinely optimal at our target shapes | Plan B: focus wins on flexibility (group sizes, epilogues) rather than raw speed; still merge-worthy |
| vLLM integration is painful | Ship as a standalone wheel with a `swordfish-vllm` plugin package first; upstream later |
| A100 cluster contention during profiling | Use 1 GPU for dev, reserve 8-GPU for final bench runs only |

## Non-goals (for this 8-week window)

- Hopper / Blackwell support (separate project)
- Training kernels (use Apex/TE)
- AWQ / GPTQ quantization methods — we accept an existing quantized checkpoint format
- FP8 support on A100 (not hardware-supported)

## Open questions

- [ ] Do we target Marlin's exact weight layout or define our own?
  - Pro of matching: zero-cost migration for existing checkpoints
  - Con: inherits Marlin's constraints we want to relax
  - Leaning: ship both — a compat path and an "optimal" path
- [ ] Triton vs CUTLASS as primary — or both?
- [ ] vLLM upstream vs plugin-first?
