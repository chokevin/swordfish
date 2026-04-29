# Quantization kernel lane

Decision: `swordfish` will specialize on quantized inference GEMM, centered on
FP8/FP4 and grounded by INT4 weight-only baselines.

## Problem

Transformer inference keeps pushing more work into small and medium GEMMs where
throughput, memory traffic, layout conversion, and serving constraints interact.
The useful kernel question is no longer "can we write a fast matmul once?" It is
"can we make the quantized GEMM path correct, measurable, and shippable across
the shapes that serving systems actually call?"

The lane is therefore:

- FP8 GEMM first, because it is production-relevant on Hopper and has strong
  reference points in DeepGEMM, TensorRT-LLM, CUTLASS/CuTe, vLLM, and ORT.
- FP4 next, because Blackwell makes it central and the ecosystem still needs
  careful correctness, scale-layout, and benchmark work.
- INT4 weight-only as the comparison baseline, because Marlin and Machete define
  the practical bar for compressed-weight serving on current systems.

## Why Microsoft cares

Microsoft serves models through multiple stacks: ONNX Runtime / ORT GenAI,
TensorRT-LLM integrations, PyTorch/Inductor paths, and product-specific serving
systems. A credible quantized GEMM lane helps in three ways:

- Lower inference cost by improving tokens per dollar on common decode and
  prefill shapes.
- Reduce risk by making correctness envelopes, scale layouts, and fallback paths
  explicit before kernels reach product traffic.
- Create reusable evidence across A100, H100/H200, and Blackwell instead of
  one-off benchmark claims that cannot survive a different GPU or serving stack.

## Reference bar

The initial SOTA map is not a list of things to clone; it is the yardstick for
what a useful contribution must understand.

| Reference | What it represents | What to learn from it |
| --- | --- | --- |
| DeepGEMM | Modern FP8 GEMM specialization | Hopper tensor-core use, scale handling, persistent scheduling, benchmark discipline |
| Marlin | Practical INT4 weight-only serving GEMM | Layout packing, small-batch decode priorities, correctness/perf tradeoffs |
| Machete | vLLM-oriented quantized GEMM path | Serving integration constraints and maintainable kernel variants |
| TensorRT-LLM FP8/FP4 | Production quantized inference baseline | Blackwell/Hopper feature direction, scale formats, end-to-end serving expectations |
| CUTLASS/CuTe | Production GEMM template substrate | How real kernels express tiling, pipelines, TMA/WGMMA, and architecture dispatch |

## Target architectures

- **A100 / Ampere:** baseline and regression guard. Important for existing
  fleet coverage, but not the primary place to invent the lane.
- **H100/H200 / Hopper:** primary execution target now. H200 should expose
  memory-capacity and HBM-bandwidth effects while keeping the Hopper programming
  model close to H100.
- **B200/GB200 / Blackwell:** forward target. FP4 and new scale formats should
  shape design choices even before local Blackwell access is stable.

## Operating rules

Every lane artifact must include the backend, shape, dtype, GPU, driver/CUDA,
source commit, correctness tolerance, baseline, and losing cases. A fast number
without provenance does not count.

Prefer small upstreamable artifacts over private hero kernels: repro scripts,
benchmark harness fixes, correctness tests, docs, NCU notes, and narrowly scoped
kernel patches. The signature should be consistent: trusted cross-architecture
evidence for quantized inference GEMMs.

## Non-goals

- Do not make attention kernels the main lane unless they directly support a
  quantized GEMM contribution.
- Do not chase every kernel DSL equally; Triton, CUTLASS/CuTe, vLLM, ORT, and
  PyTorch/Inductor get priority because they connect to production inference.
- Do not treat A100-only INT4 work as the end state. It is a baseline and a way
  to learn packing, not the 2026 center of gravity.
