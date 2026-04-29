# JAX/Pallas first touch decision

Decision: skip Pallas for the next-week public contribution target unless it can
be reframed as a quantized GEMM repro with exact provenance.

## Why skip now

The active lane is quantized inference GEMM: FP8 first, FP4 next, INT4
weight-only as the comparison baseline. The useful near-term artifacts should
therefore reinforce quantized GEMM correctness, scale/layout behavior,
cross-architecture measurement, or serving integration.

The current local `swordfish` kernel surface is a simple Triton GEMM and a
PyTorch GPT reference. That is enough to compare basic matmul behavior, but not
yet enough to produce a strong Pallas quantized-GEMM contribution.

## No-go note

> Skip Pallas this week unless we can produce a quantized GEMM repro with
> shape/dtype/GPU/provenance, correctness against a reference path, and a clear
> maintainer-useful question. A generic fp16 matmul comparison would be a
> sidecar learning exercise, not the main lane.

## Revisit criteria

Reopen this when at least one of these is true:

- there is a Pallas FP8/FP4/INT4 GEMM example to compare against the same
  `swordfish` shape,
- a Pallas issue needs H100/H200 reproduction data that this lab can provide,
- or the local benchmark protocol can emit a JAX/Pallas result with the same
  correctness and provenance fields as the PyTorch/Triton paths.

## First shape if revived

Use one fixed GEMM shape first, such as 1024x1024x1024 or 4096x4096x4096, and
compare against the relevant JAX baseline plus the `swordfish` torch reference.
