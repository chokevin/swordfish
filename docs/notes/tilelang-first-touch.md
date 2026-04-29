# TileLang first touch candidate

## Candidate

TileLang is a better near-term sidecar than Pallas because an INT4 GEMM example
maps directly to the lane's INT4 weight-only baseline. The first artifact should
be a narrow reproduction/benchmark note or docs PR around packed INT4 GEMM
correctness and provenance.

## Why contribute

The lane uses INT4 as the comparison baseline for FP8/FP4 work. A TileLang INT4
GEMM reproduction can be useful without becoming the main stack if it records:

- packing assumptions,
- reference correctness,
- exact shape and dtype,
- generated CUDA/source caveats,
- GPU and toolchain provenance.

## First comparison

Use a single fixed shape first:

```text
packed INT4 GEMM, 1024 x 1024 x 1024, compared against a torch reference
```

If the example uses int32 accumulation or a packed layout, document that layout
explicitly rather than hiding it behind the benchmark label.

## Draft contribution shape

Title:

> INT4 GEMM example reproduction with correctness and environment metadata

Artifact contents:

1. TileLang command/script and version.
2. Generated CUDA/source note if available.
3. Reference path and max error.
4. Latency samples, not just a single best number.
5. GPU, CUDA, driver, and source provenance.

## Caveats

TileLang should remain a contribution sidecar until it has a clear FP8/FP4 path
or a maintainer asks for the kind of H100/H200 evidence this lab can provide.
