# CUTLASS/CuTe first touch candidate

## Candidate

Start with a low-risk docs/example/repro contribution, not a kernel PR: document
how to probe a Hopper FP8/CuTe build environment and how to attach benchmark
provenance when comparing against the `swordfish` result protocol.

## Local probe command

On a Linux CUDA host with a CUTLASS checkout:

```bash
python -m swordfish.kernels.cute.build --cutlass-dir "$CUTLASS_DIR"
```

The current expected result is explicit: the environment probe can pass, but the
native GEMM source is not implemented yet. That is useful because it separates
"the machine can build CUTLASS/CuTe" from "the swordfish extension exists."

## Why it is low risk

- It is documentation/repro work first.
- It aligns with the lane decision: FP8 first, Hopper/H200 primary, Blackwell
  forward-looking.
- It does not ask CUTLASS maintainers to review a new kernel before there is
  local build evidence.

## Draft contribution shape

Title:

> Hopper FP8 CuTe GEMM repro notes with environment probe and benchmark provenance

The artifact should include:

1. CUTLASS commit and build command.
2. CUDA toolkit, driver, GPU, and PyTorch versions.
3. The exact GEMM shape and dtype.
4. A note that `swordfish --backend cutlass` intentionally fails until the
   native extension source exists.
5. Follow-up path from probe -> minimal extension -> benchmark JSON.

## Blockers

This needs Linux, CUDA toolkit headers, a valid CUTLASS checkout, and a CUDA
visible PyTorch environment. The local macOS development machine cannot validate
the actual build.
