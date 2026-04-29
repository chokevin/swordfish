# Triton first repro candidate

## Candidate

Open a Triton issue, or a small documentation PR if maintainers prefer, around a
minimal contiguous GEMM repro that reports correctness and benchmark provenance
using the `swordfish` result protocol.

This should not be framed as a performance claim yet. The current Triton backend
is intentionally educational: one program computes one output tile, accumulates
with `tl.dot` in fp32, and writes to a contiguous row-major output tensor.

## Why it is useful

Triton maintainers get the most value from small, reproducible reports. This
candidate has:

- one kernel,
- one fixed shape,
- one backend,
- a torch reference check,
- a JSON result with GPU, CUDA, driver, Triton version, shape, dtype,
  correctness, and latency.

That makes it suitable for a docs/example contribution or a clean issue if a
compiler/runtime behavior looks surprising.

## Evidence command

Run this on a CUDA node:

```bash
python -m swordfish.runner run-gemm \
  --backend triton \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --device cuda \
  --repeats 5 --warmup 10 --iters 50 \
  --out /tmp/triton-gemm.json
```

Attach the full JSON and summarize:

- `config.scope`, `config.backend`, `config.shape`, `config.dtype`
- `env.gpu_name`, `env.gpu_class`, `env.gpu_cc`
- `env.torch`, `env.torch_cuda`, `env.cuda_driver`, `env.nvidia_driver`,
  `env.triton`
- `correctness.matches_reference`, `max_abs_error`, `max_rel_error`
- latency samples and summary stats

## Draft issue shape

Title:

> Minimal `tl.dot` GEMM repro with protocol JSON on H100

Body outline:

1. Describe the kernel: contiguous row-major fp16 GEMM, 32x32x32 tiles, fp32
   accumulation, torch reference check.
2. Include the exact command and result JSON.
3. State expected behavior: finite output, reference match within tolerance, no
   compile/runtime failure.
4. State actual behavior. If the result is normal, convert this into a docs PR
   or example note rather than filing a bug.

## Caveats

The current kernel only supports CUDA tensors, 2D inputs, and contiguous
row-major layout. It is deliberately not tuned for cuBLAS parity.
