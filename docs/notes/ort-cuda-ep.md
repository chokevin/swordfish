# ONNX Runtime CUDA EP quant-kernel map

## Recommended first contribution

Start with `MatMulNBits` correctness/benchmark work in the CUDA Execution
Provider.

Surface:

```text
onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc
```

Why this first:

- It is already a quantized GEMM entrypoint.
- It has fast-path, prepack, and fallback behavior in one operator surface.
- It fits the `swordfish` lane better than a broad attention project: quantized
  GEMM first, benchmarkable shape, correctness against a reference path.

## Candidate 1: MatMulNBits fast-path/prepack repro

Contribution type: issue or targeted PR.

Question to answer:

> Which FP16/BF16 activation plus 4-bit/8-bit weight shapes miss the CUDA fast
> path or behave differently between prepacked and non-prepacked execution?

Evidence to collect:

- operator shape, dtype, bit width, group size, and prepack setting,
- CUDA EP path taken, if observable,
- output correctness against a dequant/reference path,
- latency with the common `swordfish` result fields: GPU, driver, CUDA, commit,
  shape, dtype, backend, correctness, and latency samples.

Useful issue title:

> CUDA EP `MatMulNBits` prepack vs non-prepack correctness/latency repro

Reviewability rule: isolate one path. Do not mix fast-path cleanup, fallback
behavior, and new dtype support in one PR.

## Candidate 2: FP8 GEMM path discovery

Contribution type: map first, implementation later.

Question to answer:

> Does ORT or ORT GenAI currently expose a Hopper FP8 GEMM path comparable to
> CUTLASS/DeepGEMM/TensorRT-LLM, and if not, which operator surface should own
> the first small FP8 contribution?

Potential surfaces:

```text
onnxruntime/contrib_ops/cuda/quantization/
onnxruntime/contrib_ops/cuda/llm/
```

This is aligned with the lane, but it is higher risk than `MatMulNBits` because
the first exploration pass did not confirm an existing FP8 kernel file. Treat
this as a mapping task before promising a kernel PR.

## Candidate 3: quantized attention cache path

Contribution type: no-go or narrow issue unless a dedicated paged-attention
kernel is confirmed.

Potential surfaces:

```text
onnxruntime/contrib_ops/cuda/quantization/attention_quantization.cc
onnxruntime/contrib_ops/cuda/transformers/generation_cuda_impl.cu
```

The likely opportunity is quantized-attention or cache-indirection behavior, not
necessarily true paged attention. Do not call this a paged-attention contribution
until an explicit ORT/ORT GenAI paged-attention surface is identified.

## Next local work

1. Build a minimal ORT repro for `MatMulNBits` with one FP16 x INT4 shape.
2. Run prepacked and non-prepacked paths if both are accessible.
3. Record correctness and latency using the `swordfish` benchmark protocol.
4. Only then decide whether the upstream artifact is an issue, a docs note, or a
   small CUDA EP test/benchmark PR.
