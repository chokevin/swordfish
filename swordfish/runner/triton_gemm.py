"""Small educational Triton GEMM backend.

This is intentionally simple: one program computes one C tile, accumulates in
fp32, and writes back to the preallocated output tensor. It is a correctness and
plumbing backend, not a tuned cuBLAS replacement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, k, block_k):
        k_idxs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * k + k_idxs[None, :],
            mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_idxs[:, None] * n + offs_n[None, :],
            mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="tf32")

    tl.store(
        c_ptr + offs_m[:, None] * n + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if a.device.type != "cuda" or b.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("triton_matmul requires CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RuntimeError("triton_matmul expects contiguous row-major tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("triton_matmul expects 2D tensors")
    m, k = a.shape
    k_b, n = b.shape
    if k != k_b or out.shape != (m, n):
        raise ValueError(
            f"shape mismatch: a={tuple(a.shape)} b={tuple(b.shape)} out={tuple(out.shape)}"
        )

    block_m = 32
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=4,
    )
    return out
