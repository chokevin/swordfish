"""Educational Triton W4A16 kernel slot.

The Python packing/reference path is the correctness oracle. This module is the
CUDA-only implementation hook for reproducing a Marlin-style inner loop without
making local CPU tests import Triton.
"""

from __future__ import annotations

import torch

from swordfish.quant.marlin_triton.pack import QuantizedInt4Weight

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _w4a16_kernel(
        a_ptr,
        packed_ptr,
        scales_ptr,
        out_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        packed_n: tl.constexpr,
        group_size: tl.constexpr,
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
            group_ids = k_idxs // group_size
            packed_cols = offs_n // 2
            packed = tl.load(
                packed_ptr + k_idxs[:, None] * packed_n + packed_cols[None, :],
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0,
            )
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            nibble = tl.where((offs_n[None, :] % 2) == 0, low, high).to(tl.int32)
            signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
            scale = tl.load(
                scales_ptr + group_ids[:, None] * n + offs_n[None, :],
                mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            b = (signed * scale).to(tl.float16)
            a = tl.load(
                a_ptr + offs_m[:, None] * k + k_idxs[None, :],
                mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
                other=0.0,
            )
            acc += tl.dot(a, b, input_precision="tf32")

        tl.store(
            out_ptr + offs_m[:, None] * n + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )
else:
    _w4a16_kernel = None


def triton_w4a16_matmul(a: torch.Tensor, weight: QuantizedInt4Weight) -> torch.Tensor:
    if a.device.type != "cuda":
        raise RuntimeError("triton_w4a16_matmul requires CUDA tensors")
    if weight.packed.device.type != "cuda" or weight.scales.device.type != "cuda":
        raise RuntimeError("packed weight and scales must be CUDA tensors")
    if a.dtype != torch.float16:
        raise RuntimeError("triton_w4a16_matmul currently supports fp16 activations only")
    if a.ndim != 2:
        raise ValueError("a must be a 2D [M, K] tensor")

    if triton is None or _w4a16_kernel is None:
        raise RuntimeError("triton_w4a16_matmul requires the triton package")

    k, n = weight.shape
    if a.shape[1] != k:
        raise ValueError(f"a has K={a.shape[1]}, but weight has K={k}")

    out = torch.empty((a.shape[0], n), device=a.device, dtype=a.dtype)
    block_m = 16
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(a.shape[0], block_m) * triton.cdiv(n, block_n),)
    _w4a16_kernel[grid](
        a,
        weight.packed,
        weight.scales,
        out,
        a.shape[0],
        n,
        k,
        weight.packed.shape[1],
        weight.group_size,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=4,
    )
    return out
