"""Numerically-faithful reference INT4 dequant + matmul in pure PyTorch.

This is the ground truth for correctness tests. It is slow — do not use it
in any production or benchmark path.
"""

from __future__ import annotations

import torch


def dequantize_int4(
    packed: torch.Tensor,  # [K, N // 2] uint8, two int4 per byte along N
    scales: torch.Tensor,  # [K // group_size, N] fp16
    zeros: torch.Tensor | None = None,  # [K // group_size, N] fp16 or int
    group_size: int = 128,
) -> torch.Tensor:
    """Unpack INT4 weights and return FP16 matrix of shape [K, N]."""
    K, N_half = packed.shape
    N = N_half * 2
    assert scales.shape == (K // group_size, N), (
        f"scales shape {scales.shape} != expected {(K // group_size, N)}"
    )

    low = (packed & 0x0F).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    # interleave along N dim: [K, N_half, 2] -> [K, N]
    unpacked = torch.stack([low, high], dim=-1).reshape(K, N)

    # broadcast scales/zeros across each group
    scales_exp = scales.repeat_interleave(group_size, dim=0)  # [K, N]
    if zeros is not None:
        zeros_exp = zeros.repeat_interleave(group_size, dim=0).to(unpacked.dtype)
        w_fp = (unpacked - zeros_exp).to(scales.dtype) * scales_exp
    else:
        # symmetric: int4 is signed in [-8, 7], center around 0
        centered = unpacked - 8
        w_fp = centered.to(scales.dtype) * scales_exp
    return w_fp  # [K, N] fp16


def reference_w4a16_matmul(
    a: torch.Tensor,  # [M, K] fp16
    packed: torch.Tensor,  # [K, N // 2] uint8
    scales: torch.Tensor,  # [K // group_size, N] fp16
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
    bias: torch.Tensor | None = None,  # [N] fp16
) -> torch.Tensor:
    """Reference dequant-then-matmul. Returns [M, N] fp16."""
    w = dequantize_int4(packed, scales, zeros, group_size=group_size)
    out = a.to(torch.float32) @ w.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.float16)
