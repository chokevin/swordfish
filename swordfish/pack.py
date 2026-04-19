"""Quantize FP16 weights to INT4 and pack them.

This is for TESTING — the real path consumes pre-quantized checkpoints
(AWQ/GPTQ). We include this so correctness tests can generate synthetic
weights without needing a model download.
"""

from __future__ import annotations

import torch


def quantize_symmetric_int4(
    w: torch.Tensor,  # [K, N] fp16
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group INT4 quantization.

    Returns:
        packed: [K, N // 2] uint8 with two int4 values per byte (low=even, high=odd)
        scales: [K // group_size, N] fp16
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert N % 2 == 0, f"N={N} must be even for INT4 packing"

    # reshape to [K//group_size, group_size, N]
    wg = w.reshape(K // group_size, group_size, N)

    # per-group absmax -> scale so that max abs maps to 7 (int4 symmetric signed)
    absmax = wg.abs().amax(dim=1)  # [K//group_size, N]
    scales = (absmax / 7.0).clamp(min=1e-6).to(torch.float16)

    # quantize: divide by scale, round, clamp to [-8, 7], shift by +8 to unsigned [0,15]
    scales_exp = scales.unsqueeze(1)
    q = torch.round(wg / scales_exp).clamp(-8, 7).to(torch.int8)
    q = (q + 8).to(torch.uint8)  # [K//group_size, group_size, N] in [0,15]
    q = q.reshape(K, N)

    # pack along N: even cols in low nibble, odd cols in high nibble
    low = q[:, 0::2]  # [K, N//2]
    high = q[:, 1::2]  # [K, N//2]
    packed = (low | (high << 4)).to(torch.uint8)

    return packed, scales


def random_quantized_weights(
    K: int,
    N: int,
    group_size: int = 128,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture: random FP16 weights, quantized and packed. Returns (packed, scales)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(K, N, device=device, dtype=torch.float16, generator=gen) * 0.02
    return quantize_symmetric_int4(w, group_size=group_size)
