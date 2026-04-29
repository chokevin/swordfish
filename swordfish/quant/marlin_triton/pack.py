"""Small signed-INT4 packing helpers for Marlin-style W4A16 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedInt4Weight:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    group_size: int


def _check_signed_int4(values: torch.Tensor) -> None:
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"expected integer tensor, got {values.dtype}")
    if values.numel() and (values.min().item() < -8 or values.max().item() > 7):
        raise ValueError("signed INT4 values must be in [-8, 7]")


def pack_int4_signed(values: torch.Tensor) -> torch.Tensor:
    """Pack two signed INT4 values into each byte, low nibble first."""
    if values.ndim != 2:
        raise ValueError("pack_int4_signed expects a 2D [K, N] tensor")
    _check_signed_int4(values)

    k, n = values.shape
    if n % 2:
        pad = torch.zeros((k, 1), dtype=values.dtype, device=values.device)
        values = torch.cat([values, pad], dim=1)

    encoded = values.to(torch.int16) & 0x0F
    low = encoded[:, 0::2]
    high = encoded[:, 1::2] << 4
    return (low | high).to(torch.uint8)


def unpack_int4_signed(packed: torch.Tensor, *, n: int) -> torch.Tensor:
    """Unpack low-nibble-first signed INT4 bytes into a [K, N] int8 tensor."""
    if packed.ndim != 2:
        raise ValueError("unpack_int4_signed expects a 2D [K, ceil(N/2)] tensor")
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected uint8 packed tensor, got {packed.dtype}")
    if n <= 0:
        raise ValueError("n must be positive")
    if packed.shape[1] != (n + 1) // 2:
        raise ValueError(f"packed width {packed.shape[1]} does not match n={n}")

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).flatten(start_dim=1)[:, :n].to(torch.int8)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles)


def quantize_weight_int4_per_group(weight: torch.Tensor, *, group_size: int) -> QuantizedInt4Weight:
    """Symmetric per-column, per-K-group INT4 quantization for a [K, N] weight."""
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D [K, N] tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    k, n = weight.shape
    groups = (k + group_size - 1) // group_size
    scales = torch.empty((groups, n), dtype=torch.float32, device=weight.device)
    quantized = torch.empty((k, n), dtype=torch.int8, device=weight.device)

    weight_fp32 = weight.float()
    for group in range(groups):
        start = group * group_size
        end = min(start + group_size, k)
        chunk = weight_fp32[start:end]
        scale = chunk.abs().amax(dim=0) / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        scales[group] = scale
        quantized[start:end] = torch.round(chunk / scale).clamp(-8, 7).to(torch.int8)

    return QuantizedInt4Weight(
        packed=pack_int4_signed(quantized),
        scales=scales,
        shape=(k, n),
        group_size=group_size,
    )


def dequantize_weight_int4(weight: QuantizedInt4Weight) -> torch.Tensor:
    """Materialize a packed signed-INT4 weight back to fp32 [K, N]."""
    k, n = weight.shape
    values = unpack_int4_signed(weight.packed, n=n).float()
    out = torch.empty((k, n), dtype=torch.float32, device=weight.packed.device)
    for group in range(weight.scales.shape[0]):
        start = group * weight.group_size
        end = min(start + weight.group_size, k)
        out[start:end] = values[start:end] * weight.scales[group]
    return out


def reference_w4a16_matmul(a: torch.Tensor, weight: QuantizedInt4Weight) -> torch.Tensor:
    """Correctness oracle for A(fp16/bf16/fp32) @ dequantized W(int4)."""
    k, _ = weight.shape
    if a.ndim != 2:
        raise ValueError("a must be a 2D [M, K] tensor")
    if a.shape[1] != k:
        raise ValueError(f"a has K={a.shape[1]}, but weight has K={k}")
    b = dequantize_weight_int4(weight).to(device=a.device, dtype=a.dtype)
    return a @ b
