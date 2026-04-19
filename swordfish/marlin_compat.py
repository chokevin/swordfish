"""Marlin layout adapter.

Converts swordfish's ``(packed, scales)`` pair into Marlin's expected weight
layout, and exposes a thin ``marlin_matmul`` wrapper around ``marlin.mul``.

We *delegate* the permutation to Marlin's own packing utility rather than
reverse-engineering the permutation table — Marlin's ``Layer.pack`` is
authoritative, and re-quantizing the dequantized fp16 yields bit-identical
INT4 values for symmetric quant (it's deterministic absmax → divide → round).

Marlin is not importable on macOS; this module is therefore split into:
  - ``to_marlin_layout`` / ``marlin_matmul``: GPU-only, raise on missing dep.
  - ``dequantized_for_marlin``: CPU-runnable; what the GPU path feeds Marlin.
The CPU test in ``tests/test_marlin_compat.py`` exercises the dequant path,
which is what would fail silently if ``swordfish.pack`` ever changed format.
"""

from __future__ import annotations

import torch

from swordfish.reference import dequantize_int4


def dequantized_for_marlin(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Returns the fp16 [N, K] tensor in nn.Linear-weight layout that Marlin's
    packer expects. CPU-runnable. This is what we hand to Marlin on the A100."""
    w_fp = dequantize_int4(packed, scales, group_size=group_size)  # [K, N]
    return w_fp.t().contiguous()  # [N, K]


def to_marlin_layout(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (swordfish-packed, scales) -> (marlin_B, marlin_s).

    Strategy: dequantize → build an nn.Linear → call ``marlin.Layer.pack``,
    which performs Marlin's own quantize + permute. This is the only way to
    guarantee layout compatibility without re-implementing Marlin's
    permutation tables.

    Requires Marlin and a CUDA device. Raises ImportError on Mac / non-CUDA.
    """
    if packed.device.type != "cuda":
        raise RuntimeError("to_marlin_layout requires CUDA tensors")

    try:
        import marlin  # type: ignore
    except ImportError as e:
        raise ImportError("marlin not installed. See docs/profiling/RUN_ME_ON_A100.md.") from e

    K, N_half = packed.shape
    N = N_half * 2

    w_fp_NK = dequantized_for_marlin(packed, scales, group_size=group_size)  # [N, K]

    linear = torch.nn.Linear(K, N, bias=False).to(device=packed.device, dtype=torch.float16)
    linear.weight.data.copy_(w_fp_NK)

    layer = marlin.Layer(infeatures=K, outfeatures=N, groupsize=group_size)
    layer = layer.to(packed.device)
    # Marlin's Layer.pack(linear, scales) — scales shape is [K//group, N]
    layer.pack(linear, scales)
    return layer.B, layer.s


def marlin_matmul(
    a: torch.Tensor,  # [M, K] fp16
    B: torch.Tensor,  # marlin-packed weight from to_marlin_layout
    s: torch.Tensor,  # marlin scales from to_marlin_layout
    group_size: int,
) -> torch.Tensor:
    """Thin wrapper around ``marlin.mul``. Returns [M, N] fp16."""
    try:
        import marlin  # type: ignore
    except ImportError as e:
        raise ImportError("marlin not installed.") from e

    M, K = a.shape
    N = s.shape[1]
    C = torch.empty(M, N, device=a.device, dtype=torch.float16)
    # marlin requires a small int32 workspace buffer; size is per-N-tile.
    # 16 ints per 128-wide N-tile is the upstream convention.
    workspace = torch.zeros(N // 128 * 16, device=a.device, dtype=torch.int32)
    marlin.mul(a, B, C, s, workspace)
    return C
