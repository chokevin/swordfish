"""Triton INT4 x FP16 decode matmul — placeholder skeleton.

Week 2 work: fill in the actual kernel. This file exists so the import
surface is stable and benchmarks can reference it.
"""

from __future__ import annotations

import torch


def triton_w4a16_matmul(
    a: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Not yet implemented. Fill in during week 2.

    Target signature matches reference_w4a16_matmul.
    """
    raise NotImplementedError(
        "triton_w4a16_matmul: skeleton only — implement in week 2 of the roadmap"
    )
