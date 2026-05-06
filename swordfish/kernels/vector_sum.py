"""Vector-sum reduction kernels."""

from __future__ import annotations

import torch

DEFAULT_BLOCK_SIZE = 8192
DEFAULT_NUM_STAGES = 1
DEFAULT_FINAL_NUM_WARPS = 16

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA hosts in integration runs.
    triton = None
    tl = None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _require_triton() -> None:
    if triton is None or tl is None or _partial_sum_kernel is None or _final_sum_kernel is None:
        raise RuntimeError("vectorsum_v2 Triton backend requires the triton package")


def torch_vector_sum_reference(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Reference reduction: fp64 accumulation, returned as fp32 like the target task."""
    result = x.to(torch.float64).sum().to(torch.float32)
    if out is None:
        return result
    out.reshape(-1)[0].copy_(result)
    return out


if triton is not None and tl is not None:

    @triton.jit
    def _partial_sum_kernel(x_ptr, partials_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        partial = tl.sum(values, axis=0)
        tl.store(partials_ptr + pid, partial)

    @triton.jit
    def _final_sum_kernel(partials_ptr, out_ptr, n_partials: int, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_partials
        values = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        total = tl.sum(values, axis=0)
        tl.store(out_ptr, total)

else:
    _partial_sum_kernel = None
    _final_sum_kernel = None


def partial_count(n_elements: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    if n_elements <= 0:
        raise ValueError("n_elements must be positive")
    if not _is_power_of_two(block_size):
        raise ValueError("block_size must be a positive power of two")
    return (n_elements + block_size - 1) // block_size


def triton_vector_sum(
    x: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Reduce a contiguous 1D CUDA tensor into an fp32 scalar using two Triton kernels."""
    _require_triton()
    if x.device.type != "cuda" or out.device.type != "cuda" or partials.device.type != "cuda":
        raise RuntimeError("vectorsum_v2 Triton backend requires CUDA tensors")
    if x.ndim != 1:
        raise ValueError("vectorsum_v2 input must be a 1D tensor")
    if not x.is_contiguous():
        raise ValueError("vectorsum_v2 Triton backend requires a contiguous input tensor")
    if out.numel() != 1 or out.dtype != torch.float32:
        raise ValueError("vectorsum_v2 output must be a single fp32 scalar tensor")

    n_elements = x.numel()
    n_partials = partial_count(n_elements, block_size)
    if partials.numel() < n_partials or partials.dtype != torch.float32:
        raise ValueError("vectorsum_v2 partials must have at least partial_count fp32 elements")

    final_block_size = 1 << (n_partials - 1).bit_length()
    _partial_sum_kernel[(n_partials,)](
        x,
        partials,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=DEFAULT_NUM_STAGES,
    )
    _final_sum_kernel[(1,)](
        partials,
        out,
        n_partials,
        BLOCK_SIZE=final_block_size,
        num_warps=DEFAULT_FINAL_NUM_WARPS,
    )
    return out
