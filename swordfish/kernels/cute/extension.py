"""Loader for the optional CuTe/CUTLASS GEMM extension."""

from __future__ import annotations

import torch


BUILD_COMMAND = "python -m swordfish.kernels.cute.build --cutlass-dir $CUTLASS_DIR"


def _extension_unavailable_error() -> RuntimeError:
    return RuntimeError(
        "CuTe/CUTLASS extension is not built. Build it on a Linux CUDA host with "
        f"`{BUILD_COMMAND}` after setting CUTLASS_DIR to a CUTLASS checkout. "
        "This backend does not fall back to torch because benchmark labels must "
        "match the kernel that actually ran."
    )


def cutlass_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    try:
        from swordfish_cutlass_gemm import matmul as extension_matmul
    except ModuleNotFoundError as exc:
        raise _extension_unavailable_error() from exc

    return extension_matmul(a, b, out)
