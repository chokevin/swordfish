"""CuTe/CUTLASS extension spike.

The Python runner can select this backend today, but the native extension is an
optional Linux/CUDA build artifact. If it is not built, the backend raises an
explicit environment error instead of silently falling back to torch.
"""

from swordfish.kernels.cute.extension import BUILD_COMMAND, cutlass_matmul

__all__ = ["BUILD_COMMAND", "cutlass_matmul"]
