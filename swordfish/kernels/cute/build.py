"""Environment probe for the future CuTe/CUTLASS extension build."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


def _has_cutlass_headers(cutlass_dir: Path) -> bool:
    return (cutlass_dir / "include" / "cutlass" / "cutlass.h").exists() and (
        cutlass_dir / "include" / "cute" / "tensor.hpp"
    ).exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="probe CuTe/CUTLASS extension build environment")
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if platform.system() != "Linux":
        raise SystemExit("CuTe/CUTLASS extension builds require a Linux CUDA host")
    if CUDA_HOME is None:
        raise SystemExit("CUDA_HOME is not set; install CUDA toolkit or use an NVIDIA CUDA image")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is not available in this environment")
    if not _has_cutlass_headers(args.cutlass_dir):
        raise SystemExit(
            f"{args.cutlass_dir} does not look like a CUTLASS checkout; expected "
            "include/cutlass/cutlass.h and include/cute/tensor.hpp"
        )

    raise SystemExit(
        "CuTe/CUTLASS environment probe passed, but the native GEMM source is not "
        "implemented yet. Add the extension sources here before expecting "
        "`--backend cutlass` to run."
    )


if __name__ == "__main__":
    raise SystemExit(main())
