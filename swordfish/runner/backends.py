"""Pluggable GEMM backends for the runner.

The runner owns timing, environment capture, and result JSON. A backend owns
only the actual C = A @ B operation so torch/cuBLAS, Triton, CuTe, or raw PTX can
be swapped without changing the benchmark contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


TORCH_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

TensorRunner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GemmState:
    a: torch.Tensor
    b: torch.Tensor
    out: torch.Tensor
    runner: TensorRunner | None = None


class GemmBackend(Protocol):
    name: str

    def prepare(
        self,
        *,
        m: int,
        n: int,
        k: int,
        dtype: str,
        device: torch.device,
        seed: int,
    ) -> GemmState:
        """Allocate backend inputs and output outside the timed region."""

    def run(self, state: GemmState) -> torch.Tensor:
        """Run one GEMM into ``state.out`` and return the output tensor."""


class TorchGemmBackend:
    name = "torch"

    def prepare(
        self,
        *,
        m: int,
        n: int,
        k: int,
        dtype: str,
        device: torch.device,
        seed: int,
    ) -> GemmState:
        torch.manual_seed(seed)
        torch_dtype = TORCH_DTYPES[dtype]
        a = torch.randn((m, k), device=device, dtype=torch_dtype)
        b = torch.randn((k, n), device=device, dtype=torch_dtype)
        out = torch.empty((m, n), device=device, dtype=torch_dtype)
        return GemmState(a=a, b=b, out=out)

    def run(self, state: GemmState) -> torch.Tensor:
        return torch.mm(state.a, state.b, out=state.out)


class TritonGemmBackend:
    name = "triton"

    def prepare(
        self,
        *,
        m: int,
        n: int,
        k: int,
        dtype: str,
        device: torch.device,
        seed: int,
    ) -> GemmState:
        if device.type != "cuda":
            raise RuntimeError("the triton GEMM backend requires a CUDA device")
        try:
            from swordfish.runner.triton_gemm import triton_matmul
        except ImportError as exc:
            raise RuntimeError("the triton GEMM backend requires the triton package") from exc

        torch.manual_seed(seed)
        torch_dtype = TORCH_DTYPES[dtype]
        a = torch.randn((m, k), device=device, dtype=torch_dtype)
        b = torch.randn((k, n), device=device, dtype=torch_dtype)
        out = torch.empty((m, n), device=device, dtype=torch_dtype)
        return GemmState(a=a, b=b, out=out, runner=triton_matmul)

    def run(self, state: GemmState) -> torch.Tensor:
        if state.runner is None:
            raise RuntimeError("triton backend state is missing its kernel runner")
        return state.runner(state.a, state.b, state.out)


class CutlassGemmBackend:
    name = "cutlass"

    def prepare(
        self,
        *,
        m: int,
        n: int,
        k: int,
        dtype: str,
        device: torch.device,
        seed: int,
    ) -> GemmState:
        if device.type != "cuda":
            raise RuntimeError("the CuTe/CUTLASS GEMM backend requires a CUDA device")
        from swordfish.kernels.cute import cutlass_matmul

        torch.manual_seed(seed)
        torch_dtype = TORCH_DTYPES[dtype]
        a = torch.randn((m, k), device=device, dtype=torch_dtype)
        b = torch.randn((k, n), device=device, dtype=torch_dtype)
        out = torch.empty((m, n), device=device, dtype=torch_dtype)
        return GemmState(a=a, b=b, out=out, runner=cutlass_matmul)

    def run(self, state: GemmState) -> torch.Tensor:
        if state.runner is None:
            raise RuntimeError("CuTe/CUTLASS backend state is missing its extension runner")
        return state.runner(state.a, state.b, state.out)


_BACKENDS: dict[str, type[GemmBackend]] = {
    CutlassGemmBackend.name: CutlassGemmBackend,
    TorchGemmBackend.name: TorchGemmBackend,
    TritonGemmBackend.name: TritonGemmBackend,
}


def available_gemm_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_gemm_backend(name: str) -> GemmBackend:
    try:
        backend_cls = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown GEMM backend {name!r}; expected one of {available_gemm_backends()}"
        ) from exc
    return backend_cls()
