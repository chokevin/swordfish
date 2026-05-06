"""Benchmark harness for the vectorsum_v2 reduction target."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from swordfish.kernels.vector_sum import DEFAULT_BLOCK_SIZE, partial_count
from swordfish.runner.backends import TORCH_DTYPES
from swordfish.runner.schema import (
    DTYPE_BYTES,
    SCHEMA_VERSION,
    latency_stats,
    pct_of_peak,
    peak_for,
    tbps_from_ms,
)
from swordfish.runner.torch_gemm import _resolve_device, _time_cpu, _time_cuda, capture_env

VECTOR_SUM_BENCHMARK_SIZES = (
    1_638_400,
    3_276_800,
    6_553_600,
    13_107_200,
    26_214_400,
    52_428_800,
)

VectorSumRunner = Callable[["VectorSumState"], torch.Tensor]


@dataclass(frozen=True)
class VectorSumState:
    x: torch.Tensor
    out: torch.Tensor
    partials: torch.Tensor | None
    runner: VectorSumRunner
    block_size: int


def make_vector_sum_input(
    *,
    size: int,
    dtype: str,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Generate the target input: normal fp32 data with deterministic scale+offset."""
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)
    data = torch.randn(
        (size,), device=device, dtype=TORCH_DTYPES[dtype], generator=gen
    ).contiguous()

    offset_gen = torch.Generator(device=device.type)
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device=device.type)
    scale_gen.manual_seed(seed + 2)

    offset = (torch.rand(1, device=device, generator=offset_gen) * 200 - 100).item()
    scale = (torch.rand(1, device=device, generator=scale_gen) * 9.9 + 0.1).item()
    return (data * scale + offset).contiguous()


def available_vector_sum_backends() -> tuple[str, ...]:
    return ("torch", "triton")


def _prepare_torch(
    *,
    size: int,
    dtype: str,
    device: torch.device,
    seed: int,
    block_size: int,
) -> VectorSumState:
    from swordfish.kernels.vector_sum import torch_vector_sum_reference

    x = make_vector_sum_input(size=size, dtype=dtype, device=device, seed=seed)
    out = torch.empty((1,), device=device, dtype=torch.float32)

    def run(state: VectorSumState) -> torch.Tensor:
        return torch_vector_sum_reference(state.x, state.out)

    return VectorSumState(x=x, out=out, partials=None, runner=run, block_size=block_size)


def _prepare_triton(
    *,
    size: int,
    dtype: str,
    device: torch.device,
    seed: int,
    block_size: int,
) -> VectorSumState:
    if device.type != "cuda":
        raise RuntimeError("vectorsum_v2 Triton backend requires a CUDA device")
    from swordfish.kernels.vector_sum import triton_vector_sum

    x = make_vector_sum_input(size=size, dtype=dtype, device=device, seed=seed)
    out = torch.empty((1,), device=device, dtype=torch.float32)
    partials = torch.empty((partial_count(size, block_size),), device=device, dtype=torch.float32)

    def run(state: VectorSumState) -> torch.Tensor:
        if state.partials is None:
            raise RuntimeError("vectorsum_v2 Triton state is missing partials")
        return triton_vector_sum(
            state.x,
            state.out,
            state.partials,
            block_size=state.block_size,
        )

    return VectorSumState(x=x, out=out, partials=partials, runner=run, block_size=block_size)


def _prepare_state(
    *,
    backend: str,
    size: int,
    dtype: str,
    device: torch.device,
    seed: int,
    block_size: int,
) -> VectorSumState:
    if backend == "torch":
        return _prepare_torch(
            size=size,
            dtype=dtype,
            device=device,
            seed=seed,
            block_size=block_size,
        )
    if backend == "triton":
        return _prepare_triton(
            size=size,
            dtype=dtype,
            device=device,
            seed=seed,
            block_size=block_size,
        )
    raise ValueError(
        f"unknown vectorsum_v2 backend {backend!r}; expected one of {available_vector_sum_backends()}"
    )


def _reference_check(state: VectorSumState, *, size: int, dtype: str) -> dict[str, Any]:
    from swordfish.kernels.vector_sum import torch_vector_sum_reference

    expected = torch_vector_sum_reference(state.x).reshape(-1)[0].detach()
    actual = state.out.reshape(-1)[0].detach()
    max_abs_error = float(torch.abs(actual - expected).item())

    # Floating-point reductions are order-dependent. The benchmark input is
    # N(0, 1), so expected numerical noise grows roughly with sqrt(N).
    dtype_scale = 2.0 if dtype in {"fp16", "bf16"} else 1.0
    atol = dtype_scale * max(1e-5, 1e-3 * math.sqrt(size))
    rtol = 1e-5 if dtype == "fp32" else 1e-3
    matches = torch.allclose(actual, expected, atol=atol, rtol=rtol)

    return {
        "reference_backend": "torch",
        "matches_reference": matches,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
        "output_shape": list(state.out.shape),
        "output_fp32": float(actual.item()),
        "reference_fp32": float(expected.item()),
    }


def run_vector_sum_benchmark(
    *,
    size: int,
    dtype: str = "fp32",
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    backend: str = "torch",
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if backend not in available_vector_sum_backends():
        raise ValueError(
            f"unknown vectorsum_v2 backend {backend!r}; expected one of {available_vector_sum_backends()}"
        )
    if min(size, repeats, iters, block_size) <= 0 or warmup < 0:
        raise ValueError(
            "size, repeats, iters, and block_size must be positive; warmup must be non-negative"
        )
    partial_count(size, block_size)

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    state = _prepare_state(
        backend=backend,
        size=size,
        dtype=dtype,
        device=device,
        seed=seed,
        block_size=block_size,
    )

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    samples_ms = [
        timer(lambda: state.runner(state), warmup=warmup, iters=iters) for _ in range(repeats)
    ]
    stats = latency_stats(samples_ms)

    state.runner(state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    finite = bool(torch.isfinite(state.out).all().item())
    reference = _reference_check(state, size=size, dtype=dtype)
    env = capture_env(device, arch_label=arch_label)
    gpu_class = env["gpu_class"]

    input_bytes = size * DTYPE_BYTES[dtype]
    output_bytes = 4
    partials_bytes = 0
    if backend == "triton":
        partials_bytes = partial_count(size, block_size) * 4
    estimated_bytes = input_bytes + output_bytes + 2 * partials_bytes
    mean_ms = stats["mean_ms"]
    bandwidth_tbps = tbps_from_ms(estimated_bytes, mean_ms)
    hbm_peak = peak_for(gpu_class, dtype, "hbm_tbps")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "vectorsum_v2",
        "config": {
            "scope": "vector_sum",
            "backend": backend,
            "shape": {"size": size},
            "size": size,
            "dtype": dtype,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "block_size": block_size,
        },
        "env": env,
        "correctness": {
            "finite_output": finite,
            **reference,
        },
        "metrics": {
            "latency": stats,
            "elements": size,
            "elements_per_second": size / (mean_ms / 1000.0)
            if mean_ms > 0 and not math.isnan(mean_ms)
            else float("nan"),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "partials_bytes": partials_bytes,
            "estimated_bytes": estimated_bytes,
            "estimated_bandwidth_tbps": bandwidth_tbps,
            "hbm_peak_tbps": hbm_peak,
            "estimated_hbm_sol_pct": pct_of_peak(bandwidth_tbps, hbm_peak),
        },
    }
