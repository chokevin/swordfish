"""Benchmark harness for the Marlin-style W4A16 reproduction artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from swordfish.quant.marlin_triton.pack import (
    quantize_weight_int4_per_group,
    reference_w4a16_matmul,
)
from swordfish.quant.marlin_triton.triton_kernel import triton_w4a16_matmul
from swordfish.runner.backends import TORCH_DTYPES
from swordfish.runner.schema import (
    SCHEMA_VERSION,
    gemm_flops,
    latency_stats,
    tflops_from_ms,
)
from swordfish.runner.torch_gemm import (
    _resolve_device,
    _time_cpu,
    _time_cuda,
    capture_env,
    write_result,
)

W4A16Backend = Literal["reference", "triton"]


def _max_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().abs().clamp_min(1e-8)
    return float(((actual.float() - expected.float()).abs() / denom).max().item())


def run_w4a16_benchmark(
    *,
    backend: W4A16Backend,
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if backend not in {"reference", "triton"}:
        raise ValueError("backend must be 'reference' or 'triton'")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if min(m, n, k, group_size, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "m, n, k, group_size, repeats, and iters must be positive; warmup must be non-negative"
        )

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    if backend == "triton" and device.type != "cuda":
        raise RuntimeError("triton W4A16 backend requires CUDA; use backend=reference for CPU")

    torch.manual_seed(seed)
    torch_dtype = TORCH_DTYPES[dtype]
    a = torch.randn((m, k), device=device, dtype=torch_dtype)
    full_weight = torch.randn((k, n), device=device, dtype=torch.float32)
    quantized = quantize_weight_int4_per_group(full_weight, group_size=group_size)

    def reference_once() -> torch.Tensor:
        return reference_w4a16_matmul(a, quantized)

    if backend == "reference":
        run_once = reference_once
    else:

        def run_once() -> torch.Tensor:
            return triton_w4a16_matmul(a, quantized)

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    last_output: list[torch.Tensor] = []

    def timed_once() -> torch.Tensor:
        output = run_once()
        if last_output:
            last_output[0] = output
        else:
            last_output.append(output)
        return output

    samples_ms = [timer(timed_once, warmup=warmup, iters=iters) for _ in range(repeats)]
    stats = latency_stats(samples_ms)
    output = last_output[0]
    expected = reference_once()

    max_abs_error = float((output.float() - expected.float()).abs().max().item())
    max_rel_error = _max_rel_error(output, expected)
    atol = 1e-3 if dtype == "fp32" else 1e-1
    rtol = 1e-3 if dtype == "fp32" else 1e-2
    matches_reference = bool(max_abs_error <= atol and max_rel_error <= rtol)
    mean_ms = stats["mean_ms"]
    flops = gemm_flops(m, n, k)

    full_precision = a.float() @ full_weight
    quantization_max_abs_error = float((expected.float() - full_precision).abs().max().item())
    quantization_max_rel_error = _max_rel_error(expected, full_precision)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "marlin_w4a16_matmul",
        "config": {
            "scope": "w4a16_matmul",
            "backend": backend,
            "shape": {"m": m, "n": n, "k": k},
            "dtype": dtype,
            "group_size": group_size,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        "env": capture_env(device, arch_label=arch_label),
        "correctness": {
            "finite_output": bool(torch.isfinite(output).all().item()),
            "checksum_fp32_sum": float(output.float().sum().item()),
            "reference_backend": "reference",
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "matches_reference": matches_reference,
            "atol": atol,
            "rtol": rtol,
            "quantization_max_abs_error_vs_fp_weight": quantization_max_abs_error,
            "quantization_max_rel_error_vs_fp_weight": quantization_max_rel_error,
            "output_shape": list(output.shape),
        },
        "metrics": {
            "latency": stats,
            "flops": flops,
            "tflops": tflops_from_ms(flops, mean_ms),
        },
    }


def write_w4a16_result(result: dict[str, Any], out_path: Path) -> None:
    write_result(result, out_path)
