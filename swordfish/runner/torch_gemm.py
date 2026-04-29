"""Torch/cuBLAS GEMM smoke benchmark used by the airun runner."""

from __future__ import annotations

import json
from importlib import metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from swordfish.runner.backends import TORCH_DTYPES, GemmState, get_gemm_backend
from swordfish.runner.schema import (
    SCHEMA_VERSION,
    attach_ncu_summary,
    gemm_estimated_bytes,
    gemm_flops,
    gpu_class_from_name,
    latency_stats,
    pct_of_peak,
    peak_for,
    tbps_from_ms,
    tflops_from_ms,
)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _tool_version(command: str) -> str | None:
    path = shutil.which(command)
    if path is None:
        return None
    try:
        proc = subprocess.run(
            [path, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines()[0] if proc.stdout else None


def _cuda_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        version = torch._C._cuda_getDriverVersion()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        version = None
    if version:
        return str(version)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines()[0] if proc.stdout else None


def _nvidia_driver_version() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines()[0] if proc.stdout else None


def _git_value(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def capture_env(device: torch.device, arch_label: str | None = None) -> dict[str, Any]:
    target_gpu_class = arch_label.lower() if arch_label else None
    env: dict[str, Any] = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_driver": _cuda_driver_version(),
        "nvidia_driver": _nvidia_driver_version(),
        "triton": _package_version("triton"),
        "ncu": _tool_version("ncu"),
        "cuda_available": torch.cuda.is_available(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"])),
        "target_gpu_class": target_gpu_class,
        "source_ref": os.environ.get("REF"),
        "container_image": os.environ.get("CONTAINER_IMAGE") or os.environ.get("IMAGE"),
    }

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        actual_gpu_class = gpu_class_from_name(props.name)
        if (
            target_gpu_class is not None
            and actual_gpu_class != "unknown"
            and target_gpu_class != actual_gpu_class
        ):
            raise RuntimeError(
                f"target arch {target_gpu_class!r} landed on {props.name!r} "
                f"({actual_gpu_class}); fix the Kueue/DRA/nodeSelector routing"
            )
        env.update(
            {
                "device": str(device),
                "gpu_name": props.name,
                "gpu_class": actual_gpu_class
                if actual_gpu_class != "unknown"
                else target_gpu_class or "unknown",
                "gpu_cc": f"{props.major}.{props.minor}",
                "gpu_mem_gb": round(props.total_memory / 2**30, 1),
                "gpu_sm_count": props.multi_processor_count,
            }
        )
    else:
        env.update(
            {
                "device": str(device),
                "gpu_name": None,
                "gpu_class": gpu_class_from_name(None, target_gpu_class),
                "gpu_cc": None,
                "gpu_mem_gb": None,
                "gpu_sm_count": None,
            }
        )

    return env


def _time_cuda(fn, warmup: int, iters: int) -> float:
    """Return average milliseconds per call using CUDA event timing.

    CUDA work is asynchronous from Python's point of view, so the synchronize
    calls create clean timing boundaries. Events measure elapsed time on the
    CUDA stream instead of Python wall-clock time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_cpu(fn, warmup: int, iters: int) -> float:
    """Return average milliseconds per call using Python wall-clock timing."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


def _resolve_device(device_name: str, allow_cpu: bool) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA is not available; pass --allow-cpu only for local smoke tests")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is not available")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU timing is disabled by default; pass --allow-cpu for local tests")
    return device


def _reference_tolerances(dtype: str) -> tuple[float, float]:
    if dtype == "fp32":
        return 1e-4, 1e-4
    return 1e-2, 1e-1


def _reference_check(state: GemmState, backend_name: str, dtype: str) -> dict[str, Any]:
    if backend_name == "torch":
        return {
            "reference_backend": "torch",
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "matches_reference": True,
            "atol": 0.0,
            "rtol": 0.0,
        }

    with torch.no_grad():
        reference = torch.mm(state.a, state.b)
        actual = state.out.float()
        expected = reference.float()
        abs_error = (actual - expected).abs()
        max_abs_error = float(abs_error.max().item())
        denom = expected.abs().clamp_min(1e-12)
        max_rel_error = float((abs_error / denom).max().item())

    rtol, atol = _reference_tolerances(dtype)
    matches = bool(max_abs_error <= atol and max_rel_error <= rtol)
    return {
        "reference_backend": "torch",
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "matches_reference": matches,
        "atol": atol,
        "rtol": rtol,
    }


def run_gemm_benchmark(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    ncu_csv: Path | None = None,
    backend: str = "torch",
) -> dict[str, Any]:
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if min(m, n, k, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "m, n, k, repeats, and iters must be positive; warmup must be non-negative"
        )

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    gemm_backend = get_gemm_backend(backend)

    # Inputs/output are allocated before timing so the benchmark measures only
    # the selected backend's matmul call. torch.empty() is safe because every
    # backend must overwrite the whole output tensor before we read it.
    state = gemm_backend.prepare(m=m, n=n, k=k, dtype=dtype, device=device, seed=seed)

    def matmul_once() -> torch.Tensor:
        return gemm_backend.run(state)

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    samples_ms = [timer(matmul_once, warmup=warmup, iters=iters) for _ in range(repeats)]
    stats = latency_stats(samples_ms)

    # _time_cuda already synchronized before returning. The .item() calls below
    # are CPU readbacks and would synchronize if any GPU work were still pending.
    finite = bool(torch.isfinite(state.out).all().item())
    checksum = float(state.out.float().sum().item())
    reference = _reference_check(state, gemm_backend.name, dtype)

    env = capture_env(device, arch_label=arch_label)
    gpu_class = env["gpu_class"]

    # One GEMM does about 2*M*N*K FLOPs: M*N output elements, each a length-K
    # dot product counted as one multiply plus one add per K step.
    flops = gemm_flops(m, n, k)

    # Lower-bound tensor traffic: read A once, read B once, write C once. Real
    # DRAM traffic can differ due to tiling, cache reuse, epilogues, padding, or
    # extra reads/writes; Nsight Compute is the source of truth for actual HBM.
    estimated_bytes = gemm_estimated_bytes(m, n, k, dtype)
    mean_ms = stats["mean_ms"]

    # Convert "amount of work" plus measured latency into achieved throughput.
    tflops = tflops_from_ms(flops, mean_ms)
    bandwidth_tbps = tbps_from_ms(estimated_bytes, mean_ms)

    compute_peak = peak_for(gpu_class, dtype, "compute_tflops")
    hbm_peak = peak_for(gpu_class, dtype, "hbm_tbps")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": f"{gemm_backend.name}_gemm",
        "config": {
            "scope": "gemm",
            "backend": gemm_backend.name,
            "shape": {
                "m": m,
                "n": n,
                "k": k,
            },
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        "env": env,
        "correctness": {
            "finite_output": finite,
            "checksum_fp32_sum": checksum,
            **reference,
        },
        # These metrics answer three separate questions:
        # 1. latency: how long did one GEMM take?
        # 2. tflops/compute_sol_pct: how much of the GPU's math peak did we reach?
        # 3. estimated_bandwidth/hbm_sol_pct: does the run look memory-limited?
        "metrics": {
            "latency": stats,
            "flops": flops,
            "tflops": tflops,
            "compute_peak_tflops": compute_peak,
            "compute_sol_pct": pct_of_peak(tflops, compute_peak),
            "estimated_bytes": estimated_bytes,
            "estimated_bandwidth_tbps": bandwidth_tbps,
            "hbm_peak_tbps": hbm_peak,
            "estimated_hbm_sol_pct": pct_of_peak(bandwidth_tbps, hbm_peak),
        },
    }
    if ncu_csv is not None:
        result = attach_ncu_summary(result, ncu_csv)
    return result


def run_torch_gemm(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    ncu_csv: Path | None = None,
) -> dict[str, Any]:
    return run_gemm_benchmark(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        repeats=repeats,
        warmup=warmup,
        iters=iters,
        device_name=device_name,
        allow_cpu=allow_cpu,
        arch_label=arch_label,
        seed=seed,
        ncu_csv=ncu_csv,
        backend="torch",
    )


def write_result(result: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
