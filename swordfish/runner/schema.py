"""Shared result schema and metric helpers for runner outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "swordfish.runner.v1"

# Training-side benchmarks (Liger per-kernel sweeps, FSDP step timing) carry
# different fields than the inference GEMM/transformer schema: peak memory,
# forward+backward latency, optimizer config, distributed strategy, and a
# liger_patch sub-block describing which kernels were swapped. Kept as a
# sibling schema so the inference schema does not get overloaded.
TRAINING_SCHEMA_VERSION = "swordfish.training.v1"


DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}

# Conservative single-GPU theoretical peaks used only to turn raw throughput
# into a rough SOL percentage. Exact SKU clocks/power caps still belong in the
# captured environment next to each run.
GPU_PEAKS = {
    "a100": {
        "fp16_tflops": 312.0,
        "bf16_tflops": 312.0,
        "fp32_tflops": 19.5,
        "hbm_tbps": 2.039,
    },
    "h100": {
        "fp16_tflops": 989.0,
        "bf16_tflops": 989.0,
        "fp32_tflops": 67.0,
        "hbm_tbps": 3.35,
    },
    "h200": {
        "fp16_tflops": 989.0,
        "bf16_tflops": 989.0,
        "fp32_tflops": 67.0,
        "hbm_tbps": 4.8,
    },
}

NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
)


COMMON_RESULT_FIELDS = ("schema_version", "benchmark", "config", "env", "correctness", "metrics")
COMMON_ENV_FIELDS = (
    "git_sha",
    "git_dirty",
    "torch",
    "torch_cuda",
    "cuda_driver",
    "nvidia_driver",
    "triton",
    "ncu",
    "gpu_name",
    "gpu_class",
    "gpu_cc",
)
COMMON_CONFIG_FIELDS = ("scope", "backend", "shape", "dtype")


def _missing(mapping: dict[str, Any], fields: tuple[str, ...], prefix: str) -> list[str]:
    return [f"{prefix}.{field}" for field in fields if field not in mapping]


def validate_result_protocol(result: dict[str, Any]) -> list[str]:
    """Return missing fields that would make cross-GPU comparisons ambiguous."""
    errors = _missing(result, COMMON_RESULT_FIELDS, "result")
    if errors:
        return errors
    if result.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"result.schema_version must be {SCHEMA_VERSION}")

    config = result["config"]
    env = result["env"]
    if not isinstance(config, dict):
        errors.append("result.config must be an object")
    else:
        errors.extend(_missing(config, COMMON_CONFIG_FIELDS, "config"))
        shape = config.get("shape")
        if not isinstance(shape, dict) or not shape:
            errors.append("config.shape must be a non-empty object")

    if not isinstance(env, dict):
        errors.append("result.env must be an object")
    else:
        errors.extend(_missing(env, COMMON_ENV_FIELDS, "env"))

    return errors


# Training-side schema. Each per-kernel result captures the matched baseline +
# Liger pair on identical input, so the metrics block carries both modes plus
# explicit deltas. End-to-end FSDP step results reuse the same wrapper but with
# config.scope == "fsdp_train_step" and metrics.modes == {"baseline": ...} only.
TRAINING_CONFIG_FIELDS = ("scope", "kernel", "dtype", "shape")
TRAINING_LIGER_FIELDS = ("applied", "version", "kernel_module")


def validate_training_result_protocol(result: dict[str, Any]) -> list[str]:
    """Return missing fields for a training-side result (Liger sweep, FSDP step)."""
    errors = _missing(result, COMMON_RESULT_FIELDS, "result")
    if errors:
        return errors
    if result.get("schema_version") != TRAINING_SCHEMA_VERSION:
        errors.append(f"result.schema_version must be {TRAINING_SCHEMA_VERSION}")

    config = result["config"]
    if not isinstance(config, dict):
        errors.append("result.config must be an object")
    else:
        errors.extend(_missing(config, TRAINING_CONFIG_FIELDS, "config"))
        shape = config.get("shape")
        if not isinstance(shape, dict) or not shape:
            errors.append("config.shape must be a non-empty object")
        liger = config.get("liger")
        if not isinstance(liger, dict):
            errors.append("config.liger must be an object")
        else:
            errors.extend(_missing(liger, TRAINING_LIGER_FIELDS, "config.liger"))

    env = result.get("env")
    if not isinstance(env, dict):
        errors.append("result.env must be an object")
    else:
        errors.extend(_missing(env, COMMON_ENV_FIELDS, "env"))

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("result.metrics must be an object")
    else:
        modes = metrics.get("modes")
        if not isinstance(modes, dict) or not modes:
            errors.append("metrics.modes must be a non-empty object")

    return errors


def gpu_class_from_name(name: str | None, fallback: str | None = None) -> str:
    """Normalize CUDA device names into the roadmap's A100/H100/H200 classes."""
    lowered = (name or "").lower()
    if "h200" in lowered:
        return "h200"
    if "h100" in lowered:
        return "h100"
    if "a100" in lowered:
        return "a100"
    if fallback:
        lowered = fallback.lower()
        if lowered in GPU_PEAKS:
            return lowered
    return "unknown"


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def latency_stats(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples_ms,
        "mean_ms": sum(samples_ms) / len(samples_ms) if samples_ms else float("nan"),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms) if samples_ms else float("nan"),
    }


def gemm_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def gemm_estimated_bytes(m: int, n: int, k: int, dtype: str) -> int:
    bytes_per = DTYPE_BYTES[dtype]
    return (m * k + k * n + m * n) * bytes_per


def tflops_from_ms(flops: int, ms: float) -> float:
    if ms <= 0 or math.isnan(ms):
        return float("nan")
    return flops / (ms * 1e9)


def tbps_from_ms(num_bytes: int, ms: float) -> float:
    if ms <= 0 or math.isnan(ms):
        return float("nan")
    return num_bytes / (ms / 1000.0) / 1e12


def pct_of_peak(value: float, peak: float | None) -> float | None:
    if peak is None or peak <= 0 or math.isnan(value):
        return None
    return value / peak * 100.0


def peak_for(gpu_class: str, dtype: str, metric: str) -> float | None:
    peaks = GPU_PEAKS.get(gpu_class)
    if not peaks:
        return None
    if metric == "compute_tflops":
        return peaks.get(f"{dtype}_tflops")
    if metric == "hbm_tbps":
        return peaks.get("hbm_tbps")
    raise ValueError(f"unknown peak metric: {metric}")


def _parse_metric_value(raw: str) -> float | str:
    cleaned = raw.strip().replace(",", "")
    if cleaned in {"", "n/a", "N/A", "--"}:
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return raw.strip()


def parse_ncu_csv(path: Path) -> dict[str, Any]:
    """Extract a small SOL summary from Nsight Compute CSV output.

    Nsight Compute has two common CSV shapes:
    1. "Metric Name","Metric Value" long-form rows.
    2. Wide rows where metric names are column headers.

    The runner accepts either and preserves only the stable metrics we need for
    the Week 1 smoke contract. It also records missing metrics explicitly so a
    malformed or partial profiler CSV cannot look like a complete NCU summary.
    """
    lines = path.read_text().splitlines()
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "Metric Name" in line or any(metric in line for metric in NCU_METRICS)
        ),
        None,
    )
    if header_idx is None:
        return {
            "path": str(path),
            "rows": 0,
            "metrics": {},
            "missing_metrics": list(NCU_METRICS),
            "complete": False,
        }

    rows = list(csv.DictReader(lines[header_idx:]))
    metrics: dict[str, Any] = {}
    for row in rows:
        metric_name = row.get("Metric Name")
        if metric_name:
            metric_name = metric_name.strip()
            if metric_name in NCU_METRICS:
                metrics[metric_name] = _parse_metric_value(row.get("Metric Value", ""))
            continue
        for metric in NCU_METRICS:
            if metric in row and metric not in metrics:
                metrics[metric] = _parse_metric_value(row[metric])

    missing_metrics = [metric for metric in NCU_METRICS if metric not in metrics]
    return {
        "path": str(path),
        "rows": len(rows),
        "metrics": metrics,
        "missing_metrics": missing_metrics,
        "complete": not missing_metrics,
    }


def attach_ncu_summary(result: dict[str, Any], ncu_csv: Path) -> dict[str, Any]:
    updated = dict(result)
    updated["ncu"] = parse_ncu_csv(ncu_csv)
    return updated
