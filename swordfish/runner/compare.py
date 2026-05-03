"""Compare benchmark result JSON files without hand-reading each artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from swordfish.runner.schema import (
    TRAINING_SCHEMA_VERSION,
    validate_result_protocol,
    validate_training_result_protocol,
)


def _shape_summary(shape: Any) -> str:
    if not isinstance(shape, dict):
        return "unknown"
    ordered = [key for key in ("m", "n", "k", "batch_size", "seq_len", "n_embd") if key in shape]
    ordered.extend(key for key in shape if key not in ordered)
    return " ".join(f"{key}={shape[key]}" for key in ordered)


def _get_float(value: Any) -> float | None:
    return value if isinstance(value, int | float) else None


def _validation_errors(result: dict[str, Any]) -> list[str]:
    if result.get("schema_version") == TRAINING_SCHEMA_VERSION:
        return validate_training_result_protocol(result)
    return validate_result_protocol(result)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _result_row(path: Path, result: dict[str, Any], baseline_ms: float | None) -> list[str]:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    env = result.get("env") if isinstance(result.get("env"), dict) else {}
    correctness = result.get("correctness") if isinstance(result.get("correctness"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    latency = metrics.get("latency") if isinstance(metrics.get("latency"), dict) else {}
    mean_ms = _get_float(latency.get("mean_ms"))
    speedup = baseline_ms / mean_ms if baseline_ms and mean_ms and mean_ms > 0 else None
    ncu = result.get("ncu")
    validation_errors = _validation_errors(result)

    return [
        path.name,
        str(result.get("benchmark", "")),
        str(config.get("backend", "")),
        str(env.get("gpu_class", "")),
        str(config.get("dtype", "")),
        _shape_summary(config.get("shape")),
        _format_float(mean_ms),
        _format_float(speedup),
        _format_float(_get_float(metrics.get("tflops"))),
        str(correctness.get("matches_reference", "")),
        str(correctness.get("finite_output", "")),
        str(ncu.get("complete", "")) if isinstance(ncu, dict) else "",
        "OK" if not validation_errors else "; ".join(validation_errors),
    ]


def render_results_comparison(paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("at least one result path is required")

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError(f"{path} must contain a JSON object")
        loaded.append((path, result))

    first_latency = loaded[0][1].get("metrics", {}).get("latency", {})
    baseline_ms = (
        _get_float(first_latency.get("mean_ms")) if isinstance(first_latency, dict) else None
    )

    header = [
        "file",
        "benchmark",
        "backend",
        "gpu",
        "dtype",
        "shape",
        "mean_ms",
        "speedup_vs_first",
        "tflops",
        "matches_reference",
        "finite_output",
        "ncu_complete",
        "protocol",
    ]
    rows = [_result_row(path, result, baseline_ms) for path, result in loaded]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_results_comparison(paths: Sequence[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_results_comparison(paths))
    return out_path
