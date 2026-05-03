"""Build machine-readable indexes from benchmark result JSON files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from swordfish.runner.schema import (
    SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
    validate_result_protocol,
    validate_training_result_protocol,
)

INDEX_SCHEMA_VERSION = "swordfish.result_index.v1"


def _float_or_none(value: Any) -> float | None:
    return value if isinstance(value, int | float) else None


def _validation_errors(result: dict[str, Any]) -> list[str]:
    if result.get("schema_version") == TRAINING_SCHEMA_VERSION:
        return validate_training_result_protocol(result)
    return validate_result_protocol(result)


def _summarize_result(result_dir: Path, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    env = result.get("env") if isinstance(result.get("env"), dict) else {}
    correctness = result.get("correctness") if isinstance(result.get("correctness"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    latency = metrics.get("latency") if isinstance(metrics.get("latency"), dict) else {}
    ncu = result.get("ncu") if isinstance(result.get("ncu"), dict) else None

    return {
        "file": str(path.relative_to(result_dir)),
        "benchmark": result.get("benchmark"),
        "scope": config.get("scope"),
        "backend": config.get("backend"),
        "dtype": config.get("dtype"),
        "shape": config.get("shape"),
        "gpu_class": env.get("gpu_class"),
        "gpu_name": env.get("gpu_name"),
        "git_sha": env.get("git_sha"),
        "git_dirty": env.get("git_dirty"),
        "mean_ms": _float_or_none(latency.get("mean_ms")),
        "p50_ms": _float_or_none(latency.get("p50_ms")),
        "p95_ms": _float_or_none(latency.get("p95_ms")),
        "min_ms": _float_or_none(latency.get("min_ms")),
        "tflops": _float_or_none(metrics.get("tflops")),
        "tokens_per_second": _float_or_none(metrics.get("tokens_per_second")),
        "finite_output": correctness.get("finite_output"),
        "matches_reference": correctness.get("matches_reference"),
        "max_abs_error": _float_or_none(correctness.get("max_abs_error")),
        "max_rel_error": _float_or_none(correctness.get("max_rel_error")),
        "ncu_complete": ncu.get("complete") if ncu is not None else None,
        "protocol_errors": _validation_errors(result),
    }


def build_result_index(
    result_dir: Path,
    *,
    recursive: bool = False,
    exclude_paths: Iterable[Path] = (),
    include_raw: bool = False,
) -> dict[str, Any]:
    pattern = "**/*.json" if recursive else "*.json"
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    excluded = {path.resolve() for path in exclude_paths}

    for path in sorted(result_dir.glob(pattern)):
        if not path.is_file():
            continue
        if path.resolve() in excluded:
            continue
        if not include_raw and path.name.endswith(".raw.json"):
            skipped.append(
                {"file": str(path.relative_to(result_dir)), "reason": "raw intermediate result"}
            )
            continue
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            skipped.append(
                {"file": str(path.relative_to(result_dir)), "reason": f"invalid JSON: {exc}"}
            )
            continue
        if not isinstance(loaded, dict):
            skipped.append(
                {"file": str(path.relative_to(result_dir)), "reason": "not a JSON object"}
            )
            continue
        if loaded.get("schema_version") not in {SCHEMA_VERSION, TRAINING_SCHEMA_VERSION}:
            skipped.append(
                {"file": str(path.relative_to(result_dir)), "reason": "not a swordfish result"}
            )
            continue
        rows.append(_summarize_result(result_dir, path, loaded))

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "result_dir": str(result_dir),
        "recursive": recursive,
        "include_raw": include_raw,
        "count": len(rows),
        "skipped_count": len(skipped),
        "results": rows,
        "skipped": skipped,
    }


def write_result_index(
    result_dir: Path, out_path: Path, *, recursive: bool = False, include_raw: bool = False
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index = build_result_index(
        result_dir,
        recursive=recursive,
        exclude_paths=(out_path,),
        include_raw=include_raw,
    )
    with out_path.open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path
