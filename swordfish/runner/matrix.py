"""Helpers for running the same GEMM contract across GPU architecture labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from swordfish.runner.schema import validate_result_protocol
from swordfish.runner.torch_gemm import run_gemm_benchmark, write_result


DEFAULT_ARCH_LABELS = ("a100", "h100", "h200")


def _find_result_file(
    result_dir: Path, filename: str, recursive: bool
) -> tuple[Path | None, list[Path]]:
    direct = result_dir / filename
    if direct.exists():
        return direct, []
    if not recursive:
        return None, []

    matches = sorted(path for path in result_dir.rglob(filename) if path.is_file())
    if len(matches) == 1:
        return matches[0], []
    if len(matches) > 1:
        return None, matches
    return None, []


def run_gemm_matrix(
    *,
    arch_labels: Sequence[str],
    out_dir: Path,
    prefix: str | None,
    m: int,
    n: int,
    k: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str,
    allow_cpu: bool,
    seed: int,
    backend: str = "torch",
    command: list[str] | None = None,
) -> list[Path]:
    """Run one GEMM per architecture label and write one JSON file per label.

    This is mainly useful for local schema smoke tests with ``--allow-cpu``.
    Real A100/H100/H200 numbers should come from one scheduled GPU job per
    architecture, because ``run_torch_gemm`` intentionally fails if the requested
    arch label does not match the CUDA device that Kueue routed us onto.
    """
    written: list[Path] = []
    result_prefix = prefix or f"{backend}-gemm"
    for arch_label in arch_labels:
        result = run_gemm_benchmark(
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
            backend=backend,
        )
        if command is not None:
            result["command"] = command
        out_path = out_dir / f"{result_prefix}-{arch_label}.json"
        write_result(result, out_path)
        written.append(out_path)
    return written


def validate_gemm_matrix_results(
    *,
    arch_labels: Sequence[str],
    result_dir: Path,
    prefix: str | None,
    backend: str,
    dtype: str | None = None,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    require_ncu: bool = False,
    recursive: bool = False,
) -> list[str]:
    """Validate that a cross-arch GEMM result directory is ready to compare."""
    errors: list[str] = []
    result_prefix = prefix or f"{backend}-gemm"

    for arch_label in arch_labels:
        filename = f"{result_prefix}-{arch_label}.json"
        path, duplicates = _find_result_file(result_dir, filename, recursive)
        if duplicates:
            formatted = ", ".join(str(path) for path in duplicates)
            errors.append(f"{arch_label}: multiple result files named {filename}: {formatted}")
            continue
        if path is None:
            errors.append(f"{arch_label}: missing result file {result_dir / filename}")
            continue

        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{arch_label}: invalid JSON in {path}: {exc}")
            continue
        if not isinstance(result, dict):
            errors.append(f"{arch_label}: result file {path} must contain a JSON object")
            continue

        errors.extend(f"{arch_label}: {error}" for error in validate_result_protocol(result))
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        env = result.get("env") if isinstance(result.get("env"), dict) else {}
        correctness = (
            result.get("correctness") if isinstance(result.get("correctness"), dict) else {}
        )

        if result.get("benchmark") != "torch_gemm":
            errors.append(f"{arch_label}: benchmark must be torch_gemm")
        if config.get("scope") != "gemm":
            errors.append(f"{arch_label}: config.scope must be gemm")
        if config.get("backend") != backend:
            errors.append(f"{arch_label}: config.backend must be {backend}")
        if dtype is not None and config.get("dtype") != dtype:
            errors.append(f"{arch_label}: config.dtype must be {dtype}")

        shape = config.get("shape") if isinstance(config.get("shape"), dict) else {}
        for dim_name, expected in (("m", m), ("n", n), ("k", k)):
            if expected is not None and shape.get(dim_name) != expected:
                errors.append(f"{arch_label}: config.shape.{dim_name} must be {expected}")

        if env.get("gpu_class") != arch_label:
            errors.append(f"{arch_label}: env.gpu_class must be {arch_label}")
        if correctness.get("finite_output") is not True:
            errors.append(f"{arch_label}: correctness.finite_output must be true")
        if correctness.get("matches_reference") is not True:
            errors.append(f"{arch_label}: correctness.matches_reference must be true")

        if require_ncu:
            ncu = result.get("ncu")
            if not isinstance(ncu, dict):
                errors.append(f"{arch_label}: missing ncu summary")
            elif ncu.get("complete") is not True:
                missing = ncu.get("missing_metrics", [])
                errors.append(f"{arch_label}: incomplete ncu summary; missing_metrics={missing}")

    return errors
