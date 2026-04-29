"""Human-readable benchmark completion reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from swordfish.runner.index import build_result_index
from swordfish.runner.matrix import validate_gemm_matrix_results


def _shape_summary(shape: Any) -> str:
    if not isinstance(shape, dict):
        return "unknown"
    preferred = ["m", "n", "k", "batch_size", "seq_len", "n_embd"]
    ordered = [key for key in preferred if key in shape]
    ordered.extend(key for key in shape if key not in ordered)
    return " ".join(f"{key}={shape[key]}" for key in ordered)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _render_observed_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No benchmark result JSONs were indexed."]

    header = [
        "file",
        "benchmark",
        "backend",
        "gpu",
        "dtype",
        "shape",
        "mean_ms",
        "tflops",
        "matches_reference",
        "ncu_complete",
        "protocol",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("benchmark", "")),
            str(item.get("backend", "")),
            str(item.get("gpu_class", "")),
            str(item.get("file", "")),
        ),
    ):
        protocol_errors = row.get("protocol_errors", [])
        protocol = (
            "OK" if not protocol_errors else "; ".join(str(error) for error in protocol_errors)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("file", "")),
                    str(row.get("benchmark", "")),
                    str(row.get("backend", "")),
                    str(row.get("gpu_class", "")),
                    str(row.get("dtype", "")),
                    _shape_summary(row.get("shape")),
                    _format_value(row.get("mean_ms")),
                    _format_value(row.get("tflops")),
                    _format_value(row.get("matches_reference")),
                    _format_value(row.get("ncu_complete")),
                    protocol,
                ]
            )
            + " |"
        )
    return lines


def render_completion_report(
    *,
    result_dir: Path,
    arch_labels: Sequence[str],
    prefix: str | None,
    backend: str,
    dtype: str | None = None,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    require_ncu: bool = False,
    recursive: bool = False,
) -> tuple[str, list[str]]:
    errors = validate_gemm_matrix_results(
        arch_labels=arch_labels,
        result_dir=result_dir,
        prefix=prefix,
        backend=backend,
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        require_ncu=require_ncu,
        recursive=recursive,
    )
    index = build_result_index(result_dir, recursive=recursive)
    status = "READY" if not errors else "BLOCKED"

    lines = [
        "# Swordfish benchmark completion report",
        "",
        f"**Status:** {status}",
        "",
        "## Gate configuration",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Architectures: `{', '.join(arch_labels)}`",
        f"- Backend: `{backend}`",
        f"- Prefix: `{prefix or f'{backend}-gemm'}`",
        f"- Dtype: `{dtype or 'any'}`",
        f"- Shape: `{_shape_summary({'m': m, 'n': n, 'k': k})}`",
        f"- Recursive search: `{recursive}`",
        f"- Require complete NCU: `{require_ncu}`",
        "",
        "## Completion gate",
        "",
    ]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- Complete: every requested architecture has a valid matching result.")

    lines.extend(
        [
            "",
            "## Indexed artifacts",
            "",
            f"- Result rows: `{index['count']}`",
            f"- Skipped JSON files: `{index['skipped_count']}`",
            "",
        ]
    )
    lines.extend(_render_observed_rows(index["results"]))
    lines.append("")
    return "\n".join(lines), errors


def write_completion_report(
    *,
    result_dir: Path,
    out_path: Path,
    arch_labels: Sequence[str],
    prefix: str | None,
    backend: str,
    dtype: str | None = None,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    require_ncu: bool = False,
    recursive: bool = False,
) -> tuple[Path, list[str]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report, errors = render_completion_report(
        result_dir=result_dir,
        arch_labels=arch_labels,
        prefix=prefix,
        backend=backend,
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        require_ncu=require_ncu,
        recursive=recursive,
    )
    out_path.write_text(report)
    return out_path, errors
