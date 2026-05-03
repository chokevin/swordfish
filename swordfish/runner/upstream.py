"""Render maintainer-ready upstream contribution packets from result JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from swordfish.runner.schema import (
    TRAINING_SCHEMA_VERSION,
    validate_result_protocol,
    validate_training_result_protocol,
)

UpstreamTarget = Literal[
    "triton",
    "pytorch-inductor",
    "cutlass-cute",
    "jax-pallas",
    "tilelang",
    "vllm",
    "ort",
    "pyptx",
    "liger",
]

TARGET_LABELS: dict[str, str] = {
    "triton": "Triton",
    "pytorch-inductor": "PyTorch/Inductor",
    "cutlass-cute": "CUTLASS/CuTe",
    "jax-pallas": "JAX/Pallas",
    "tilelang": "TileLang",
    "vllm": "vLLM",
    "ort": "ONNX Runtime / ORT GenAI",
    "pyptx": "pyptx",
    "liger": "Liger Kernel",
}


def _format_shape(shape: Any) -> str:
    if not isinstance(shape, dict):
        return "unknown"
    preferred = [key for key in ("m", "n", "k", "batch_size", "seq_len", "n_embd") if key in shape]
    rest = [key for key in shape if key not in preferred]
    return ", ".join(f"{key}={shape[key]}" for key in [*preferred, *rest])


def _format_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _latency_summary(result: dict[str, Any]) -> str:
    latency = result.get("metrics", {}).get("latency", {})
    if not isinstance(latency, dict):
        return "Latency summary unavailable."
    return (
        f"mean={_format_value(latency.get('mean_ms'))} ms, "
        f"p50={_format_value(latency.get('p50_ms'))} ms, "
        f"p95={_format_value(latency.get('p95_ms'))} ms, "
        f"min={_format_value(latency.get('min_ms'))} ms"
    )


def _validation_errors(result: dict[str, Any]) -> list[str]:
    if result.get("schema_version") == TRAINING_SCHEMA_VERSION:
        return validate_training_result_protocol(result)
    return validate_result_protocol(result)


def _correctness_summary(result: dict[str, Any]) -> str:
    correctness = result.get("correctness", {})
    if not isinstance(correctness, dict):
        return "Correctness summary unavailable."
    fields = [
        f"finite_output={_format_value(correctness.get('finite_output'))}",
        f"matches_reference={_format_value(correctness.get('matches_reference'))}",
    ]
    if "max_abs_error" in correctness:
        fields.append(f"max_abs_error={_format_value(correctness['max_abs_error'])}")
    if "max_rel_error" in correctness:
        fields.append(f"max_rel_error={_format_value(correctness['max_rel_error'])}")
    if "final_loss" in correctness:
        fields.append(f"final_loss={_format_value(correctness['final_loss'])}")
    return ", ".join(fields)


def _ncu_summary(result: dict[str, Any]) -> str:
    ncu = result.get("ncu")
    if not isinstance(ncu, dict):
        return "No NCU summary attached."
    metrics = ncu.get("metrics", {})
    missing = ncu.get("missing_metrics", [])
    return (
        f"complete={_format_value(ncu.get('complete'))}, "
        f"metrics={json.dumps(metrics, sort_keys=True)}, "
        f"missing_metrics={json.dumps(missing)}"
    )


def render_upstream_packet(
    *,
    result_path: Path,
    target: UpstreamTarget,
    title: str | None = None,
    ask: str | None = None,
) -> str:
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        raise ValueError("result JSON must contain an object")

    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    env = result.get("env") if isinstance(result.get("env"), dict) else {}
    validation_errors = _validation_errors(result)
    target_label = TARGET_LABELS[target]
    packet_title = title or (
        f"{target_label} repro: {result.get('benchmark', 'benchmark')} "
        f"{config.get('backend', 'unknown')} {_format_shape(config.get('shape'))}"
    )
    command = result.get("command")

    lines = [
        f"# {packet_title}",
        "",
        f"**Target:** {target_label}",
        f"**Benchmark:** {_format_value(result.get('benchmark'))}",
        f"**Backend:** {_format_value(config.get('backend'))}",
        f"**Scope:** {_format_value(config.get('scope'))}",
        f"**Shape:** {_format_shape(config.get('shape'))}",
        f"**Dtype:** {_format_value(config.get('dtype'))}",
        "",
        "## Ask",
        "",
        ask or "Please use this as a reproducible benchmark/correctness artifact.",
        "",
        "## Reproduction",
        "",
        "```bash",
        " ".join(str(part) for part in command)
        if isinstance(command, list)
        else "# command unavailable",
        "```",
        "",
        "## Environment",
        "",
        f"- GPU: {_format_value(env.get('gpu_name'))} ({_format_value(env.get('gpu_class'))})",
        f"- Compute capability: {_format_value(env.get('gpu_cc'))}",
        f"- Torch/CUDA: {_format_value(env.get('torch'))} / {_format_value(env.get('torch_cuda'))}",
        f"- Driver: CUDA {_format_value(env.get('cuda_driver'))}, NVIDIA {_format_value(env.get('nvidia_driver'))}",
        f"- Triton: {_format_value(env.get('triton'))}",
        f"- Git: {_format_value(env.get('git_sha'))}, dirty={_format_value(env.get('git_dirty'))}",
        "",
        "## Correctness",
        "",
        _correctness_summary(result),
        "",
        "## Latency",
        "",
        _latency_summary(result),
        "",
        "## Nsight Compute",
        "",
        _ncu_summary(result),
        "",
        "## Result protocol validation",
        "",
    ]
    if validation_errors:
        lines.extend(f"- {error}" for error in validation_errors)
    else:
        lines.append("- OK")
    lines.extend(
        [
            "",
            "## Notes for maintainers",
            "",
            "- This packet is generated from a `swordfish` JSON result.",
            "- Performance claims should not be copied without attaching the full JSON result.",
            "- If this is not a bug, convert it into a docs/example/benchmark artifact instead.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_upstream_packet(
    *,
    result_path: Path,
    target: UpstreamTarget,
    out_path: Path,
    title: str | None = None,
    ask: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_upstream_packet(result_path=result_path, target=target, title=title, ask=ask)
    )
    return out_path
