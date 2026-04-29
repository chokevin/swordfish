"""Liger Kernel per-kernel benchmark runner (Week 1 Wednesday).

Pairs an unmodified Hugging Face / PyTorch baseline against the
`liger_kernel`-fused equivalent on identical input, captures forward and
backward latency plus peak GPU memory, and reports correctness deltas.

Result JSONs follow the `TRAINING_SCHEMA_VERSION` sibling schema in
`swordfish.runner.schema` so the dashboard indexer treats them as
training-side rows distinct from the inference GEMM matrix.

Liger imports are deferred so this module loads on hosts without
`liger-kernel` installed; when `--mode baseline` is requested the
liger half is reported as `skipped` with an explicit reason. CPU smoke
runs (`--allow-cpu`) similarly skip the liger half because Liger's
fused kernels assume CUDA + Triton.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from swordfish.runner.schema import (
    TRAINING_SCHEMA_VERSION,
    attach_ncu_summary,
    latency_stats,
)
from swordfish.runner.torch_gemm import capture_env

KERNEL_NAMES = ("rmsnorm", "swiglu", "rope", "fused_linear_ce")
SUPPORTED_KERNELS = ("rmsnorm", "swiglu")
DEFAULT_DTYPE = "bf16"
DTYPE_TO_TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass
class KernelInputs:
    """Input tensors and the closure that runs one forward+backward pass."""

    tensors: list[torch.Tensor]
    forward: Callable[[], torch.Tensor]


@dataclass
class KernelOutcome:
    """One mode (baseline or liger) of a paired kernel benchmark."""

    forward_ms: dict[str, Any]
    backward_ms: dict[str, Any]
    peak_gpu_mem_mb: float | None
    output_checksum_fp32: float
    finite_output: bool
    skipped: bool = False
    skip_reason: str | None = None


# --- kernel specs --------------------------------------------------------


def _liger_version() -> str | None:
    try:
        return metadata.version("liger-kernel")
    except metadata.PackageNotFoundError:
        return None


def _build_rmsnorm_baseline(
    hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device
) -> nn.Module:
    # Reference RMSNorm matching the LLaMA / Liger formulation: weight*x/rms(x).
    class RMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x.to(input_dtype)).to(input_dtype)

    return RMSNorm().to(device=device, dtype=dtype)


def _build_rmsnorm_liger(
    hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device
) -> nn.Module:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm  # noqa: PLC0415

    module = LigerRMSNorm(hidden_size=hidden_size, eps=eps)
    return module.to(device=device, dtype=dtype)


def _build_swiglu_baseline(
    hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: torch.device
) -> nn.Module:
    # LLaMA-style SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
    class SwiGLUMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    return SwiGLUMLP().to(device=device, dtype=dtype)


def _build_swiglu_liger(
    hidden_size: int, intermediate_size: int, dtype: torch.dtype, device: torch.device
) -> nn.Module:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP  # noqa: PLC0415

    # LigerSwiGLUMLP expects an HF-shaped config object exposing hidden_size,
    # intermediate_size, hidden_act. Build a tiny stand-in.
    class _Cfg:
        def __init__(self) -> None:
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.hidden_act = "silu"

    return LigerSwiGLUMLP(_Cfg()).to(device=device, dtype=dtype)


def _make_inputs_for_kernel(
    kernel: str, batch: int, seq: int, hidden: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(batch, seq, hidden, dtype=dtype, device=device, requires_grad=True)


# --- timing primitives ---------------------------------------------------


def _time_cuda_ms(fn: Callable[[], Any], warmup: int, iters: int) -> float:
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


def _time_cpu_ms(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) * 1000.0 / iters


@contextlib.contextmanager
def _track_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024 * 1024)


# --- per-mode runner -----------------------------------------------------


def _run_mode(
    *,
    module: nn.Module,
    make_input: Callable[[], torch.Tensor],
    repeats: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> KernelOutcome:
    timer = _time_cuda_ms if device.type == "cuda" else _time_cpu_ms

    # Forward-only timing reuses one input across iters; the module's params
    # do not move so the closure remains pure.
    fwd_input = make_input()

    def forward_only() -> torch.Tensor:
        return module(fwd_input)

    fwd_samples = [timer(forward_only, warmup=warmup, iters=iters) for _ in range(repeats)]

    # Forward+backward timing builds a new graph each iteration so backward()
    # is legal more than once. Inputs are cheap to reallocate; the timer
    # absorbs that cost only once during warmup.
    def forward_backward() -> None:
        x = make_input()
        out = module(x)
        loss = out.float().sum()
        loss.backward()

    bwd_samples = [timer(forward_backward, warmup=warmup, iters=iters) for _ in range(repeats)]

    # One clean output tensor for correctness fingerprinting outside of timing.
    with torch.no_grad():
        clean_out = module(make_input()).detach()
    finite = bool(torch.isfinite(clean_out).all().item())
    checksum = float(clean_out.float().sum().item())

    return KernelOutcome(
        forward_ms=latency_stats(fwd_samples),
        backward_ms=latency_stats(bwd_samples),
        peak_gpu_mem_mb=_peak_memory_mb(device),
        output_checksum_fp32=checksum,
        finite_output=finite,
    )


def _skipped_outcome(reason: str) -> KernelOutcome:
    return KernelOutcome(
        forward_ms={},
        backward_ms={},
        peak_gpu_mem_mb=None,
        output_checksum_fp32=float("nan"),
        finite_output=False,
        skipped=True,
        skip_reason=reason,
    )


# --- kernel-spec dispatch ------------------------------------------------


def _build_pair(
    kernel: str,
    *,
    batch: int,
    seq: int,
    hidden: int,
    intermediate: int,
    eps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[nn.Module, nn.Module | None, str | None]:
    """Return (baseline_module, liger_module_or_None, skip_reason_or_None)."""
    if kernel == "rmsnorm":
        baseline = _build_rmsnorm_baseline(hidden, eps, dtype, device)
        try:
            liger = _build_rmsnorm_liger(hidden, eps, dtype, device)
            return baseline, liger, None
        except (ImportError, RuntimeError) as e:
            return baseline, None, f"liger import or build failed: {e!s}"
    if kernel == "swiglu":
        baseline = _build_swiglu_baseline(hidden, intermediate, dtype, device)
        try:
            liger = _build_swiglu_liger(hidden, intermediate, dtype, device)
            return baseline, liger, None
        except (ImportError, RuntimeError) as e:
            return baseline, None, f"liger import or build failed: {e!s}"
    if kernel in {"rope", "fused_linear_ce"}:
        # TODO(W1-Wed-followup): RoPE needs cos/sin caches; FusedLinearCE pairs
        # an lm_head Linear with target labels. Both need bespoke harnesses
        # rather than the (forward x, output y) shape used by RMSNorm/SwiGLU.
        # Stubbed to keep Wednesday scaffolding shippable.
        raise NotImplementedError(
            f"{kernel!r} runner harness not implemented yet; tracked as a Wednesday follow-up"
        )
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNEL_NAMES}")


# --- public entry --------------------------------------------------------


def run_liger_perkernel(
    *,
    kernel: str,
    batch: int,
    seq: int,
    hidden: int,
    intermediate: int,
    eps: float,
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
    """Run one paired (baseline, liger) per-kernel benchmark."""
    if kernel not in KERNEL_NAMES:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNEL_NAMES}")
    if dtype not in DTYPE_TO_TORCH:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(DTYPE_TO_TORCH)}")
    if min(batch, seq, hidden, intermediate, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "batch, seq, hidden, intermediate, repeats, and iters must be positive; "
            "warmup must be non-negative"
        )

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    torch_dtype = DTYPE_TO_TORCH[dtype]

    torch.manual_seed(seed)
    baseline_module, liger_module, build_skip_reason = _build_pair(
        kernel,
        batch=batch,
        seq=seq,
        hidden=hidden,
        intermediate=intermediate,
        eps=eps,
        dtype=torch_dtype,
        device=device,
    )

    def make_input() -> torch.Tensor:
        return _make_inputs_for_kernel(kernel, batch, seq, hidden, torch_dtype, device)

    with _track_peak_memory(device):
        baseline_outcome = _run_mode(
            module=baseline_module,
            make_input=make_input,
            repeats=repeats,
            warmup=warmup,
            iters=iters,
            device=device,
        )

    if liger_module is None:
        liger_outcome = _skipped_outcome(build_skip_reason or "liger not available")
    elif device.type != "cuda":
        liger_outcome = _skipped_outcome(
            "liger kernels require CUDA + Triton; running baseline only on CPU smoke"
        )
    else:
        with _track_peak_memory(device):
            liger_outcome = _run_mode(
                module=liger_module,
                make_input=make_input,
                repeats=repeats,
                warmup=warmup,
                iters=iters,
                device=device,
            )

    deltas = _compute_deltas(baseline_outcome, liger_outcome)

    env = capture_env(device, arch_label=arch_label)
    env["liger"] = _liger_version()

    result: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "benchmark": f"liger_perkernel_{kernel}",
        "config": {
            "scope": "liger_perkernel",
            "kernel": kernel,
            "shape": {
                "batch": batch,
                "seq": seq,
                "hidden": hidden,
                "intermediate": intermediate,
            },
            "dtype": dtype,
            "eps": eps,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "liger": {
                "applied": liger_module is not None and not liger_outcome.skipped,
                "version": _liger_version(),
                "kernel_module": _liger_kernel_module(kernel),
            },
        },
        "env": env,
        "correctness": {
            "baseline_finite": baseline_outcome.finite_output,
            "liger_finite": liger_outcome.finite_output if not liger_outcome.skipped else None,
            "checksum_baseline_fp32": baseline_outcome.output_checksum_fp32,
            "checksum_liger_fp32": (
                liger_outcome.output_checksum_fp32 if not liger_outcome.skipped else None
            ),
            "checksum_abs_delta": (
                abs(baseline_outcome.output_checksum_fp32 - liger_outcome.output_checksum_fp32)
                if not liger_outcome.skipped
                else None
            ),
        },
        "metrics": {
            "modes": {
                "baseline": _outcome_to_dict(baseline_outcome),
                "liger": _outcome_to_dict(liger_outcome),
            },
            "deltas": deltas,
        },
    }
    if ncu_csv is not None:
        result = attach_ncu_summary(result, ncu_csv)
    return result


def _liger_kernel_module(kernel: str) -> str | None:
    return {
        "rmsnorm": "liger_kernel.transformers.rms_norm.LigerRMSNorm",
        "swiglu": "liger_kernel.transformers.swiglu.LigerSwiGLUMLP",
    }.get(kernel)


def _outcome_to_dict(outcome: KernelOutcome) -> dict[str, Any]:
    return {
        "skipped": outcome.skipped,
        "skip_reason": outcome.skip_reason,
        "forward_ms": outcome.forward_ms,
        "backward_ms": outcome.backward_ms,
        "peak_gpu_mem_mb": outcome.peak_gpu_mem_mb,
        "finite_output": outcome.finite_output,
        "checksum_fp32": outcome.output_checksum_fp32,
    }


def _compute_deltas(baseline: KernelOutcome, liger: KernelOutcome) -> dict[str, Any]:
    if liger.skipped or not baseline.forward_ms or not liger.forward_ms:
        return {
            "forward_speedup": None,
            "backward_speedup": None,
            "peak_gpu_mem_ratio": None,
        }

    def _safe_ratio(num: float | None, den: float | None) -> float | None:
        if num is None or den is None or den == 0:
            return None
        return num / den

    fwd_speedup = _safe_ratio(baseline.forward_ms.get("mean_ms"), liger.forward_ms.get("mean_ms"))
    bwd_speedup = _safe_ratio(baseline.backward_ms.get("mean_ms"), liger.backward_ms.get("mean_ms"))
    mem_ratio = _safe_ratio(liger.peak_gpu_mem_mb, baseline.peak_gpu_mem_mb)
    return {
        "forward_speedup": fwd_speedup,
        "backward_speedup": bwd_speedup,
        "peak_gpu_mem_ratio": mem_ratio,
    }


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
