"""Unit tests for swordfish.runner.ncu_optimize.

The classifier and report formatter are pure functions over an `NcuSummary`,
so these tests build small synthetic summaries instead of going through the
CSV parser. Calibrated against real cluster traces (see _classify docstring
in ncu_optimize.py for the source numbers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swordfish.runner.ncu_optimize import (
    Boundedness,
    _classify,
    _is_aten_elementwise,
    _is_gemm_like,
    analyze_ncu_summary,
    format_optimization_report,
)
from swordfish.runner.ncu_summary import KernelStats, MetricStats, NcuSummary


def _metric(name: str, value: float, unit: str = "%") -> MetricStats:
    return MetricStats(
        name=name,
        unit=unit,
        samples=1,
        mean=value,
        median=value,
        max=value,
        p99=value,
    )


def _kernel(
    name: str,
    total_ns: float,
    *,
    invocations: int = 1,
    sm: float | None = None,
    mem: float | None = None,
    dram: float | None = None,
) -> KernelStats:
    metrics: dict[str, MetricStats] = {}
    if sm is not None:
        metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] = _metric(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed", sm
        )
    if mem is not None:
        metrics[
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
        ] = _metric(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            mem,
        )
    if dram is not None:
        metrics["dram__throughput.avg.pct_of_peak_sustained_elapsed"] = _metric(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed", dram
        )
    return KernelStats(
        name=name,
        short_name=name,
        invocations=invocations,
        block_size="(256,1,1)",
        grid_size="(1,1,1)",
        total_time_ns=total_ns,
        mean_time_ns=total_ns / max(invocations, 1),
        max_time_ns=total_ns / max(invocations, 1),
        metrics=metrics,
    )


def _summary(kernels: list[KernelStats], path: str = "/tmp/test.csv") -> NcuSummary:
    total = sum(k.total_time_ns for k in kernels)
    inv = sum(k.invocations for k in kernels)
    return NcuSummary(
        path=Path(path),
        rows=len(kernels) * 4,
        unique_kernels=len(kernels),
        total_invocations=inv,
        total_time_ns=total,
        kernels=sorted(kernels, key=lambda k: k.total_time_ns, reverse=True),
        parse_warnings=[],
    )


# --- Classification calibration tests. Numbers from real cluster runs.


@pytest.mark.parametrize(
    "sm,mem,dram,expected",
    [
        # cuBLAS GEMM on H100/H200 — SM saturated, DRAM not the wall.
        (91.0, 70.0, 16.0, Boundedness.NEAR_PEAK),
        # cuBLAS GEMM on A100 — same shape.
        (88.0, 47.0, 16.0, Boundedness.NEAR_PEAK),
        # ATen reduce — DRAM-bound.
        (19.0, 73.0, 73.0, Boundedness.MEMORY_BOUND_DRAM),
        # Elementwise distribution — compute-leaning, low memory pressure.
        (74.0, 12.0, 3.0, Boundedness.COMPUTE_BOUND),
        # On-chip memory pressure — high MEM but DRAM is fine.
        (30.0, 75.0, 20.0, Boundedness.MEMORY_BOUND_ONCHIP),
        # Both low — underutilized.
        (15.0, 12.0, 8.0, Boundedness.UNDERUTILIZED),
        # No metrics at all → unknown.
        (None, None, None, Boundedness.UNKNOWN),
    ],
)
def test_classify_real_world_calibration(sm, mem, dram, expected):
    """The classifier must agree with the cluster-run numbers documented in
    its docstring. If you change a threshold, this test is the canary."""
    assert _classify(sm, mem, dram) == expected


def test_classify_dram_dominates_other_signals():
    """DRAM at the wall trumps everything else: even if SM is high, if DRAM is
    pegged we are DRAM-bound and tuning compute won't help."""
    assert _classify(85.0, 90.0, 65.0) == Boundedness.MEMORY_BOUND_DRAM


# --- Job-level finding tests.


def test_single_kernel_hotspot_finding_when_top_owns_most_time():
    """If one kernel owns 99%, the report should say 'all effort here'."""
    kernels = [
        _kernel("nvjet_hsh_256x128", 99e6, sm=91, mem=70, dram=16),
        _kernel("at::vectorized_elementwise_kernel", 1e6, sm=15, mem=76, dram=76),
    ]
    report = analyze_ncu_summary(_summary(kernels))
    text = "\n".join(report.job_findings)
    assert "single-kernel hotspot" in text
    assert "99.0%" in text
    assert "nvjet_hsh_256x128" in text


def test_gemm_bound_pattern_with_elementwise_tail():
    """The GEMM-bound pattern must trigger when a GEMM dominates and the
    next few kernels are ATen elementwise — the user's "rounding error"
    callout."""
    kernels = [
        _kernel("nvjet_hsh_256x128", 90e6, sm=91, mem=70, dram=16),
        _kernel("at::vectorized_elementwise_kernel", 4e6, sm=20, mem=70, dram=70),
        _kernel("at::reduce_kernel<512>", 3e6, sm=15, mem=73, dram=73),
        _kernel("at::unrolled_elementwise_kernel", 2e6, sm=18, mem=68, dram=65),
        _kernel("at::distribution_elementwise_grid_stride_kernel", 1e6, sm=70, mem=12, dram=8),
    ]
    report = analyze_ncu_summary(_summary(kernels))
    text = "\n".join(report.job_findings)
    assert "GEMM-bound" in text
    assert "rounding-error" in text


def test_no_gemm_pattern_when_top_kernel_is_elementwise():
    """If the top kernel isn't GEMM-shaped, don't claim GEMM-bound."""
    kernels = [
        _kernel("at::reduce_kernel<512>", 80e6, sm=15, mem=73, dram=73),
        _kernel("at::vectorized_elementwise_kernel", 20e6, sm=20, mem=70, dram=70),
    ]
    report = analyze_ncu_summary(_summary(kernels))
    text = "\n".join(report.job_findings)
    assert "GEMM-bound" not in text


def test_negligible_kernel_count_finding():
    """If 5+ kernels each own <1% of time, the report should suggest CUDA
    graphs / fusion to amortize launch overhead."""
    big = _kernel("hot", 950e6, sm=91, mem=70, dram=16)
    smalls = [_kernel(f"tiny_{i}", 1e6, sm=20, mem=20, dram=20) for i in range(7)]
    report = analyze_ncu_summary(_summary([big, *smalls]))
    text = "\n".join(report.job_findings)
    assert "7 kernels each contribute <1%" in text
    assert "CUDA graphs" in text
    assert report.negligible_kernel_count == 7


def test_no_sol_metrics_finding_when_csv_lacks_them():
    """If the CSV has no SoL metrics at all, the report must say so —
    otherwise the user gets per-kernel "unknown" advice with no explanation."""
    kernels = [_kernel("gemm_no_metrics", 100e6)]  # no sm/mem/dram
    report = analyze_ncu_summary(_summary(kernels))
    text = "\n".join(report.job_findings)
    assert "no SoL" in text


# --- Per-kernel suggestion tests.


def test_near_peak_gemm_suggests_dtype_and_grouping():
    """A near-peak GEMM should get the dtype/sparsity/grouping suggestion,
    not the generic 'algorithm change' line."""
    k = _kernel("nvjet_hsh_256x128", 100e6, sm=91, mem=70, dram=16)
    report = analyze_ncu_summary(_summary([k]))
    advice = report.kernel_advice[0]
    assert advice.bound is Boundedness.NEAR_PEAK
    text = "\n".join(advice.suggestions)
    assert "tensor-core" in text or "tensor core" in text
    assert "bf16" in text or "fp8" in text


def test_dram_bound_suggests_tiling_and_fusion():
    """DRAM-bound kernels should get tiling + fusion advice; ATen elementwise
    should additionally get the torch.compile/Liger fusion call-out."""
    k = _kernel("at::vectorized_elementwise_kernel", 100e6, sm=15, mem=76, dram=76)
    report = analyze_ncu_summary(_summary([k]))
    text = "\n".join(report.kernel_advice[0].suggestions)
    assert "DRAM-bound" in text
    assert "fusion" in text or "fuse" in text
    assert "torch.compile" in text or "Liger" in text


def test_underutilized_with_many_invocations_suggests_cuda_graphs():
    """Low SoL + many invocations is the classic launch-bound case; suggest
    CUDA graphs or batching."""
    k = _kernel("setup_kernel", 100e6, sm=10, mem=10, dram=10, invocations=200)
    report = analyze_ncu_summary(_summary([k]))
    text = "\n".join(report.kernel_advice[0].suggestions)
    assert "CUDA graphs" in text or "batched kernel" in text


def test_negligible_kernel_gets_short_dismissal():
    """If a top-N kernel happens to be negligible (<1% of time), don't waste
    the user's attention — say so and stop."""
    big = _kernel("hot", 999e6, sm=91, mem=70, dram=16)
    small = _kernel("tiny", 1e5, sm=15, mem=15, dram=15)
    report = analyze_ncu_summary(_summary([big, small]), top_kernels=2)
    second = report.kernel_advice[1]
    assert second.kernel.short_name == "tiny"
    assert any("setup/overhead" in s for s in second.suggestions)


# --- Pattern detector tests.


@pytest.mark.parametrize(
    "name,expected",
    [
        ("nvjet_hsh_256x128_64x4_1x2_h_bz_coopA_NNT", True),
        ("ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nn", True),
        ("cutlass::Kernel<...>", True),
        ("sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize128x128", True),
        ("at::vectorized_elementwise_kernel", False),
        ("at::reduce_kernel<512>", False),
    ],
)
def test_is_gemm_like(name, expected):
    assert _is_gemm_like(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("at::vectorized_elementwise_kernel", True),
        ("at::unrolled_elementwise_kernel<at::direct_copy_kernel_cuda>", True),
        ("at::distribution_elementwise_grid_stride_kernel", True),
        ("at::reduce_kernel<512>", True),
        ("nvjet_hsh_256x128", False),
        ("ampere_fp16_s16816gemm", False),
    ],
)
def test_is_aten_elementwise(name, expected):
    assert _is_aten_elementwise(name) is expected


# --- Formatter test (smoke: no crashes, output contains expected sections).


def test_format_optimization_report_renders_full_text():
    kernels = [
        _kernel("nvjet_hsh_256x128", 90e6, invocations=14, sm=91, mem=70, dram=16),
        _kernel("at::vectorized_elementwise_kernel", 5e6, sm=20, mem=70, dram=70),
        _kernel("at::reduce_kernel<512>", 3e6, sm=15, mem=73, dram=73),
        _kernel("at::unrolled_elementwise_kernel", 2e6, sm=20, mem=68, dram=65),
    ]
    report = analyze_ncu_summary(_summary(kernels))
    text = format_optimization_report(report)
    assert text.startswith("Optimization report:")
    assert "nvjet_hsh_256x128" in text
    assert "Top " in text
    assert "bound=" in text
    assert "% of time" in text


def test_empty_summary_does_not_crash():
    """A summary with no kernels (e.g. parser found header but zero rows)
    must still produce a valid report — used by the inspect-run fallback path."""
    empty = NcuSummary(
        path=Path("/tmp/empty.csv"),
        rows=0,
        unique_kernels=0,
        total_invocations=0,
        total_time_ns=0.0,
        kernels=[],
    )
    report = analyze_ncu_summary(empty)
    text = format_optimization_report(report)
    assert "no kernels" in text.lower()
