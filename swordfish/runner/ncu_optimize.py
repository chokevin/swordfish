"""Heuristic optimization analyzer for NCU profiles.

Given an `NcuSummary` (see ncu_summary.py), produce a short, actionable report
classifying the top kernels (compute-bound / memory-bound / underutilized /
already-near-peak) and surfacing job-level patterns (single-kernel hotspot,
"GEMM-bound; activations are rounding error", tail of negligible kernels).

Why this exists: per-kernel SM%/MEM%/DRAM% numbers are useful but unactionable
without context. The user wants "what should I optimize next?" not "here's a
table". This module is the same set of questions a kernel-engineering reviewer
would ask, automated and emitted on every profiled experiment.

Honest scope: heuristic, not a roofline. The default rune `ncu` capture does
not include FLOP/byte counters, so we can't compute arithmetic intensity. We
can read the SoL percentages NCU does report, plus kernel names, plus the
shape of the kernel distribution. That's enough for "where to focus" and
"what kind of bound it is", which is what the agent loop needs.

Decisions are derived from:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — overall SM utilization
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` — on-chip
  memory subsystem utilization (L1/L2/shared)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM bandwidth utilization
- kernel name heuristics (cuBLAS nvjet_, CUTLASS, ATen, Liger Triton, etc.)
- per-kernel time share + invocation count

The thresholds are deliberately conservative; we'd rather under-recommend
than send the user chasing false leads. They live as named module-level
constants so they're trivial to revisit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from swordfish.runner.ncu_summary import KernelStats, NcuSummary


# --- Thresholds. Conservative on purpose; tune from real workloads, not vibes.

# A kernel is "saturating compute" when SM is near peak. We use SM (not SM AND
# MEM) because cuBLAS/CUTLASS GEMMs typically run at SM 85-95% with MEM 60-75%
# — high MEM here is GEMM data movement, not the bottleneck. As long as DRAM
# isn't the wall (see DRAM_NOT_BOTTLENECK), the wall is the tensor cores.
SOL_NEAR_PEAK_SM = 80.0

# When DRAM is below this, external bandwidth isn't the bottleneck.
DRAM_NOT_BOTTLENECK = 40.0

# Below this, the SoL channel isn't the bottleneck (cheaply ignorable).
SOL_LOW = 30.0

# Compute-bound thresholds: SM is leading meaningfully and DRAM isn't the wall.
SM_BOUND = 60.0
SM_OVER_MEM_DELTA = 15.0  # SM must lead MEM by this many percentage points

# Memory-bound (on-chip) — MEM saturated but SM not, and DRAM not the wall.
MEM_BOUND = 70.0
MEM_BOUND_OPP_MAX = 50.0  # ... and SM% must be below this

# DRAM "we're hitting external bandwidth" threshold.
DRAM_BOUND = 60.0

# Single-kernel hotspot: % of total time owned by the top kernel.
HOTSPOT_DOMINANT = 80.0  # one kernel owns the whole job
HOTSPOT_PRIMARY = 60.0  # top kernel matters most but not the only thing

# A kernel is "negligible" for tuning purposes below this share.
NEGLIGIBLE_PCT = 1.0

# Number of top kernels we deeply analyze (the rest are summarized en masse).
DEFAULT_TOP_KERNELS = 5


# --- Engine metric names (kept in sync with ncu_summary._FRIENDLY_TO_ENGINE_METRIC).

_M_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_M_MEM = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
_M_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
_M_TIME = "gpu__time_duration.sum"


class Boundedness(str, Enum):
    """Coarse classification of where a kernel spends its time."""

    NEAR_PEAK = "near-peak"  # >SOL_NEAR_PEAK on both SM and on-chip MEM
    COMPUTE_BOUND = "compute-bound"  # SM high, MEM low
    MEMORY_BOUND_DRAM = "memory-bound (DRAM)"  # MEM high AND DRAM high
    MEMORY_BOUND_ONCHIP = "memory-bound (on-chip)"  # MEM high but DRAM low
    UNDERUTILIZED = "underutilized"  # both SM and MEM low
    MIXED = "mixed / inconclusive"
    UNKNOWN = "unknown (no SoL metrics)"


@dataclass(frozen=True)
class KernelAdvice:
    """One kernel's classification + targeted suggestions."""

    kernel: KernelStats
    pct_of_total: float  # share of profiled wall time
    sm_pct: float | None
    mem_pct: float | None
    dram_pct: float | None
    bound: Boundedness
    suggestions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizationReport:
    """Per-job analysis of an NcuSummary, ready to print."""

    summary: NcuSummary
    job_findings: list[str]  # job-level observations (hotspot pattern, tail, etc.)
    kernel_advice: list[KernelAdvice]  # top-N kernels, in time-descending order
    negligible_kernel_count: int  # how many kernels were below NEGLIGIBLE_PCT

    @property
    def top_kernel(self) -> KernelStats | None:
        return self.summary.kernels[0] if self.summary.kernels else None


# --- Kernel-name pattern detection. Used to give name-aware suggestions.


def _is_gemm_like(name: str) -> bool:
    """True for kernels that are matmul-shaped: cuBLAS heuristics, CUTLASS,
    cuDNN's xmma family. We don't try to be exhaustive — false negatives just
    mean we fall back to generic boundedness advice.
    """
    n = name.lower()
    return (
        "gemm" in n
        or n.startswith("nvjet_")  # cuBLAS heuristic GEMM family on H100/H200
        or "xmma" in n  # cuBLAS/cuDNN tensor-core kernels
        or "cutlass" in n
    )


def _is_aten_elementwise(name: str) -> bool:
    """ATen-style elementwise/reduce kernels — fusion candidates."""
    n = name.lower()
    return (
        "vectorized_elementwise_kernel" in n
        or "unrolled_elementwise_kernel" in n
        or "distribution_elementwise" in n
        or "reduce_kernel" in n
    )


def _is_triton(name: str) -> bool:
    """Triton/Liger-compiled kernels typically carry the JIT-compiled tag."""
    n = name.lower()
    return n.startswith("triton_") or "liger" in n or "_kernel_0d1d" in n


# --- Classification + advice generation.


def _classify(sm: float | None, mem: float | None, dram: float | None) -> Boundedness:
    """Bucket a kernel from its three SoL percentages.

    Decision order is significant: we check the strongest positive cases first
    (NEAR_PEAK / DRAM_BOUND) before falling through to the softer buckets.

    Calibration notes (real numbers from cluster runs):
    - cuBLAS GEMM on H100/H200: SM≈91, MEM≈70, DRAM≈16 → NEAR_PEAK
    - cuBLAS GEMM on A100:      SM≈88, MEM≈47, DRAM≈16 → NEAR_PEAK
    - ATen reduce_kernel:       SM≈19, MEM≈73, DRAM≈73 → MEMORY_BOUND_DRAM
    - elementwise distribution: SM≈74, MEM≈12, DRAM≈3  → COMPUTE_BOUND
    """
    if sm is None and mem is None and dram is None:
        return Boundedness.UNKNOWN

    sm_v = sm or 0.0
    mem_v = mem or 0.0
    dram_v = dram or 0.0

    # External bandwidth is the wall — nothing else matters.
    if dram_v >= DRAM_BOUND:
        return Boundedness.MEMORY_BOUND_DRAM

    # Compute is saturated and DRAM isn't the bottleneck → near roofline.
    if sm_v >= SOL_NEAR_PEAK_SM and dram_v <= DRAM_NOT_BOTTLENECK:
        return Boundedness.NEAR_PEAK

    # Leaning compute: SM clearly leads MEM, DRAM not the wall.
    if (
        sm_v >= SM_BOUND
        and (sm_v - mem_v) >= SM_OVER_MEM_DELTA
        and dram_v <= DRAM_NOT_BOTTLENECK
    ):
        return Boundedness.COMPUTE_BOUND

    # On-chip memory pressure: MEM saturated, SM low, DRAM not the wall.
    if mem_v >= MEM_BOUND and sm_v <= MEM_BOUND_OPP_MAX:
        return Boundedness.MEMORY_BOUND_ONCHIP

    if sm_v < SOL_LOW and mem_v < SOL_LOW:
        return Boundedness.UNDERUTILIZED

    return Boundedness.MIXED


def _suggest(
    kernel: KernelStats,
    bound: Boundedness,
    sm: float | None,
    mem: float | None,
    dram: float | None,
    pct_of_total: float,
) -> list[str]:
    """Concrete, kernel-aware suggestions. Each item is one short sentence."""
    out: list[str] = []
    name = kernel.name
    short = kernel.short_name

    if pct_of_total < NEGLIGIBLE_PCT:
        out.append(
            f"only {pct_of_total:.2f}% of time — likely setup/overhead, not a tuning target."
        )
        return out

    if bound is Boundedness.NEAR_PEAK:
        if _is_gemm_like(name):
            out.append(
                f"compute-saturated GEMM (SM={sm or 0:.0f}%, DRAM={dram or 0:.0f}%): "
                "cuBLAS/CUTLASS picked a healthy tensor-core path. To go faster: "
                "smaller dtype (bf16/fp8 on H100/H200), structured sparsity, "
                "grouped/fused GEMMs, or epilogue fusion."
            )
        else:
            out.append(
                f"near peak SM utilization ({sm or 0:.0f}%) with DRAM not the wall "
                f"({dram or 0:.0f}%): tensor cores saturated, further gains require "
                "an algorithm change (smaller dtype, sparsity, or rewrite as fewer ops)."
            )
        return out

    if bound is Boundedness.COMPUTE_BOUND:
        if _is_gemm_like(name):
            out.append(
                "compute-bound GEMM, healthy: cuBLAS/CUTLASS already picked a "
                "tensor-core path. To go faster: smaller dtype (bf16/fp8), "
                "structured sparsity, or fewer GEMMs (e.g., grouped/fused)."
            )
        else:
            out.append(
                "compute-bound but not on the tensor cores: check whether the "
                "kernel can be expressed as a GEMM/conv (cuBLAS/CUTLASS), or "
                "verify it's using fp16/bf16 not fp32 accumulators."
            )
        return out

    if bound is Boundedness.MEMORY_BOUND_DRAM:
        out.append(
            f"DRAM-bound at {dram or 0.0:.0f}% of peak: tile to reuse data in "
            "shared/L2, fuse with neighbors to avoid round trips, vectorize "
            "global loads (cp.async / float4)."
        )
        if _is_aten_elementwise(name):
            out.append(
                "ATen elementwise/reduce kernel — strong fusion candidate; "
                "torch.compile or a Triton/Liger kernel that fuses with the "
                "adjacent op would eliminate a DRAM round-trip."
            )
        return out

    if bound is Boundedness.MEMORY_BOUND_ONCHIP:
        out.append(
            "on-chip memory pressure (L1/L2/shared/regs), but DRAM is fine: "
            "inspect register count and shared-mem usage; consider smaller "
            "tiles, fewer in-flight values, or reordering to shorten live ranges."
        )
        return out

    if bound is Boundedness.UNDERUTILIZED:
        if pct_of_total < 5.0:
            out.append(
                "low utilization but also low share of total time — likely a "
                "small/setup kernel; not worth chasing unless invocation count is high."
            )
        else:
            out.append(
                "low SM and memory utilization at non-trivial time share: "
                "likely launch-bound or grid-too-small. Try larger problem "
                "size, batched launch, or fewer/larger blocks."
            )
        if kernel.invocations > 50:
            out.append(
                f"{kernel.invocations} invocations — if these are identical, "
                "consider CUDA graphs or a single batched kernel."
            )
        return out

    if bound is Boundedness.MIXED:
        out.append(
            f"SoL split across compute and memory (SM={sm or 0:.0f}%, "
            f"MEM={mem or 0:.0f}%, DRAM={dram or 0:.0f}%); profile in ncu-ui to "
            "see which subsystem is the actual stall reason."
        )
        return out

    # UNKNOWN
    out.append(
        "no SoL metrics in this report — re-capture with default sections "
        "(rune `--profile-mode ncu`) or add `--section SpeedOfLight` to ncu."
    )
    return out


def _job_findings(
    summary: NcuSummary,
    advice: list[KernelAdvice],
    negligible_count: int,
) -> list[str]:
    """High-level patterns about the whole job — not per-kernel."""
    out: list[str] = []
    if not summary.kernels:
        out.append("no kernels parsed; nothing to analyze.")
        return out
    if summary.total_time_ns <= 0:
        out.append(
            "total kernel time is zero — the report likely lacks the Duration "
            "metric; re-capture with `--page details` or default sections."
        )
        return out

    top = summary.kernels[0]
    top_pct = top.total_time_ns / summary.total_time_ns * 100.0

    if top_pct >= HOTSPOT_DOMINANT:
        out.append(
            f"single-kernel hotspot: `{top.short_name}` owns {top_pct:.1f}% "
            f"of profiled time across {top.invocations} invocations. All "
            "tuning effort should target this kernel; everything else is "
            "rounding error."
        )
    elif top_pct >= HOTSPOT_PRIMARY:
        out.append(
            f"dominant kernel: `{top.short_name}` is {top_pct:.1f}% of time. "
            "Focus there first; secondary kernels likely won't move the needle "
            "until the top one is addressed."
        )

    # GEMM-bound + ATen elementwise tail pattern.
    if _is_gemm_like(top.name) and top_pct >= 60.0:
        elementwise_tail = sum(
            1
            for k in summary.kernels[1:6]
            if _is_aten_elementwise(k.name)
            and (k.total_time_ns / summary.total_time_ns * 100.0) < 5.0
        )
        if elementwise_tail >= 2:
            out.append(
                "GEMM-bound workload: matmul dominates and the surrounding "
                "ATen elementwise kernels are each <5% of time. Fusing the "
                "activation/normalization into the GEMM epilogue (or via "
                "torch.compile/Liger) is rounding-error optimization compared "
                "to anything that speeds up the matmul itself."
            )

    if negligible_count >= 5:
        out.append(
            f"{negligible_count} kernels each contribute <{NEGLIGIBLE_PCT:.0f}% "
            "of time. If these are repeated launches of small kernels, CUDA "
            "graphs or a single fused kernel would amortize launch overhead."
        )

    # Diagnostic: did NCU capture the SoL metrics at all?
    have_sol = any(
        a.sm_pct is not None or a.mem_pct is not None or a.dram_pct is not None
        for a in advice
    )
    if not have_sol:
        out.append(
            "no SoL (SM%/MEM%/DRAM%) metrics in this report — re-capture with "
            "rune's default `--profile-mode ncu` (which includes SpeedOfLight) "
            "to enable boundedness analysis."
        )
    return out


def analyze_ncu_summary(
    summary: NcuSummary, top_kernels: int = DEFAULT_TOP_KERNELS
) -> OptimizationReport:
    """Build an OptimizationReport from a parsed NcuSummary.

    `top_kernels` controls how many kernels (by total time) get individual
    advice; the rest are still counted in the negligible-kernel summary.
    """
    advice: list[KernelAdvice] = []
    negligible = 0

    for k in summary.kernels:
        pct = (
            k.total_time_ns / summary.total_time_ns * 100.0
            if summary.total_time_ns > 0
            else 0.0
        )
        if pct < NEGLIGIBLE_PCT:
            negligible += 1

    for k in summary.kernels[:top_kernels]:
        sm = k.metrics.get(_M_SM)
        mem = k.metrics.get(_M_MEM)
        dram = k.metrics.get(_M_DRAM)
        sm_v = sm.mean if sm else None
        mem_v = mem.mean if mem else None
        dram_v = dram.mean if dram else None
        pct = (
            k.total_time_ns / summary.total_time_ns * 100.0
            if summary.total_time_ns > 0
            else 0.0
        )
        bound = _classify(sm_v, mem_v, dram_v)
        suggestions = _suggest(k, bound, sm_v, mem_v, dram_v, pct)
        advice.append(
            KernelAdvice(
                kernel=k,
                pct_of_total=pct,
                sm_pct=sm_v,
                mem_pct=mem_v,
                dram_pct=dram_v,
                bound=bound,
                suggestions=suggestions,
            )
        )

    findings = _job_findings(summary, advice, negligible)
    return OptimizationReport(
        summary=summary,
        job_findings=findings,
        kernel_advice=advice,
        negligible_kernel_count=negligible,
    )


def _bullets(items: Iterable[str], indent: str = "  ") -> list[str]:
    """Wrap each item with a bullet, preserve ordering. No text-wrapping; we
    keep line lengths reasonable in the source instead."""
    return [f"{indent}- {item}" for item in items]


def format_optimization_report(report: OptimizationReport) -> str:
    """Render an OptimizationReport as plain text suitable for stdout.

    Layout:
      Optimization report: <path>
        - job-level finding 1
        - job-level finding 2
      Top N kernels:
        1. <short_name>  (pct%, SM/MEM/DRAM)  bound=<class>
           - suggestion 1
           - suggestion 2
    """
    lines: list[str] = []
    lines.append(f"Optimization report: {report.summary.path}")
    if report.job_findings:
        lines.extend(_bullets(report.job_findings))
    else:
        lines.append("  - no notable job-level patterns.")

    if not report.kernel_advice:
        lines.append("")
        lines.append("No kernel-level advice (no kernels parsed).")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Top {len(report.kernel_advice)} kernels:")
    for i, a in enumerate(report.kernel_advice, start=1):
        sm = f"{a.sm_pct:.0f}%" if a.sm_pct is not None else "—"
        mem = f"{a.mem_pct:.0f}%" if a.mem_pct is not None else "—"
        dram = f"{a.dram_pct:.0f}%" if a.dram_pct is not None else "—"
        short = a.kernel.short_name
        if len(short) > 80:
            short = short[:79] + "…"
        lines.append(
            f"  {i}. {short}"
        )
        lines.append(
            f"     {a.pct_of_total:.1f}% of time  "
            f"({a.kernel.invocations} invocations, "
            f"SM={sm} MEM={mem} DRAM={dram})  "
            f"bound={a.bound.value}"
        )
        for s in a.suggestions:
            lines.append(f"       - {s}")
    return "\n".join(lines)
