"""Rich per-kernel summarizer for Nsight Compute CSV exports.

Why this exists: `swordfish.runner.schema.parse_ncu_csv` collapses every NCU
CSV into 4 aggregate numbers attached to the result JSON. That's enough for
"is the matmul memory-bound on H100?" but useless for kernel-tuning iteration
where you need per-kernel detail (top kernels by time, per-kernel SoL%,
distribution across invocations).

This module produces the richer view: for each unique kernel, group every
invocation and every metric, then summarize. Output is a typed dataclass and
a text pretty-printer suitable for stdout / piping into a markdown report.

Limitations (be honest):
- This reads NCU's CSV export, NOT the `.ncu-rep` binary. To get a CSV from
  a `.ncu-rep` you need NVIDIA's `ncu` CLI installed somewhere — locally
  (`ncu --import file.ncu-rep --csv`) or on the cluster (already there).
  Until cluster-side conversion lands for `--profile-mode=ncu` jobs, this
  module only works on legacy `SWORDFISH_PROFILE=ncu` CSV outputs (which
  includes the existing torch-gemm-{a100,h100,h200}.ncu.csv week-1 fixtures).
- Source-line attribution, SASS view, occupancy widget, and the full
  Speed-of-Light radial chart live only in `ncu-ui` against the `.ncu-rep`.
  This summarizer is the agent-readable substitute, not a replacement.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Demangler: NCU emits the full C++ mangled name. We want the function path
# (`namespace::function`) without template-argument noise.
#
# Strategy:
#   1. strip leading "void " return type
#   2. flatten "<unnamed>::" / "(anonymous namespace)::" inserts
#   3. cut off everything from the first "(" onward (the call signature)
#   4. iteratively strip balanced "<...>" template-arg lists
#
# Handles the two common shapes in our CSVs:
#   - cuBLAS-style: "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN"  →  unchanged
#   - PyTorch ATen: "void at::<unnamed>::distribution_elementwise_grid_stride_kernel<float, 4, ...>(...)"
#                                          ↓
#                   "at::distribution_elementwise_grid_stride_kernel"
_TEMPLATE_RE = re.compile(r"<[^<>]*>")


def _short_name(mangled: str) -> str:
    """Best-effort short kernel name. Falls back to a 80-char prefix."""
    s = mangled.strip()
    if s.startswith("void "):
        s = s[5:]
    s = s.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    paren = s.find("(")
    if paren > 0:
        s = s[:paren]
    # Iteratively strip the innermost balanced <...> until none remain.
    # Bounded loop guard: kernel names shouldn't have >32 nested templates.
    for _ in range(32):
        new = _TEMPLATE_RE.sub("", s)
        if new == s:
            break
        s = new
    s = s.strip()
    return s or mangled[:80]


@dataclass(frozen=True)
class MetricStats:
    """Per-metric aggregate across one kernel's invocations."""

    name: str
    unit: str
    samples: int
    mean: float
    median: float
    max: float
    p99: float


@dataclass(frozen=True)
class KernelStats:
    """All measurements for one unique kernel name."""

    name: str
    short_name: str
    invocations: int
    block_size: str  # CSV reports it as a tuple-string e.g. "(256, 1, 1)"
    grid_size: str
    total_time_ns: float
    mean_time_ns: float
    max_time_ns: float
    metrics: dict[str, MetricStats]


@dataclass(frozen=True)
class NcuSummary:
    """Top-level summary of one NCU CSV file."""

    path: Path
    rows: int
    unique_kernels: int
    total_invocations: int
    total_time_ns: float
    kernels: list[KernelStats]  # sorted by total_time_ns desc
    parse_warnings: list[str] = field(default_factory=list)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Strip ==PROF== preamble lines and return DictReader rows.

    Mirrors swordfish.runner.schema.parse_ncu_csv's header-finding logic so
    both parsers tolerate the same CSV shapes.
    """
    lines = path.read_text().splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith('"ID"') or '"Kernel Name"' in line),
        None,
    )
    if header_idx is None:
        return []
    return list(csv.DictReader(lines[header_idx:]))


def _percentile(values: list[float], pct: float) -> float:
    """Inclusive linear-interpolation percentile (matches numpy default).

    Pure-stdlib (no numpy import in the runner) because this module needs to
    work in the bench container without optional dependencies.
    """
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _parse_float(raw: str) -> float | None:
    cleaned = (raw or "").strip().replace(",", "")
    if cleaned in {"", "n/a", "N/A", "--"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ncu_csv_full(path: Path) -> NcuSummary:
    """Parse an NCU CSV export into per-kernel + per-metric statistics.

    The CSV is long-form: one row per (kernel invocation × metric). We pivot
    by (kernel name, invocation id) and group by kernel name for aggregates.

    Robust against:
    - ==PROF== preamble lines from `ncu --csv`.
    - Mangled C++ kernel names with embedded commas (csv module handles).
    - Missing or non-numeric metric values (skipped, recorded as parse_warnings).
    - The `gpu__time_duration.sum` metric being either present or absent
      (drives total time when present; falls back to 0).

    Returns an NcuSummary with kernels sorted by total_time_ns descending.
    """
    rows = _read_csv_rows(path)
    if not rows:
        return NcuSummary(
            path=path,
            rows=0,
            unique_kernels=0,
            total_invocations=0,
            total_time_ns=0.0,
            kernels=[],
            parse_warnings=["no header row found in CSV"],
        )

    # First pass: collect raw values per (kernel, invocation_id, metric).
    # Inv id is the "ID" column NCU emits per kernel launch.
    invocations: dict[str, dict[str, dict[str, tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    block_grid: dict[str, tuple[str, str]] = {}
    warnings: list[str] = []

    for row in rows:
        kname = (row.get("Kernel Name") or "").strip()
        if not kname:
            continue
        inv_id = (row.get("ID") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        unit = (row.get("Metric Unit") or "").strip()
        value = _parse_float(row.get("Metric Value", ""))
        if value is None:
            warnings.append(
                f"row id={inv_id!r} kernel={_short_name(kname)} metric={metric}: "
                f"non-numeric value {row.get('Metric Value')!r}"
            )
            continue
        invocations[kname][inv_id][metric] = (unit, value)
        block_grid.setdefault(kname, (row.get("Block Size") or "", row.get("Grid Size") or ""))

    kernels: list[KernelStats] = []
    grand_total_time = 0.0
    total_invocations = 0

    for kname, inv_map in invocations.items():
        # Per-metric series across invocations of THIS kernel.
        per_metric: dict[str, tuple[str, list[float]]] = defaultdict(lambda: ("", []))
        time_series: list[float] = []
        for _inv_id, metrics in inv_map.items():
            if "gpu__time_duration.sum" in metrics:
                _, t = metrics["gpu__time_duration.sum"]
                time_series.append(t)
            for mname, (unit, val) in metrics.items():
                bucket = per_metric[mname]
                if not bucket[0]:
                    per_metric[mname] = (unit, bucket[1])
                per_metric[mname][1].append(val)

        metric_stats: dict[str, MetricStats] = {}
        for mname, (unit, vals) in per_metric.items():
            if not vals:
                continue
            metric_stats[mname] = MetricStats(
                name=mname,
                unit=unit,
                samples=len(vals),
                mean=statistics.fmean(vals),
                median=statistics.median(vals),
                max=max(vals),
                p99=_percentile(vals, 99),
            )

        invocation_count = len(inv_map)
        total_time = sum(time_series) if time_series else 0.0
        mean_time = statistics.fmean(time_series) if time_series else 0.0
        max_time = max(time_series) if time_series else 0.0
        block, grid = block_grid.get(kname, ("", ""))

        kernels.append(
            KernelStats(
                name=kname,
                short_name=_short_name(kname),
                invocations=invocation_count,
                block_size=block,
                grid_size=grid,
                total_time_ns=total_time,
                mean_time_ns=mean_time,
                max_time_ns=max_time,
                metrics=metric_stats,
            )
        )
        grand_total_time += total_time
        total_invocations += invocation_count

    kernels.sort(key=lambda k: k.total_time_ns, reverse=True)

    return NcuSummary(
        path=path,
        rows=len(rows),
        unique_kernels=len(kernels),
        total_invocations=total_invocations,
        total_time_ns=grand_total_time,
        kernels=kernels,
        parse_warnings=warnings,
    )


# Metrics surfaced in the per-kernel table. Kept short for line-length sanity.
_DISPLAY_METRICS: tuple[tuple[str, str], ...] = (
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM%"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "MEM%"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM%"),
)


def _fmt_ns(ns: float) -> str:
    """Human-readable duration. CSV time_duration is in nanoseconds."""
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.0f} ns"


def format_summary_text(
    summary: NcuSummary,
    top_n: int = 10,
    short_name_width: int = 50,
) -> str:
    """Render an NcuSummary as a plain-text table for stdout.

    Includes a header line (file, total wall, kernel count), then a per-kernel
    table sorted by total time descending. Each row shows: short kernel name,
    invocation count, total time, % of total, mean/max time, and the headline
    SoL metrics (SM%, MEM%, DRAM% means).
    """
    lines: list[str] = []
    lines.append(f"NCU summary: {summary.path}")
    lines.append(
        f"  rows={summary.rows}  unique_kernels={summary.unique_kernels}  "
        f"invocations={summary.total_invocations}  "
        f"total_time={_fmt_ns(summary.total_time_ns)}"
    )
    if summary.parse_warnings:
        lines.append(f"  warnings: {len(summary.parse_warnings)} (showing first 3)")
        for w in summary.parse_warnings[:3]:
            lines.append(f"    - {w}")
    lines.append("")

    header = (
        f"{'kernel':<{short_name_width}}  "
        f"{'invs':>5}  {'total':>10}  {'%':>5}  {'mean':>10}  {'max':>10}  "
        f"{'SM%':>6}  {'MEM%':>6}  {'DRAM%':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    if summary.total_time_ns <= 0:
        pct_div = float("nan")
    else:
        pct_div = summary.total_time_ns

    for k in summary.kernels[:top_n]:
        sm = k.metrics.get(_DISPLAY_METRICS[0][0])
        mem = k.metrics.get(_DISPLAY_METRICS[1][0])
        dram = k.metrics.get(_DISPLAY_METRICS[2][0])
        pct = (k.total_time_ns / pct_div * 100.0) if pct_div else float("nan")

        def _f(s: MetricStats | None) -> str:
            if s is None:
                return "—".rjust(6)
            return f"{s.mean:>6.1f}"

        short = k.short_name
        if len(short) > short_name_width:
            short = short[: short_name_width - 1] + "…"
        lines.append(
            f"{short:<{short_name_width}}  "
            f"{k.invocations:>5}  {_fmt_ns(k.total_time_ns):>10}  "
            f"{pct:>5.1f}  {_fmt_ns(k.mean_time_ns):>10}  "
            f"{_fmt_ns(k.max_time_ns):>10}  "
            f"{_f(sm)}  {_f(mem)}  {_f(dram)}"
        )

    if len(summary.kernels) > top_n:
        lines.append(
            f"... ({len(summary.kernels) - top_n} more kernels not shown; pass --top to widen)"
        )

    return "\n".join(lines) + "\n"
