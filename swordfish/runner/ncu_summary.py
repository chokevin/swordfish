"""Rich per-kernel summarizer for Nsight Compute reports.

Why this exists: `swordfish.runner.schema.parse_ncu_csv` collapses every NCU
CSV into 4 aggregate numbers attached to the result JSON. That's enough for
"is the matmul memory-bound on H100?" but useless for kernel-tuning iteration
where you need per-kernel detail (top kernels by time, per-kernel SoL%,
distribution across invocations).

This module produces the richer view: for each unique kernel, group every
invocation and every metric, then summarize. Output is a typed dataclass and
a text pretty-printer suitable for stdout / piping into a markdown report.

Two input formats are supported via the same NcuSummary output:

1. **NCU CSV export** (`*.ncu.csv`, `*.ncu-summary.csv`, etc.) — produced by
   `ncu --csv` or `ncu --import <rep> --csv`. Pure stdlib parser (`csv` +
   `statistics`). Works in the bench container, in CI, and on any machine.

2. **NCU binary report** (`*.ncu-rep`, `*.ncu-repz`) — proprietary NVIDIA
   format. Read via the `ncu_report` Python module that ships with Nsight
   Compute (Mac install at `/Applications/NVIDIA Nsight Compute.app/Contents/
   MacOS/python/`; Linux at `/opt/nvidia/nsight-compute/<ver>/extras/python/`).
   The module is auto-discovered or pointed at via the `NCU_REPORT_PYTHON_DIR`
   env var. Falls back to a clear actionable error if neither is found.

Limitations (honest):
- Source-line attribution, SASS view, occupancy widget, and the full
  Speed-of-Light radial chart live only in `ncu-ui` against the `.ncu-rep`.
  This summarizer is the agent-readable substitute, not a replacement.
- The CSV path can only see the metrics NCU was asked to capture (the
  `--metrics` flag at capture time). The .ncu-rep path can read every
  metric NCU recorded — but we still report the same headline 4 by default
  for parity with the JSON `ncu` field.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


# Friendly metric names → canonical engine names. Rune's default ncu invocation
# (no `--section`) emits human-readable metric names like "Duration"; the
# legacy SWORDFISH_PROFILE=ncu path (with explicit `--section LaunchStats
# Occupancy SpeedOfLight MemoryWorkloadAnalysis`) emits engine names. Map the
# friendly form into the engine form so the rest of the parser and the display
# table are naming-convention agnostic.
_FRIENDLY_TO_ENGINE_METRIC: dict[str, str] = {
    "Duration": "gpu__time_duration.sum",
    "Compute (SM) Throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Memory Throughput": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM Throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
}

# Time units NCU may emit for Duration; values normalized to nanoseconds so
# downstream code can treat gpu__time_duration.sum uniformly.
_TIME_UNIT_TO_NS: dict[str, float] = {
    "ns": 1.0,
    "us": 1e3,
    "ms": 1e6,
    "s": 1e9,
}


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
        canonical = _FRIENDLY_TO_ENGINE_METRIC.get(metric, metric)
        if canonical == "gpu__time_duration.sum" and unit in _TIME_UNIT_TO_NS:
            value = value * _TIME_UNIT_TO_NS[unit]
            unit = "ns"
        invocations[kname][inv_id][canonical] = (unit, value)
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


# ---------------------------------------------------------------------------
# .ncu-rep binary reader (via NVIDIA's ncu_report Python module)
#
# `ncu_report` ships with every Nsight Compute install. It's a SWIG wrapper
# around _ncu_report.so. The Mac install puts both files at
# `<app>/Contents/MacOS/python/`. The Linux install puts them at
# `/opt/nvidia/nsight-compute/<ver>/extras/python/`. We auto-discover both
# and respect $NCU_REPORT_PYTHON_DIR as an explicit override.
#
# The module is pure inspect API: load_report → IContext → IRange × N →
# IAction × M (one per kernel launch) → IMetric → as_double() / as_uint64().
# We walk the tree and produce the same NcuSummary the CSV path produces.
#
# Why not bundle ncu_report? Two reasons:
#   1. NVIDIA's EULA forbids redistribution.
#   2. The .so is platform-specific (arm64-mac vs x86_64-linux); a single
#      bundled copy wouldn't work on both bench-container and dev-laptop.
# ---------------------------------------------------------------------------


# Full SoL metric set we extract per IAction. Same 4 the CSV path can read,
# plus a few extras the binary report carries by default that are useful
# for kernel tuning (achieved/peak FP throughput).
_NCU_REP_METRICS: tuple[str, ...] = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


# Default Mac install path for Nsight Compute (Homebrew cask + NVIDIA installer
# both land here). Listed first so a stock Mac dev box "just works".
_NCU_REPORT_DEFAULT_PATHS: tuple[str, ...] = (
    "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python",
    "/opt/nvidia/nsight-compute/2025.1.0/extras/python",
    "/opt/nvidia/nsight-compute/2024.3.0/extras/python",
    "/opt/nvidia/nsight-compute/2024.2.0/extras/python",
)


class NcuReportUnavailableError(RuntimeError):
    """Raised when we can't import NVIDIA's ncu_report module.

    Carries an actionable message that points the user at the install or at
    the override env var. Distinct exception type so the CLI can catch it
    and degrade gracefully rather than print a traceback.
    """


def _import_ncu_report() -> Any:
    """Locate and import NVIDIA's ncu_report module.

    Resolution order:
      1. $NCU_REPORT_PYTHON_DIR (explicit override)
      2. Already importable on the current sys.path
      3. _NCU_REPORT_DEFAULT_PATHS (well-known install locations)

    Imports via `importlib.util.spec_from_file_location` so we don't mutate
    sys.path (which would leak across test boundaries and pollute other
    pytest invocations of this module).

    Raises NcuReportUnavailableError with install instructions on failure.
    """
    import importlib.machinery
    import importlib.util
    import sys

    override = os.environ.get("NCU_REPORT_PYTHON_DIR")

    def _find_native_companion(d: Path) -> Path | None:
        """Locate the SWIG `_ncu_report` extension in the same dir as ncu_report.py.

        On Mac it's `_ncu_report.so`; on Linux it may carry a CPython suffix
        like `_ncu_report.cpython-311-x86_64-linux-gnu.so`. We accept any.
        """
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            cand = d / f"_ncu_report{suffix}"
            if cand.is_file():
                return cand
        # Fallback to a glob over `_ncu_report*.so` for unusual layouts.
        for cand in d.glob("_ncu_report*"):
            if cand.suffix in {".so", ".pyd", ".dylib"}:
                return cand
        return None

    def _try_load_from_dir(d: str) -> Any | None:
        dir_path = Path(d)
        py = dir_path / "ncu_report.py"
        if not py.is_file():
            return None

        # The SWIG-generated ncu_report.py wrapper does `import _ncu_report`.
        # When loaded via spec_from_file_location, the companion .so isn't on
        # sys.path, so we pre-register it in sys.modules ourselves.
        native = _find_native_companion(dir_path)
        if native is not None and "_ncu_report" not in sys.modules:
            ext_spec = importlib.util.spec_from_file_location("_ncu_report", native)
            if ext_spec is not None and ext_spec.loader is not None:
                try:
                    ext_mod = importlib.util.module_from_spec(ext_spec)
                    sys.modules["_ncu_report"] = ext_mod
                    ext_spec.loader.exec_module(ext_mod)
                except Exception:  # pragma: no cover — only when .so is broken
                    sys.modules.pop("_ncu_report", None)
                    return None

        spec = importlib.util.spec_from_file_location("ncu_report", py)
        if spec is None or spec.loader is None:
            return None
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:  # pragma: no cover — only when .so is broken / quarantined
            return None

    # 1. explicit env override always wins
    if override:
        mod = _try_load_from_dir(override)
        if mod is not None:
            return mod

    # 2. already importable
    try:
        import ncu_report  # type: ignore[import-not-found]

        return ncu_report
    except ImportError:
        pass

    # 3. well-known install locations
    for cand in _NCU_REPORT_DEFAULT_PATHS:
        mod = _try_load_from_dir(cand)
        if mod is not None:
            return mod

    raise NcuReportUnavailableError(
        "\n".join(
            [
                "Cannot import NVIDIA's ncu_report module (required to read .ncu-rep files).",
                "Install Nsight Compute and one of these will work:",
                "  Mac:    brew install --cask nvidia-nsight-compute",
                "  Linux:  download from https://developer.nvidia.com/nsight-compute",
                "Or point NCU_REPORT_PYTHON_DIR at the directory containing ncu_report.py.",
            ]
        )
    )


def _action_metric_value(action: Any, metric_name: str) -> tuple[str, float] | None:
    """Pull (unit, value) for one metric off an IAction. Returns None if absent.

    `metric_by_name` returns None when the metric wasn't captured. When
    present, prefer `as_double()` for SoL/throughput metrics and fall back
    to `as_uint64()` for counters (the SWIG wrapper raises if you ask for
    the wrong kind, so we try-except).
    """
    metric = action.metric_by_name(metric_name)
    if metric is None:
        return None
    unit = ""
    try:
        unit = metric.unit() or ""
    except Exception:
        pass
    try:
        return unit, float(metric.as_double())
    except Exception:
        pass
    try:
        return unit, float(metric.as_uint64())
    except Exception:
        return None


def parse_ncu_rep(path: Path, metrics: tuple[str, ...] = _NCU_REP_METRICS) -> NcuSummary:
    """Parse a `.ncu-rep` (or `.ncu-repz`) binary report into per-kernel stats.

    Walks every IRange × IAction in the report, groups by kernel name (the
    demangled IAction.name), and pivots metrics by kernel just like the CSV
    path does. Output is the same NcuSummary type so callers (and
    `format_summary_text`) treat both formats uniformly.

    Requires NVIDIA's ncu_report Python module — see `_import_ncu_report` for
    the discovery mechanism.

    Args:
        path: A `.ncu-rep` or `.ncu-repz` file produced by Nsight Compute.
        metrics: Names of metrics to extract per kernel. Defaults to the
            same headline 4 the CSV path reports. Pass extras to get the
            richer metric set the binary format carries (e.g. achieved
            FP32 / FP16 throughput, register usage, occupancy).

    Raises:
        NcuReportUnavailableError: ncu_report module not found.
        FileNotFoundError: report file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    ncu_report = _import_ncu_report()
    ctx = ncu_report.load_report(str(path))

    invocations: dict[str, dict[int, dict[str, tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    warnings: list[str] = []
    inv_idx = 0  # global IAction counter, used as invocation key

    NameBase_DEMANGLED = getattr(ncu_report.IAction, "NameBase_DEMANGLED", None)

    for r in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(r)
        for a in range(rng.num_actions()):
            action = rng.action_by_idx(a)
            try:
                kname = (
                    action.name(NameBase_DEMANGLED)
                    if NameBase_DEMANGLED is not None
                    else action.name()
                )
            except Exception:
                kname = action.name()
            kname = (kname or "").strip()
            if not kname:
                continue
            for mname in metrics:
                pair = _action_metric_value(action, mname)
                if pair is None:
                    continue
                invocations[kname][inv_idx][mname] = pair
            inv_idx += 1

    if inv_idx == 0:
        warnings.append("ncu-rep contained no IAction objects (empty profile?)")

    kernels: list[KernelStats] = []
    grand_total_time = 0.0
    total_invocations = 0

    for kname, inv_map in invocations.items():
        per_metric: dict[str, tuple[str, list[float]]] = defaultdict(lambda: ("", []))
        time_series: list[float] = []
        for _idx, m in inv_map.items():
            if "gpu__time_duration.sum" in m:
                _, t = m["gpu__time_duration.sum"]
                time_series.append(t)
            for mname, (unit, val) in m.items():
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

        kernels.append(
            KernelStats(
                name=kname,
                short_name=_short_name(kname),
                invocations=invocation_count,
                block_size="",  # not currently extracted from .ncu-rep; CSV has it
                grid_size="",
                total_time_ns=total_time,
                mean_time_ns=mean_time,
                max_time_ns=max_time,
                metrics=metric_stats,
            )
        )
        grand_total_time += total_time
        total_invocations += invocation_count

    kernels.sort(key=lambda k: k.total_time_ns, reverse=True)
    # `rows` for the binary path = total IAction count (= total invocations
    # × metrics), to keep the field meaningful relative to the CSV path.
    return NcuSummary(
        path=path,
        rows=inv_idx * len(metrics),
        unique_kernels=len(kernels),
        total_invocations=total_invocations,
        total_time_ns=grand_total_time,
        kernels=kernels,
        parse_warnings=warnings,
    )


def summarize_ncu_file(path: Path) -> NcuSummary:
    """Dispatch on file extension: `.ncu-rep` / `.ncu-repz` → binary parser,
    everything else → CSV parser. The unified entrypoint callers should use.
    """
    suffix = path.suffix.lower()
    if suffix in {".ncu-rep", ".ncu-repz"}:
        return parse_ncu_rep(path)
    return parse_ncu_csv_full(path)


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
