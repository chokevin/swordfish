"""Roofline plot from ncu CSV output for A100.

Consumes the ``<shape>.ncu.csv`` files written by bench/profile_marlin.sh,
computes per-kernel arithmetic intensity (FLOPs / HBM-byte) and achieved
TFLOPS, and overlays them on the A100 roofline.

A100 SXM 80GB peaks (the deployment target):
  - HBM2e bandwidth:        2.039 TB/s
  - FP16 tensor-core peak:  312 TFLOPS
  - FP32 (non-TC) peak:     19.5 TFLOPS  (drawn for context)

Usage:
  python -m bench.roofline docs/profiling/marlin/<timestamp>/
  python -m bench.roofline docs/profiling/marlin/<timestamp>/ --gpu a100-40gb
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# A100 hardware peaks. We expose two SKUs because both ship in real fleets.
GPU_PEAKS = {
    "a100-80gb-sxm": {"hbm_TBps": 2.039, "fp16_tc_TFLOPS": 312.0, "fp32_TFLOPS": 19.5},
    "a100-40gb": {"hbm_TBps": 1.555, "fp16_tc_TFLOPS": 312.0, "fp32_TFLOPS": 19.5},
}


def parse_ncu_csv(path: Path) -> list[dict]:
    """ncu --csv emits a header line preceded by ==PROF== noise lines.
    We strip those, then DictReader the rest."""
    raw = path.read_text().splitlines()
    # find first line that looks like the CSV header (contains a known column)
    header_idx = next(
        (i for i, ln in enumerate(raw) if "ID" in ln and "Kernel Name" in ln),
        None,
    )
    if header_idx is None:
        return []
    rows = list(csv.DictReader(raw[header_idx:]))
    return rows


def _f(row: dict, key: str) -> float:
    """Pull a numeric metric. ncu emits values with thousands separators sometimes."""
    v = row.get(key, "")
    if v in ("", "n/a", "N/A"):
        return float("nan")
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return float("nan")


def kernel_ai_and_tflops(row: dict) -> tuple[str, float, float] | None:
    """Returns (kernel_name, arithmetic_intensity_FLOPs_per_byte, achieved_TFLOPS)
    or None if the row lacks the metrics we need."""
    name = row.get("Kernel Name", "").strip()
    if not name:
        return None

    # FP16 FLOPs: 2 ops/HFMA + 1/HMUL + 1/HADD; tensor-core ops are counted
    # via inst_executed_pipe_tensor (each op = 256 FMA at m16n8k16 fp16->fp32).
    hfma = _f(row, "sm__sass_thread_inst_executed_op_hfma_pred_on.sum")
    hmul = _f(row, "sm__sass_thread_inst_executed_op_hmul_pred_on.sum")
    hadd = _f(row, "sm__sass_thread_inst_executed_op_hadd_pred_on.sum")
    tc_inst = _f(row, "sm__inst_executed_pipe_tensor.sum")
    bytes_hbm = _f(row, "dram__bytes.sum")
    duration_ns = _f(row, "gpu__time_duration.sum")

    # mma.sync.m16n8k16 fp16: 16*8*16 = 2048 multiply-adds = 4096 FLOPs per warp
    # instruction; ncu counts at the warp-scheduler level so multiplier is 4096.
    flops = (2 * hfma + hmul + hadd) + tc_inst * 4096.0

    if not (flops > 0 and bytes_hbm > 0 and duration_ns > 0):
        return None

    ai = flops / bytes_hbm
    tflops = flops / (duration_ns * 1e3)  # ns -> s, then /1e12 -> /1e3 since flops is raw
    return name, ai, tflops


def collect(profile_dir: Path) -> list[tuple[str, str, float, float]]:
    """Returns [(shape_name, kernel_name, AI, TFLOPS), ...]"""
    out: list[tuple[str, str, float, float]] = []
    for csv_path in sorted(profile_dir.glob("*.ncu.csv")):
        shape = csv_path.name.removesuffix(".ncu.csv")
        for row in parse_ncu_csv(csv_path):
            r = kernel_ai_and_tflops(row)
            if r is None:
                continue
            name, ai, tf = r
            out.append((shape, name, ai, tf))
    return out


def plot(points, peaks: dict, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; install bench extras: uv sync --extra bench", file=sys.stderr
        )
        sys.exit(1)

    import numpy as np

    hbm = peaks["hbm_TBps"] * 1e12  # bytes/s
    tc_peak = peaks["fp16_tc_TFLOPS"] * 1e12  # FLOPs/s
    fp32_peak = peaks["fp32_TFLOPS"] * 1e12

    ai_grid = np.logspace(-1, 4, 256)  # 0.1 .. 10000 FLOPs/byte
    mem_bound = hbm * ai_grid  # bytes/s * FLOPs/byte = FLOPs/s
    tc_roof = np.minimum(mem_bound, tc_peak)
    fp32_roof = np.minimum(mem_bound, fp32_peak)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(ai_grid, tc_roof / 1e12, "k-", label="FP16 tensor-core peak (312 TF)")
    ax.loglog(ai_grid, fp32_roof / 1e12, "k--", alpha=0.5, label="FP32 non-TC peak (19.5 TF)")

    # group points by shape so legend stays readable
    by_shape: dict[str, list[tuple[float, float]]] = {}
    for shape, _kname, ai, tf in points:
        by_shape.setdefault(shape, []).append((ai, tf))
    for shape, pts in by_shape.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, label=shape, s=40, alpha=0.8)

    # ridge point
    ridge_ai = tc_peak / hbm
    ax.axvline(ridge_ai, color="grey", linestyle=":", alpha=0.5)
    ax.text(ridge_ai * 1.1, 0.5, f"ridge AI = {ridge_ai:.1f}", color="grey", fontsize=9)

    ax.set_xlabel("Arithmetic intensity (FLOPs / HBM byte)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("A100 roofline — Marlin INT4×FP16 decode kernels")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.1, 1e3)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("profile_dir", type=Path, help="directory containing <shape>.ncu.csv files")
    p.add_argument("--gpu", choices=list(GPU_PEAKS), default="a100-80gb-sxm")
    p.add_argument(
        "--out", type=Path, default=None, help="output PNG (default: <profile_dir>/roofline.png)"
    )
    args = p.parse_args()

    if not args.profile_dir.is_dir():
        raise SystemExit(f"not a directory: {args.profile_dir}")

    points = collect(args.profile_dir)
    if not points:
        raise SystemExit(f"no usable kernel rows in {args.profile_dir}/*.ncu.csv")

    out_path = args.out or args.profile_dir / "roofline.png"
    plot(points, GPU_PEAKS[args.gpu], out_path)

    print(f"\n{len(points)} kernel rows across {len({s for s, *_ in points})} shapes")
    print("top-5 kernels by achieved TFLOPS:")
    for shape, kname, ai, tf in sorted(points, key=lambda r: -r[3])[:5]:
        print(f"  {shape:20s} {tf:6.1f} TF  AI={ai:7.2f}  {kname[:60]}")


if __name__ == "__main__":
    main()
