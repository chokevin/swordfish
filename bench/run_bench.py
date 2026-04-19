"""Microbenchmark harness for INT4 x FP16 decode matmul.

Usage:
    uv run python -m bench.run_bench --shapes voice
    uv run python -m bench.run_bench --shapes full --out bench_results/run1.csv

Currently compares:
    - fp16 PyTorch baseline (torch.matmul on dequantized weights)
    - reference (slow, for correctness anchor)
    - swordfish triton kernel (once implemented)
    - marlin (once hooked up; optional dep)
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict

import torch

from bench.shapes import Shape, resolve
from swordfish.pack import random_quantized_weights
from swordfish.reference import reference_w4a16_matmul, dequantize_int4


def _cuda_time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """Time a callable on CUDA using events. Returns mean ms per call."""
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


def _bench_shape(shape: Shape, impls: list[str], device: str = "cuda") -> dict:
    M, N, K, g = shape.M, shape.N, shape.K, shape.group_size
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    packed, scales = random_quantized_weights(K, N, group_size=g, device=device)

    result: dict = {**asdict(shape)}

    # ---- fp16 baseline: dequant once, matmul many times ----
    if "fp16" in impls:
        w_fp = dequantize_int4(packed, scales, group_size=g)
        ms = _cuda_time_ms(lambda: torch.matmul(a, w_fp))
        result["fp16_ms"] = ms

    # ---- reference: dequant + matmul every call (correctness anchor, not a real baseline) ----
    if "reference" in impls:
        try:
            ms = _cuda_time_ms(
                lambda: reference_w4a16_matmul(a, packed, scales, group_size=g),
                warmup=2, iters=5,
            )
            result["reference_ms"] = ms
        except Exception as e:
            result["reference_ms"] = float("nan")
            result["reference_err"] = str(e)[:80]

    # ---- swordfish (not yet implemented) ----
    if "swordfish" in impls:
        try:
            from swordfish.kernels.triton_w4a16 import triton_w4a16_matmul
            ms = _cuda_time_ms(
                lambda: triton_w4a16_matmul(a, packed, scales, group_size=g)
            )
            result["swordfish_ms"] = ms
        except NotImplementedError:
            result["swordfish_ms"] = float("nan")
            result["swordfish_err"] = "not_implemented"
        except Exception as e:
            result["swordfish_ms"] = float("nan")
            result["swordfish_err"] = str(e)[:80]

    # ---- marlin (optional external) ----
    if "marlin" in impls:
        try:
            import marlin  # type: ignore
            # TODO: wire this up once marlin is in the env. Marlin has a specific
            # weight layout that's NOT what random_quantized_weights produces,
            # so this path will need a repack step. Deferred to week 1.
            result["marlin_ms"] = float("nan")
            result["marlin_err"] = "repack_not_implemented"
        except ImportError:
            result["marlin_ms"] = float("nan")
            result["marlin_err"] = "not_installed"

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="voice", help="shape set name (voice/full/...)")
    p.add_argument(
        "--impls",
        default="fp16,swordfish,marlin",
        help="comma-separated list of implementations to benchmark",
    )
    p.add_argument("--out", default=None, help="CSV output path (optional)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("swordfish benchmarks require CUDA. No GPU found.")

    shapes = resolve(args.shapes)
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: {args.shapes} ({len(shapes)} shapes)")
    print(f"Impls:  {impls}\n")

    rows = []
    for s in shapes:
        print(f"▶ {s.name}  M={s.M} N={s.N} K={s.K} g={s.group_size}")
        row = _bench_shape(s, impls, device=args.device)
        rows.append(row)
        # concise print
        times = {k: v for k, v in row.items() if k.endswith("_ms") and v == v}  # NaN-filter
        fmt = "  " + "  ".join(f"{k.replace('_ms', ''):>10s}={v:7.3f}ms" for k, v in times.items())
        print(fmt)

    if args.out:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
