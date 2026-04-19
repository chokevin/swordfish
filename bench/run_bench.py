"""Microbenchmark harness for INT4 x FP16 decode matmul on A100.

Usage
-----
Quick:
    uv run python -m bench.run_bench --shapes voice

Full run with JSON output and Perfetto trace:
    uv run python -m bench.run_bench \
        --shapes voice \
        --impls fp16,marlin,swordfish \
        --repeats 5 \
        --json bench_results/run1/ \
        --profile

Output schema
-------------
For each (shape, impl) pair we record:
    name, M, N, K, group_size, priority, tag,        # from the Shape
    impl,                                            # which implementation
    ms_mean, ms_p50, ms_p95, ms_min,                 # latency stats (ms)
    tflops_mean,                                     # 2*M*N*K / (ms_mean * 1e9)
    speedup_vs_fp16,                                 # ms_mean(fp16) / ms_mean(impl), NaN if no fp16 row
    error                                            # short error string if impl failed

CSV (one row per (shape,impl)) + JSON (env header + rows) + optional
Perfetto-loadable Chrome trace at <out>/trace.json.

NVTX ranges are emitted around every impl call as ``<impl>/<shape.name>``
so nsys timelines have clean per-call regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from bench.shapes import Shape, resolve
from swordfish.pack import random_quantized_weights
from swordfish.reference import reference_w4a16_matmul, dequantize_int4

# ----------------------------------------------------------------------
# environment capture
# ----------------------------------------------------------------------


def capture_env() -> dict:
    """Snapshot of the runtime — committed alongside every result file."""
    env: dict = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env.update(
            {
                "gpu_name": props.name,
                "gpu_cc": f"{props.major}.{props.minor}",
                "gpu_mem_gb": round(props.total_memory / 2**30, 1),
                "gpu_sm_count": props.multi_processor_count,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
        try:
            import triton  # type: ignore

            env["triton"] = triton.__version__
        except ImportError:
            env["triton"] = None
    return env


# ----------------------------------------------------------------------
# NVTX wrapper — no-op on non-CUDA
# ----------------------------------------------------------------------


@contextmanager
def nvtx_range(label: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


# ----------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------


def cuda_time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """Mean ms per call using cuda.Event timing. Single sample."""
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


def repeat_time_ms(fn, repeats: int, warmup: int, iters: int) -> dict:
    """Run cuda_time_ms `repeats` times and aggregate."""
    samples = [cuda_time_ms(fn, warmup=warmup, iters=iters) for _ in range(repeats)]
    samples_sorted = sorted(samples)
    return {
        "ms_mean": statistics.fmean(samples),
        "ms_p50": samples_sorted[len(samples) // 2],
        "ms_p95": samples_sorted[min(len(samples) - 1, int(0.95 * len(samples)))],
        "ms_min": samples_sorted[0],
        "samples": samples,
    }


def tflops(M: int, N: int, K: int, ms: float) -> float:
    """2*M*N*K FLOPs per matmul. ms -> TFLOPS."""
    if ms <= 0 or ms != ms:  # NaN check
        return float("nan")
    return (2.0 * M * N * K) / (ms * 1e9)


# ----------------------------------------------------------------------
# impl registry
# ----------------------------------------------------------------------
# An impl takes (a, packed, scales, group_size) and returns a callable
# that, when invoked, runs one matmul and returns the output tensor.
# It returns (callable, error_str_or_None). callable is None on error.


def impl_fp16(a, packed, scales, group_size):
    """Dequant once, then time bare torch.matmul. The "what FP16 would do."""
    w_fp = dequantize_int4(packed, scales, group_size=group_size)
    return (lambda: torch.matmul(a, w_fp)), None


def impl_reference(a, packed, scales, group_size):
    """Slow reference. Correctness anchor; not for speed claims."""
    return (lambda: reference_w4a16_matmul(a, packed, scales, group_size=group_size)), None


def impl_swordfish(a, packed, scales, group_size):
    try:
        from swordfish.kernels.triton_w4a16 import triton_w4a16_matmul
    except ImportError as e:
        return None, f"import_failed:{e}"
    # We probe with one call; if it raises NotImplementedError we surface that.
    try:
        triton_w4a16_matmul(a, packed, scales, group_size=group_size)
    except NotImplementedError:
        return None, "not_implemented"
    except Exception as e:  # noqa: BLE001
        return None, f"call_failed:{type(e).__name__}:{str(e)[:60]}"
    return (lambda: triton_w4a16_matmul(a, packed, scales, group_size=group_size)), None


def impl_marlin(a, packed, scales, group_size):
    try:
        import marlin  # type: ignore  # noqa: F401
    except ImportError:
        return None, "not_installed"
    try:
        from swordfish.marlin_compat import to_marlin_layout, marlin_matmul
    except ImportError as e:
        return None, f"compat_import_failed:{e}"
    try:
        marlin_w, marlin_s = to_marlin_layout(packed, scales, group_size=group_size)
    except Exception as e:  # noqa: BLE001
        return None, f"repack_failed:{type(e).__name__}:{str(e)[:60]}"
    return (lambda: marlin_matmul(a, marlin_w, marlin_s, group_size=group_size)), None


IMPLS = {
    "fp16": impl_fp16,
    "reference": impl_reference,
    "swordfish": impl_swordfish,
    "marlin": impl_marlin,
}


# ----------------------------------------------------------------------
# bench one shape across impls
# ----------------------------------------------------------------------


def bench_shape(
    shape: Shape,
    impls: list[str],
    *,
    repeats: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> list[dict]:
    """Returns one row per impl, all sharing the same shape fields."""
    M, N, K, g = shape.M, shape.N, shape.K, shape.group_size
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    packed, scales = random_quantized_weights(K, N, group_size=g, device=device)

    rows: list[dict] = []
    fp16_mean: float | None = None

    for impl_name in impls:
        if impl_name not in IMPLS:
            rows.append({**asdict(shape), "impl": impl_name, "error": "unknown_impl"})
            continue

        runner, err = IMPLS[impl_name](a, packed, scales, g)
        row: dict = {**asdict(shape), "impl": impl_name}
        if runner is None:
            row["error"] = err
            rows.append(row)
            continue

        # NVTX-tagged timed run.
        with nvtx_range(f"{impl_name}/{shape.name}"):
            stats = repeat_time_ms(runner, repeats=repeats, warmup=warmup, iters=iters)

        row.update(
            {
                "ms_mean": stats["ms_mean"],
                "ms_p50": stats["ms_p50"],
                "ms_p95": stats["ms_p95"],
                "ms_min": stats["ms_min"],
                "tflops_mean": tflops(M, N, K, stats["ms_mean"]),
                "error": None,
            }
        )

        if impl_name == "fp16":
            fp16_mean = stats["ms_mean"]

        rows.append(row)

    # second pass: speedup_vs_fp16
    for r in rows:
        if r.get("error") is None and fp16_mean is not None and "ms_mean" in r:
            r["speedup_vs_fp16"] = fp16_mean / r["ms_mean"]
        else:
            r["speedup_vs_fp16"] = float("nan")

    return rows


# ----------------------------------------------------------------------
# output writers
# ----------------------------------------------------------------------

CSV_FIELDS = [
    "name",
    "impl",
    "M",
    "N",
    "K",
    "group_size",
    "priority",
    "tag",
    "ms_mean",
    "ms_p50",
    "ms_p95",
    "ms_min",
    "tflops_mean",
    "speedup_vs_fp16",
    "error",
]


def write_outputs(out_dir: Path, env: dict, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # JSON manifest
    with (out_dir / "manifest.json").open("w") as f:
        json.dump({"env": env, "rows": rows}, f, indent=2, default=str)


# ----------------------------------------------------------------------
# torch.profiler -> Perfetto trace
# ----------------------------------------------------------------------


@contextmanager
def maybe_profile(out_dir: Path | None):
    """Activates torch.profiler if out_dir is set. Trace lands at <out>/trace.json
    and is loadable in https://ui.perfetto.dev or chrome://tracing."""
    if out_dir is None:
        yield
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        yield
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"perfetto trace: {trace_path}  (open at https://ui.perfetto.dev)")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="voice", help="shape set (voice/full/llama8b/...)")
    p.add_argument(
        "--impls", default="fp16,marlin,swordfish", help=f"comma-separated subset of {list(IMPLS)}"
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory (writes results.csv + manifest.json + trace.json if --profile)",
    )
    p.add_argument("--json", dest="out", help="alias for --out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--repeats", type=int, default=3, help="number of timing-loop repeats per impl")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20, help="iterations inside one timing loop")
    p.add_argument(
        "--profile",
        action="store_true",
        help="emit a Perfetto-loadable Chrome trace to <out>/trace.json",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("swordfish benchmarks require CUDA. No GPU found.")

    shapes = resolve(args.shapes)
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    env = capture_env()

    print(f"host: {env['host']}  gpu: {env.get('gpu_name', '?')} (cc {env.get('gpu_cc', '?')})")
    print(
        f"torch: {env['torch']}  cuda: {env.get('torch_cuda', '?')}  triton: {env.get('triton', '?')}"
    )
    print(
        f"shapes: {args.shapes} ({len(shapes)})  impls: {impls}  "
        f"repeats={args.repeats} iters={args.iters} warmup={args.warmup}\n"
    )

    out_dir = Path(args.out) if args.out else None
    profile_dir = out_dir if (args.profile and out_dir is not None) else None
    if args.profile and out_dir is None:
        print("warning: --profile without --out; trace will not be written", file=sys.stderr)

    rows: list[dict] = []
    with maybe_profile(profile_dir):
        for s in shapes:
            print(f"▶ {s.name}  M={s.M} N={s.N} K={s.K} g={s.group_size}")
            for r in bench_shape(
                s,
                impls,
                repeats=args.repeats,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
            ):
                rows.append(r)
                if r.get("error"):
                    print(f"  {r['impl']:>10s}  ERROR: {r['error']}")
                else:
                    print(
                        f"  {r['impl']:>10s}  "
                        f"{r['ms_mean']:7.3f}ms (p95 {r['ms_p95']:7.3f})  "
                        f"{r['tflops_mean']:6.1f} TF  "
                        f"x{r['speedup_vs_fp16']:.2f} vs fp16"
                    )

    if out_dir is not None:
        write_outputs(out_dir, env, rows)
        print(f"\nwrote {out_dir / 'results.csv'} and {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
