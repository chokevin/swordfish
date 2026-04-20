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
    """Mean ms per call using cuda.Event timing. Single sample.

    NOTE (W1 lesson, see docs/profiling/marlin-bottlenecks.md): cuda.Event
    measures stream time between events, which CAPTURES host-side idle when
    Python can't keep up with the GPU. For ~20 µs kernels, dispatch
    dominates and shows up as "kernel" time. Use `--capture` to remove the
    Python dispatch from the loop and see the kernel-only number.
    """
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


def cuda_graph_time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """Capture `fn` into a CUDA graph and time `iters` graph replays.

    This is the rvLLM lesson (§7.3): graph capture moved their stack from
    551 → 14,745 tok/s (27×) before any kernel work. For us it's the
    methodologically clean way to MEASURE the wrapper-overhead cost we
    identified in W1: the delta between `cuda_time_ms(fn)` and
    `cuda_graph_time_ms(fn)` is exactly the per-call host-side dispatch
    cost that any future graph-captured caller would NOT pay.

    GOTCHAS (rvLLM §5.2):
    - `fn` must not allocate inside the captured region. Our marlin_compat
      workspace cache + optional `out=` arg cover this.
    - First call records, subsequent replays bind to the recorded device
      offsets — any tensor whose storage moves between calls produces
      silent wrong-data on replay. We capture once with stable inputs and
      replay against the same inputs.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    # Time graph replays
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def repeat_time_ms(fn, repeats: int, warmup: int, iters: int, *, capture: bool = False) -> dict:
    """Run timing `repeats` times and aggregate. `capture=True` uses CUDA graphs."""
    timer = cuda_graph_time_ms if capture else cuda_time_ms
    samples = [timer(fn, warmup=warmup, iters=iters) for _ in range(repeats)]
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


def _max_relerr(out, ref) -> float:
    """Max |out - ref| / (|ref| + eps), finite only. Cheap correctness signal."""
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs() + 1e-3
    return float((diff / denom).max().item())


def _cosine_sim(out, ref) -> float:
    """Cosine similarity flattened. rvLLM uses 0.999 as their FP8 kernel
    correctness band — it's the right shape for "do these compute the same
    math up to accumulation order" because it's invariant to per-row scale."""
    a = out.float().flatten()
    b = ref.float().flatten()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b / denom).item())


def bench_shape(
    shape: Shape,
    impls: list[str],
    *,
    repeats: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
    capture: bool = False,
) -> list[dict]:
    """Returns one row per impl, all sharing the same shape fields."""
    M, N, K, g = shape.M, shape.N, shape.K, shape.group_size
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    packed, scales = random_quantized_weights(K, N, group_size=g, device=device)

    # L0 correctness anchor: every impl's output is compared to the slow
    # reference matmul on identical inputs. Bands picked from the bounded
    # error of group-quant W4A16 (within-group error is ~ scale/16 absolute,
    # but matmul accumulation can amplify so we allow rtol=0.05 / atol=5e-3
    # in fp16). A kernel that gives a wildly different answer flunks here
    # before any speed number is reported — this is what catches scale-format
    # mismatches, packing-layout bugs, and group-size off-by-ones.
    with torch.no_grad():
        ref_out = reference_w4a16_matmul(a, packed, scales, group_size=g)

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

        # Correctness check first — speed numbers from a wrong kernel are
        # worse than no number at all.
        try:
            with torch.no_grad():
                out = runner()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            row["max_relerr"] = _max_relerr(out, ref_out)
            row["cosine"] = _cosine_sim(out, ref_out)
            # Two-band check: pointwise allclose (catches local bugs) AND
            # cosine ≥ 0.999 (catches systematic-bias bugs that allclose
            # might pass with a generous rtol). rvLLM lesson §3.3.
            row["correct"] = bool(
                torch.allclose(out, ref_out, atol=5e-3, rtol=5e-2)
                and row["cosine"] >= 0.999
            )
        except Exception as e:  # noqa: BLE001
            row["error"] = f"correctness_call_failed:{type(e).__name__}:{str(e)[:60]}"
            rows.append(row)
            continue

        if not row["correct"]:
            # Don't time something that's numerically wrong. Surface the
            # delta so the operator can see HOW wrong before fixing.
            row["error"] = f"correctness_failed:max_relerr={row['max_relerr']:.3g}"
            rows.append(row)
            continue

        # NVTX-tagged timed run. Eager + (optional) graph-captured.
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

        # Optional second pass under CUDA graph capture. The delta
        # (ms_mean - ms_captured) is the per-call host-side overhead that
        # any future graph-captured caller would NOT pay. This is the
        # rvLLM §7.3 measurement: see how big the wrapper-overhead share
        # of wall-clock is, BEFORE we attempt to remove it.
        if capture:
            try:
                with nvtx_range(f"{impl_name}/{shape.name}/captured"):
                    cap_stats = repeat_time_ms(
                        runner, repeats=repeats, warmup=warmup, iters=iters, capture=True
                    )
                row["ms_captured"] = cap_stats["ms_mean"]
                row["wrapper_overhead_ms"] = stats["ms_mean"] - cap_stats["ms_mean"]
                row["wrapper_overhead_pct"] = (
                    (stats["ms_mean"] - cap_stats["ms_mean"]) / stats["ms_mean"] * 100.0
                    if stats["ms_mean"] > 0 else float("nan")
                )
            except Exception as e:  # noqa: BLE001
                row["capture_error"] = f"{type(e).__name__}:{str(e)[:60]}"

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
    "ms_captured",
    "wrapper_overhead_ms",
    "wrapper_overhead_pct",
    "correct",
    "max_relerr",
    "cosine",
    "error",
    "capture_error",
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
    p.add_argument(
        "--capture",
        action="store_true",
        help="also time each impl under CUDA-graph capture; reports "
        "wrapper_overhead_ms = (eager_ms - captured_ms). This is the "
        "rvLLM §7.3 / W1 wrapper-overhead measurement: see how big the "
        "Python-dispatch share of wall-clock is BEFORE we remove it.",
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
                capture=args.capture,
            ):
                rows.append(r)
                if r.get("error"):
                    print(f"  {r['impl']:>10s}  ERROR: {r['error']}")
                else:
                    line = (
                        f"  {r['impl']:>10s}  "
                        f"{r['ms_mean']:7.3f}ms (p95 {r['ms_p95']:7.3f})  "
                        f"{r['tflops_mean']:6.1f} TF  "
                        f"x{r['speedup_vs_fp16']:.2f} vs fp16"
                    )
                    if "ms_captured" in r:
                        line += (
                            f"  | captured {r['ms_captured']:.3f}ms  "
                            f"(wrapper {r['wrapper_overhead_pct']:+.0f}%)"
                        )
                    if "cosine" in r:
                        line += f"  cos={r['cosine']:.4f}"
                    print(line)

    if out_dir is not None:
        write_outputs(out_dir, env, rows)
        print(f"\nwrote {out_dir / 'results.csv'} and {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
