"""Command-line interface for the swordfish runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swordfish.quant.marlin_triton import run_w4a16_benchmark, write_w4a16_result
from swordfish.runner.backends import available_gemm_backends
from swordfish.runner.compare import write_results_comparison
from swordfish.runner.index import write_result_index
from swordfish.runner.liger_perkernel import (
    DEFAULT_DTYPE as LIGER_DEFAULT_DTYPE,
    KERNEL_NAMES as LIGER_KERNEL_NAMES,
    run_liger_perkernel,
)
from swordfish.runner.matrix import (
    DEFAULT_ARCH_LABELS,
    run_gemm_matrix,
    validate_gemm_matrix_results,
)
from swordfish.runner.schema import attach_ncu_summary
from swordfish.runner.status import write_completion_report
from swordfish.runner.torch_gemm import run_gemm_benchmark, write_result
from swordfish.runner.upstream import TARGET_LABELS, write_upstream_packet
from swordfish.transformer.bench import (
    run_transformer_forward_benchmark,
    run_transformer_train_step_benchmark,
)


def _cmd_run_gemm(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    result = run_gemm_benchmark(
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=args.dtype,
        repeats=args.repeats,
        warmup=args.warmup,
        iters=args.iters,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
        arch_label=args.arch_label,
        seed=args.seed,
        ncu_csv=args.ncu_csv,
        backend=args.backend,
    )
    result["command"] = argv
    write_result(result, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_run_liger_perkernel(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    result = run_liger_perkernel(
        kernel=args.kernel,
        batch=args.batch,
        seq=args.seq,
        hidden=args.hidden,
        intermediate=args.intermediate,
        eps=args.eps,
        dtype=args.dtype,
        repeats=args.repeats,
        warmup=args.warmup,
        iters=args.iters,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
        arch_label=args.arch_label,
        seed=args.seed,
        ncu_csv=args.ncu_csv,
    )
    result["command"] = argv
    write_result(result, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_attach_ncu(args: argparse.Namespace) -> int:
    result = json.loads(args.result.read_text())
    updated = attach_ncu_summary(result, args.ncu_csv)
    write_result(updated, args.out or args.result)
    print(f"wrote {args.out or args.result}", file=sys.stderr)
    return 0


def _cmd_run_gemm_matrix(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    paths = run_gemm_matrix(
        arch_labels=args.arch_labels,
        out_dir=args.out_dir,
        prefix=args.prefix,
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=args.dtype,
        repeats=args.repeats,
        warmup=args.warmup,
        iters=args.iters,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
        seed=args.seed,
        backend=args.backend,
        command=argv,
    )
    for path in paths:
        print(f"wrote {path}", file=sys.stderr)
    return 0


def _cmd_validate_gemm_matrix(args: argparse.Namespace) -> int:
    errors = validate_gemm_matrix_results(
        arch_labels=args.arch_labels,
        result_dir=args.result_dir,
        prefix=args.prefix,
        backend=args.backend,
        dtype=args.dtype,
        m=args.m,
        n=args.n,
        k=args.k,
        require_ncu=args.require_ncu,
        recursive=args.recursive,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("GEMM result matrix is complete", file=sys.stderr)
    return 0


def _cmd_bench_transformer(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    common_args = {
        "scope": args.scope,
        "preset": args.preset,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "iters": args.iters,
        "device_name": args.device,
        "allow_cpu": args.allow_cpu,
        "arch_label": args.arch_label,
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
    }
    if args.mode == "forward":
        result = run_transformer_forward_benchmark(**common_args)
    else:
        result = run_transformer_train_step_benchmark(
            **common_args,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    result["command"] = argv
    write_result(result, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_bench_w4a16(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    result = run_w4a16_benchmark(
        backend=args.backend,
        m=args.m,
        n=args.n,
        k=args.k,
        group_size=args.group_size,
        dtype=args.dtype,
        repeats=args.repeats,
        warmup=args.warmup,
        iters=args.iters,
        device_name=args.device,
        allow_cpu=args.allow_cpu,
        arch_label=args.arch_label,
        seed=args.seed,
    )
    result["command"] = argv
    write_w4a16_result(result, args.out)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_render_upstream_packet(args: argparse.Namespace) -> int:
    path = write_upstream_packet(
        result_path=args.result,
        target=args.target,
        out_path=args.out,
        title=args.title,
        ask=args.ask,
    )
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _cmd_compare_results(args: argparse.Namespace) -> int:
    path = write_results_comparison(args.results, args.out)
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _cmd_index_results(args: argparse.Namespace) -> int:
    path = write_result_index(
        args.result_dir,
        args.out,
        recursive=args.recursive,
        include_raw=args.include_raw,
    )
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _cmd_render_completion_report(args: argparse.Namespace) -> int:
    path, errors = write_completion_report(
        result_dir=args.result_dir,
        out_path=args.out,
        arch_labels=args.arch_labels,
        prefix=args.prefix,
        backend=args.backend,
        dtype=args.dtype,
        m=args.m,
        n=args.n,
        k=args.k,
        require_ncu=args.require_ncu,
        recursive=args.recursive,
    )
    print(f"wrote {path}", file=sys.stderr)
    return 1 if args.fail_on_incomplete and errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="swordfish cross-arch benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-gemm", help="run one torch/cuBLAS GEMM benchmark")
    run.add_argument("--backend", choices=available_gemm_backends(), default="torch")
    run.add_argument("--m", type=int, default=4096)
    run.add_argument("--n", type=int, default=4096)
    run.add_argument("--k", type=int, default=4096)
    run.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--iters", type=int, default=50)
    run.add_argument("--device", default="auto")
    run.add_argument(
        "--allow-cpu", action="store_true", help="allow CPU timing for local smoke tests"
    )
    run.add_argument("--arch-label", choices=["a100", "h100", "h200"], default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--ncu-csv", type=Path, default=None)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(func=_cmd_run_gemm)

    liger = sub.add_parser(
        "liger-perkernel",
        help="run one paired baseline-vs-Liger per-kernel benchmark",
    )
    liger.add_argument(
        "--kernel",
        choices=LIGER_KERNEL_NAMES,
        required=True,
        help="which kernel to bench (rmsnorm and swiglu are implemented; "
        "rope and fused_linear_ce raise NotImplementedError until follow-up work)",
    )
    liger.add_argument("--batch", type=int, default=4)
    liger.add_argument("--seq", type=int, default=2048)
    liger.add_argument("--hidden", type=int, default=4096)
    liger.add_argument("--intermediate", type=int, default=14336)
    liger.add_argument("--eps", type=float, default=1e-6)
    liger.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default=LIGER_DEFAULT_DTYPE,
        help="bf16 matches Liger's published defaults",
    )
    liger.add_argument("--repeats", type=int, default=5)
    liger.add_argument("--warmup", type=int, default=10)
    liger.add_argument("--iters", type=int, default=50)
    liger.add_argument("--device", default="auto")
    liger.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow CPU smoke; Liger half is reported as skipped on CPU",
    )
    liger.add_argument("--arch-label", choices=["a100", "h100", "h200"], default=None)
    liger.add_argument("--seed", type=int, default=0)
    liger.add_argument("--ncu-csv", type=Path, default=None)
    liger.add_argument("--out", type=Path, required=True)
    liger.set_defaults(func=_cmd_run_liger_perkernel)

    attach = sub.add_parser("attach-ncu", help="attach Nsight Compute CSV summary to a JSON result")
    attach.add_argument("--result", type=Path, required=True)
    attach.add_argument("--ncu-csv", type=Path, required=True)
    attach.add_argument("--out", type=Path, default=None)
    attach.set_defaults(func=_cmd_attach_ncu)

    matrix = sub.add_parser(
        "run-gemm-matrix",
        help="run the same GEMM contract for multiple arch labels",
    )
    matrix.add_argument("--arch-labels", nargs="+", default=list(DEFAULT_ARCH_LABELS))
    matrix.add_argument("--backend", choices=available_gemm_backends(), default="torch")
    matrix.add_argument("--m", type=int, default=4096)
    matrix.add_argument("--n", type=int, default=4096)
    matrix.add_argument("--k", type=int, default=4096)
    matrix.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    matrix.add_argument("--repeats", type=int, default=5)
    matrix.add_argument("--warmup", type=int, default=10)
    matrix.add_argument("--iters", type=int, default=50)
    matrix.add_argument("--device", default="auto")
    matrix.add_argument(
        "--allow-cpu", action="store_true", help="allow CPU timing for local smoke tests"
    )
    matrix.add_argument("--seed", type=int, default=0)
    matrix.add_argument("--prefix", default=None)
    matrix.add_argument("--out-dir", type=Path, required=True)
    matrix.set_defaults(func=_cmd_run_gemm_matrix)

    validate_matrix = sub.add_parser(
        "validate-gemm-matrix",
        help="validate one JSON GEMM result per requested GPU architecture",
    )
    validate_matrix.add_argument("--arch-labels", nargs="+", default=list(DEFAULT_ARCH_LABELS))
    validate_matrix.add_argument("--backend", choices=available_gemm_backends(), default="torch")
    validate_matrix.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    validate_matrix.add_argument("--m", type=int, default=None)
    validate_matrix.add_argument("--n", type=int, default=None)
    validate_matrix.add_argument("--k", type=int, default=None)
    validate_matrix.add_argument("--prefix", default=None)
    validate_matrix.add_argument("--result-dir", type=Path, required=True)
    validate_matrix.add_argument(
        "--recursive",
        action="store_true",
        help="search result-dir recursively for result files",
    )
    validate_matrix.add_argument(
        "--require-ncu",
        action="store_true",
        help="require attached Nsight Compute metrics to be complete",
    )
    validate_matrix.set_defaults(func=_cmd_validate_gemm_matrix)

    transformer = sub.add_parser(
        "bench-transformer",
        help="benchmark the PyTorch GPT reference block/model forward or train step",
    )
    transformer.add_argument("--mode", choices=["forward", "train-step"], default="forward")
    transformer.add_argument("--scope", choices=["block", "model"], default="block")
    transformer.add_argument("--preset", choices=["tiny", "gpt1"], default="tiny")
    transformer.add_argument("--batch-size", type=int, default=1)
    transformer.add_argument("--seq-len", type=int, default=8)
    transformer.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    transformer.add_argument("--repeats", type=int, default=3)
    transformer.add_argument("--warmup", type=int, default=1)
    transformer.add_argument("--iters", type=int, default=5)
    transformer.add_argument("--device", default="auto")
    transformer.add_argument(
        "--allow-cpu", action="store_true", help="allow CPU timing for local smoke tests"
    )
    transformer.add_argument("--arch-label", choices=["a100", "h100", "h200"], default=None)
    transformer.add_argument("--seed", type=int, default=0)
    transformer.add_argument("--vocab-size", type=int, default=None)
    transformer.add_argument("--block-size", type=int, default=None)
    transformer.add_argument("--n-layer", type=int, default=None)
    transformer.add_argument("--n-head", type=int, default=None)
    transformer.add_argument("--n-embd", type=int, default=None)
    transformer.add_argument("--lr", type=float, default=3e-4)
    transformer.add_argument("--weight-decay", type=float, default=0.1)
    transformer.add_argument("--out", type=Path, required=True)
    transformer.set_defaults(func=_cmd_bench_transformer)

    w4a16 = sub.add_parser(
        "bench-w4a16",
        help="benchmark the Marlin-style W4A16 reference or Triton artifact",
    )
    w4a16.add_argument("--backend", choices=["reference", "triton"], default="reference")
    w4a16.add_argument("--m", type=int, default=128)
    w4a16.add_argument("--n", type=int, default=256)
    w4a16.add_argument("--k", type=int, default=256)
    w4a16.add_argument("--group-size", type=int, default=128)
    w4a16.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    w4a16.add_argument("--repeats", type=int, default=3)
    w4a16.add_argument("--warmup", type=int, default=1)
    w4a16.add_argument("--iters", type=int, default=5)
    w4a16.add_argument("--device", default="auto")
    w4a16.add_argument(
        "--allow-cpu", action="store_true", help="allow CPU timing for local smoke tests"
    )
    w4a16.add_argument("--arch-label", choices=["a100", "h100", "h200"], default=None)
    w4a16.add_argument("--seed", type=int, default=0)
    w4a16.add_argument("--out", type=Path, required=True)
    w4a16.set_defaults(func=_cmd_bench_w4a16)

    packet = sub.add_parser(
        "render-upstream-packet",
        help="render a maintainer-ready upstream issue/PR packet from a result JSON",
    )
    packet.add_argument("--result", type=Path, required=True)
    packet.add_argument("--target", choices=sorted(TARGET_LABELS), required=True)
    packet.add_argument("--out", type=Path, required=True)
    packet.add_argument("--title", default=None)
    packet.add_argument("--ask", default=None)
    packet.set_defaults(func=_cmd_render_upstream_packet)

    compare = sub.add_parser(
        "compare-results",
        help="render a Markdown comparison table from benchmark result JSON files",
    )
    compare.add_argument("--result", dest="results", type=Path, nargs="+", required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.set_defaults(func=_cmd_compare_results)

    index = sub.add_parser(
        "index-results",
        help="scan benchmark result JSON files into a dashboard-friendly JSON index",
    )
    index.add_argument("--result-dir", type=Path, required=True)
    index.add_argument("--recursive", action="store_true")
    index.add_argument(
        "--include-raw",
        action="store_true",
        help="include *.raw.json intermediate benchmark outputs",
    )
    index.add_argument("--out", type=Path, required=True)
    index.set_defaults(func=_cmd_index_results)

    completion = sub.add_parser(
        "render-completion-report",
        help="render a Markdown report for the cross-arch completion gate",
    )
    completion.add_argument("--arch-labels", nargs="+", default=list(DEFAULT_ARCH_LABELS))
    completion.add_argument("--backend", choices=available_gemm_backends(), default="torch")
    completion.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    completion.add_argument("--m", type=int, default=None)
    completion.add_argument("--n", type=int, default=None)
    completion.add_argument("--k", type=int, default=None)
    completion.add_argument("--prefix", default=None)
    completion.add_argument("--result-dir", type=Path, required=True)
    completion.add_argument("--recursive", action="store_true")
    completion.add_argument("--require-ncu", action="store_true")
    completion.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="exit 1 after writing the report when the completion gate is not satisfied",
    )
    completion.add_argument("--out", type=Path, required=True)
    completion.set_defaults(func=_cmd_render_completion_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = ["swordfish.runner", *(argv or sys.argv[1:])]
    return args.func(args)
