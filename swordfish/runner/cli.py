"""Command-line interface for the swordfish runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swordfish.dispatch import (
    LigerPerkernelRun,
    TorchGemmRun,
    build_run_for_experiment,
    format_experiment_explain,
    format_experiment_table,
    list_experiments,
)
from swordfish.dispatch.profiles import PACK_YAML_PATH, render_pack_yaml
from swordfish.quant.marlin_triton import run_w4a16_benchmark, write_w4a16_result
from swordfish.runner.backends import available_gemm_backends
from swordfish.runner.compare import write_results_comparison
from swordfish.runner.index import write_result_index
from swordfish.runner.liger_perkernel import (
    DEFAULT_DTYPE as LIGER_DEFAULT_DTYPE,
    KERNEL_NAMES as LIGER_KERNEL_NAMES,
    run_liger_perkernel,
)
from swordfish.runner.liger_fsdp import (
    MODEL_PRESETS as LIGER_FSDP_MODEL_PRESETS,
    run_liger_fsdp_step,
)
from swordfish.runner.matrix import (
    DEFAULT_ARCH_LABELS,
    run_gemm_matrix,
    validate_gemm_matrix_results,
)
from swordfish.runner.profile_torch import (
    resolve_torch_profile_out,
    torch_profiler_context,
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
    with torch_profiler_context(resolve_torch_profile_out()):
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
    with torch_profiler_context(resolve_torch_profile_out()):
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


def _cmd_run_liger_fsdp_step(args: argparse.Namespace) -> int:
    argv = sys.argv if args.argv is None else args.argv
    with torch_profiler_context(resolve_torch_profile_out()):
        result = run_liger_fsdp_step(
            mode=args.liger_mode,
            model_source=args.model_source,
            model_preset=args.model_preset,
            micro_batch_size=args.micro_batch_size,
            seq_len=args.seq_len,
            dtype=args.dtype,
            repeats=args.repeats,
            warmup=args.warmup,
            iters=args.iters,
            device_name=args.device,
            allow_cpu=args.allow_cpu,
            arch_label=args.arch_label,
            seed=args.seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    if result is None:
        return 0
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


def _build_submit_run(args: argparse.Namespace):
    """Translate `submit-bench` argv into the matching dispatch dataclass."""
    common: dict = {}
    if getattr(args, "result_root", None):
        common["result_root"] = args.result_root
    if getattr(args, "script", None):
        common["script"] = args.script
    repeats = args.repeats if args.repeats is not None else 5
    warmup = args.warmup if args.warmup is not None else 10
    iters = args.iters if args.iters is not None else 50
    if args.workload == "gemm":
        return TorchGemmRun(
            arch=args.arch,
            backend=args.backend,
            m=args.m,
            n=args.n,
            k=args.k,
            dtype=args.dtype or "fp16",
            repeats=repeats,
            warmup=warmup,
            iters=iters,
            name=args.name,
            profile_mode=args.profile_mode,
            **common,
        )
    if args.workload == "liger-fsdp":
        repeats = args.repeats if args.repeats is not None else 3
        warmup = args.warmup if args.warmup is not None else 1
        iters = args.iters if args.iters is not None else 5
        from swordfish.dispatch import LigerFsdpRun

        return LigerFsdpRun(
            arch=args.arch,
            mode=args.liger_mode,
            model_source=args.model_source,
            model_preset=args.model_preset,
            micro_batch_size=args.micro_batch_size,
            seq_len=args.seq_len,
            dtype=args.dtype or "bf16",
            repeats=repeats,
            warmup=warmup,
            iters=iters,
            nproc_per_node=args.nproc_per_node,
            name=args.name,
            profile_mode=args.profile_mode,
            gradient_checkpointing=args.gradient_checkpointing,
            **common,
        )
    kernel = "rmsnorm" if args.workload == "liger-rmsnorm" else "swiglu"
    return LigerPerkernelRun(
        kernel=kernel,
        arch=args.arch,
        dtype=args.dtype or "bf16",
        repeats=repeats,
        warmup=warmup,
        iters=iters,
        name=args.name,
        profile_mode=args.profile_mode,
        **common,
    )


def _cmd_submit_bench(args: argparse.Namespace) -> int:
    run = _build_submit_run(args)
    result = run.submit(dry_run=args.dry_run)
    print(result.name, file=sys.stderr)
    if args.print_yaml and result.rendered_yaml:
        print(result.rendered_yaml)
    return 0


def _experiment_overrides(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "backend",
        "m",
        "n",
        "k",
        "dtype",
        "repeats",
        "warmup",
        "iters",
        "model_source",
        "model_preset",
        "micro_batch_size",
        "seq_len",
        "gradient_checkpointing",
        "nproc_per_node",
        "name",
        "profile_mode",
        "result_root",
        "script",
    )
    out: dict[str, object] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    if getattr(args, "liger_mode", None) is not None:
        out["mode"] = args.liger_mode
    return out


def _cmd_list_experiments(args: argparse.Namespace) -> int:
    print(format_experiment_table())
    return 0


def _cmd_explain_experiment(args: argparse.Namespace) -> int:
    print(format_experiment_explain(args.experiment, args.arch))
    return 0


def _cmd_submit_experiment(args: argparse.Namespace) -> int:
    run = build_run_for_experiment(
        args.experiment,
        args.arch,
        _experiment_overrides(args),
    )
    result = run.submit(dry_run=args.dry_run)
    print(result.name, file=sys.stderr)
    if args.print_yaml and result.rendered_yaml:
        print(result.rendered_yaml)
    return 0


def _cmd_inspect_run(args: argparse.Namespace) -> int:
    """Fetch a finished rune job's artifacts and open the trace locally.

    The day-to-day inspection loop on a Mac:
      $ python -m swordfish.runner inspect-run NAME --profile-mode ncu
      ... fetches NAME.json + NAME.ncu-rep into runs/inspect/NAME/ ...
      ... opens NAME.ncu-rep with ncu-ui (via macOS `open`) ...

    Idempotent: re-running with the same name skips re-fetch unless
    --overwrite is passed. Use --no-open to fetch without launching the GUI.
    """
    from swordfish.dispatch import fetch_run_artifacts
    from swordfish.dispatch.results import ResultFetchError

    local_dir = args.local_dir or Path("runs/inspect") / args.name
    fetched = fetch_run_artifacts(
        name=args.name,
        profile_mode=args.profile_mode,
        local_dir=local_dir,
        namespace=args.namespace,
        context=args.context,
        pvc=args.pvc,
        overwrite=args.overwrite,
    )

    print(f"result json:      {fetched.result_json}", file=sys.stderr)
    if fetched.profile_artifact:
        print(f"profile artifact: {fetched.profile_artifact}", file=sys.stderr)

    # If a CSV companion or the .ncu-rep itself is in the fetched artifacts,
    # auto-print the per-kernel summary on stdout. The CSV path is pure
    # stdlib; the .ncu-rep path needs NVIDIA's ncu_report module (ships with
    # any Nsight Compute install — Mac: `brew install --cask
    # nvidia-nsight-compute`). When neither readable form is available, we
    # print an actionable hint instead of silently skipping.
    if args.profile_mode == "ncu":
        from .ncu_summary import (
            NcuReportUnavailableError,
            format_summary_text,
            summarize_ncu_file,
        )

        # Prefer CSV when both are present (it's faster and never fails on
        # Linux dev boxes that don't have ncu_report installed).
        csv_candidates = sorted(local_dir.glob("*.ncu.csv")) + sorted(
            local_dir.glob("*.ncu-summary.csv")
        )
        rep_candidates = sorted(local_dir.glob("*.ncu-rep")) + sorted(local_dir.glob("*.ncu-repz"))
        targets = csv_candidates if csv_candidates else rep_candidates

        printed = False
        for tgt in targets:
            try:
                summary = summarize_ncu_file(tgt)
            except NcuReportUnavailableError as exc:
                print(
                    f"warning: cannot read {tgt.name}: {exc}",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:
                # The binary parser can fail on a corrupted .ncu-rep
                # (truncated PVC fetch, wrong file extension on a JSON, etc).
                # Don't crash inspect-run; warn and let the install-hint
                # path below print the user-facing fallback instructions.
                print(
                    f"warning: cannot parse {tgt.name}: {exc}",
                    file=sys.stderr,
                )
                continue
            print(format_summary_text(summary))
            if not getattr(args, "no_optimize", False):
                from .ncu_optimize import (
                    analyze_ncu_summary,
                    format_optimization_report,
                )

                print()
                print(format_optimization_report(analyze_ncu_summary(summary)))
            printed = True

        # If we couldn't read any local form AND the caller asked for it,
        # spin up the cluster-side converter, re-fetch, and try again. This
        # lets a Linux dev box / CI runner with no nsight install still get
        # the per-kernel summary on stdout — at the cost of one extra Pod.
        if (
            not printed
            and getattr(args, "convert_ncu", False)
            and rep_candidates
            and fetched.profile_artifact
        ):
            from swordfish.dispatch import NcuConvertError, submit_ncu_convert

            print(
                "no readable summary; submitting cluster-side converter Pod...",
                file=sys.stderr,
            )
            try:
                conv = submit_ncu_convert(
                    job_name=args.name,
                    namespace=args.namespace,
                    pvc=args.convert_ncu_pvc,
                    image=args.convert_ncu_image,
                    context=args.context,
                )
                print(
                    f"converter pod {conv.pod_name} done in {conv.elapsed_seconds:.1f}s; "
                    f"re-fetching {conv.csv_path}",
                    file=sys.stderr,
                )
                # Pull the freshly-written CSV. The PVC name comes from the
                # converter args (we just used it). The path is computed by
                # the converter — pass it through explicitly so we don't
                # depend on rune annotations for this fetch.
                from swordfish.dispatch import fetch_via_rune_submit_get

                csv_local = local_dir / f"{args.name}.ncu-summary.csv"
                csv_bytes = fetch_via_rune_submit_get(
                    name=args.name,
                    namespace=args.namespace,
                    context=args.context,
                    path=str(Path(conv.csv_path).parent),
                    pvc=args.convert_ncu_pvc,
                    artifact=Path(conv.csv_path).name,
                )
                csv_local.write_bytes(csv_bytes)
                print(f"wrote {csv_local}", file=sys.stderr)
                summary = summarize_ncu_file(csv_local)
                print(format_summary_text(summary))
                if not getattr(args, "no_optimize", False):
                    from .ncu_optimize import (
                        analyze_ncu_summary,
                        format_optimization_report,
                    )

                    print()
                    print(format_optimization_report(analyze_ncu_summary(summary)))
                printed = True
            except (NcuConvertError, ResultFetchError) as exc:
                print(
                    f"warning: cluster-side conversion failed: {exc}",
                    file=sys.stderr,
                )

        if not printed and fetched.profile_artifact:
            print(
                "tip: install Nsight Compute to read the .ncu-rep directly:\n"
                "       brew install --cask nvidia-nsight-compute   # Mac\n"
                "     then re-run inspect-run, or invoke directly:\n"
                "       uv run python -m swordfish.runner ncu-summary "
                f"{fetched.profile_artifact}\n"
                "     or convert it on the cluster via:\n"
                "       uv run python -m swordfish.runner convert-ncu "
                f"{args.name}\n"
                "     (or pass --convert-ncu to inspect-run to do it inline)",
                file=sys.stderr,
            )

    if args.open and fetched.profile_artifact:
        # macOS `open` triggers the .ncu-rep / .nsys-rep file association
        # (ncu-ui / nsys-ui) when those Mac clients are installed; otherwise
        # falls back to whatever the user has registered (or no-op + stderr
        # message). On Linux this would be `xdg-open`; we punt on that
        # because the developer loop is Mac-first today.
        if sys.platform != "darwin":
            print(
                f"--open requested but platform is {sys.platform!r}; "
                f"open {fetched.profile_artifact} manually",
                file=sys.stderr,
            )
        else:
            import subprocess

            subprocess.run(["open", str(fetched.profile_artifact)], check=False)
    return 0


def _cmd_ncu_summary(args: argparse.Namespace) -> int:
    """Pretty-print a per-kernel summary of an Nsight Compute report.

    Accepts both NCU CSV exports (`*.ncu.csv`, `ncu --csv` output) AND the
    binary `.ncu-rep` / `.ncu-repz` reports that `--profile-mode=ncu` jobs
    produce. Dispatches by file extension. The binary path needs NVIDIA's
    `ncu_report` Python module — install Nsight Compute (e.g. `brew install
    --cask nvidia-nsight-compute` on Mac) and it's auto-discovered.

    See swordfish.runner.ncu_summary for the parser's rationale and limits.
    """
    from .ncu_summary import (
        NcuReportUnavailableError,
        format_summary_text,
        summarize_ncu_file,
    )

    try:
        summary = summarize_ncu_file(args.csv)
    except NcuReportUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"error: file not found: {args.csv}", file=sys.stderr)
        return 1

    if summary.rows == 0:
        print(
            f"error: no kernels found in {args.csv}; "
            "for a CSV check it's a valid `ncu --csv` export, "
            "for a .ncu-rep check it isn't an empty profile",
            file=sys.stderr,
        )
        return 1
    print(format_summary_text(summary, top_n=args.top, short_name_width=args.name_width))
    if not getattr(args, "no_optimize", False):
        from .ncu_optimize import analyze_ncu_summary, format_optimization_report

        print()
        print(
            format_optimization_report(analyze_ncu_summary(summary, top_kernels=args.optimize_top))
        )
    return 0


def _cmd_convert_ncu(args: argparse.Namespace) -> int:
    """Spin up a CPU-only Pod that runs `ncu --import` to convert a
    cluster-side `.ncu-rep` into a `.ncu-summary.csv` companion on the PVC.

    Useful when the local dev box / CI runner does NOT have Nsight Compute
    installed (so `ncu_report` can't read the binary directly). Mac users with
    `brew install --cask nvidia-nsight-compute` should skip this and just run
    `inspect-run` — `ncu_report` reads the .ncu-rep in-process.
    """
    from swordfish.dispatch import NcuConvertError, submit_ncu_convert

    try:
        result = submit_ncu_convert(
            job_name=args.name,
            namespace=args.namespace,
            pvc=args.pvc,
            image=args.image,
            rep_path=args.rep_path,
            csv_path=args.csv_path,
            timeout_seconds=args.timeout_seconds,
            cleanup=args.cleanup,
            context=args.context,
        )
    except NcuConvertError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"converter pod:    {result.pod_name}", file=sys.stderr)
    print(f"source .ncu-rep:  {result.rep_path}", file=sys.stderr)
    print(f"wrote .ncu.csv:   {result.csv_path}", file=sys.stderr)
    print(f"elapsed:          {result.elapsed_seconds:.1f}s", file=sys.stderr)
    return 0


def _cmd_generate_rune_profiles(args: argparse.Namespace) -> int:
    rendered = render_pack_yaml()
    out: Path = args.out
    if args.check:
        existing = out.read_text() if out.exists() else ""
        if existing != rendered:
            print(
                f"FAIL: {out} is out of sync with swordfish.dispatch.profiles.\n"
                f"      Run: uv run python -m swordfish.runner generate-rune-profiles",
                file=sys.stderr,
            )
            return 1
        print(f"{out} in sync", file=sys.stderr)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"wrote {out}", file=sys.stderr)
    return 0


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

    fsdp = sub.add_parser(
        "liger-fsdp-step",
        help="run one Llama train-step reproduction row (baseline or Liger-patched)",
    )
    fsdp.add_argument("--liger-mode", choices=["baseline", "liger"], default="baseline")
    fsdp.add_argument(
        "--model-source", choices=["reference", "transformers"], default="transformers"
    )
    fsdp.add_argument(
        "--model-preset", choices=sorted(LIGER_FSDP_MODEL_PRESETS), default="llama3-8b"
    )
    fsdp.add_argument("--micro-batch-size", type=int, default=1)
    fsdp.add_argument("--seq-len", type=int, default=2048)
    fsdp.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    fsdp.add_argument("--repeats", type=int, default=3)
    fsdp.add_argument("--warmup", type=int, default=1)
    fsdp.add_argument("--iters", type=int, default=5)
    fsdp.add_argument("--device", default="auto")
    fsdp.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow CPU smoke tests; real FSDP reproduction still requires CUDA",
    )
    fsdp.add_argument("--arch-label", choices=["a100", "h100", "h200"], default=None)
    fsdp.add_argument("--seed", type=int, default=0)
    fsdp.add_argument("--lr", type=float, default=3e-4)
    fsdp.add_argument("--weight-decay", type=float, default=0.1)
    fsdp.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable model gradient checkpointing (default: enabled)",
    )
    fsdp.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="consumed by infra/rune/scripts/swordfish-bench.sh to launch torchrun",
    )
    fsdp.add_argument("--out", type=Path, required=True)
    fsdp.set_defaults(func=_cmd_run_liger_fsdp_step)

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

    submit = sub.add_parser(
        "submit-bench",
        help="dispatch a swordfish benchmark via the rune SDK (replaces "
        "Makefile rune-submit-* shell-outs)",
    )
    submit.add_argument(
        "--workload",
        choices=["gemm", "liger-rmsnorm", "liger-swiglu", "liger-fsdp"],
        required=True,
    )
    submit.add_argument("--arch", choices=["a100", "h100", "h200"], required=True)
    submit.add_argument("--name", default=None, help="override generated job name")
    submit.add_argument("--profile-mode", choices=["ncu", "nsys", "torch"], default=None)
    submit.add_argument(
        "--dry-run",
        choices=["client", "server"],
        default=None,
        help="render-only without submitting (client = local, server = api dry-run)",
    )
    submit.add_argument(
        "--print-yaml",
        action="store_true",
        help="print the rendered Job manifest after submit/dry-run",
    )
    submit.add_argument("--m", type=int, default=4096, help="GEMM only")
    submit.add_argument("--n", type=int, default=4096, help="GEMM only")
    submit.add_argument("--k", type=int, default=4096, help="GEMM only")
    submit.add_argument(
        "--dtype",
        default=None,
        help="GEMM dtype (default fp16) or liger dtype (default bf16)",
    )
    submit.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="default: 5 for GEMM/per-kernel, 3 for liger-fsdp",
    )
    submit.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="default: 10 for GEMM/per-kernel, 1 for liger-fsdp",
    )
    submit.add_argument(
        "--iters",
        type=int,
        default=None,
        help="default: 50 for GEMM/per-kernel, 5 for liger-fsdp",
    )
    submit.add_argument(
        "--liger-mode",
        choices=["baseline", "liger"],
        default="baseline",
        help="liger-fsdp only: baseline or Liger-patched row",
    )
    submit.add_argument(
        "--model-source",
        choices=["reference", "transformers"],
        default="transformers",
        help="liger-fsdp only: model implementation source",
    )
    submit.add_argument(
        "--model-preset",
        choices=sorted(LIGER_FSDP_MODEL_PRESETS),
        default="llama3-8b",
        help="liger-fsdp only: model shape preset",
    )
    submit.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="liger-fsdp only: per-rank batch size",
    )
    submit.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="liger-fsdp only: sequence length",
    )
    submit.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="liger-fsdp only: enable model gradient checkpointing",
    )
    submit.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="liger-fsdp only: torchrun workers per node",
    )
    submit.add_argument(
        "--backend",
        choices=available_gemm_backends(),
        default="torch",
        help="GEMM only",
    )
    submit.add_argument(
        "--result-root",
        default=None,
        help=("PVC directory the in-pod runner writes results to (default: /data/swordfish/week1)"),
    )
    submit.add_argument(
        "--script",
        default=None,
        help=(
            "override the in-pod entrypoint script (default: infra/rune/scripts/swordfish-bench.sh)"
        ),
    )
    submit.set_defaults(func=_cmd_submit_bench)

    experiment_names = [spec.name for spec in list_experiments()]

    list_exp = sub.add_parser(
        "list-experiments",
        help="list repo-approved experiment intents and their profile families",
    )
    list_exp.set_defaults(func=_cmd_list_experiments)

    explain_exp = sub.add_parser(
        "explain-experiment",
        help="show which generated Rune profile and queue an experiment resolves to",
    )
    explain_exp.add_argument("experiment", choices=experiment_names)
    explain_exp.add_argument("--arch", choices=["a100", "h100", "h200"], required=True)
    explain_exp.set_defaults(func=_cmd_explain_experiment)

    submit_exp = sub.add_parser(
        "submit-experiment",
        help="dispatch by experiment intent; placement resolves through repo-approved profiles",
    )
    submit_exp.add_argument("experiment", choices=experiment_names)
    submit_exp.add_argument("--arch", choices=["a100", "h100", "h200"], required=True)
    submit_exp.add_argument("--name", default=None, help="override generated job name")
    submit_exp.add_argument("--profile-mode", choices=["ncu", "nsys", "torch"], default=None)
    submit_exp.add_argument(
        "--dry-run",
        choices=["client", "server"],
        default=None,
        help="render-only without submitting (client = local, server = api dry-run)",
    )
    submit_exp.add_argument(
        "--print-yaml",
        action="store_true",
        help="print the rendered Job manifest after submit/dry-run",
    )
    submit_exp.add_argument("--m", type=int, default=None, help="gemm only")
    submit_exp.add_argument("--n", type=int, default=None, help="gemm only")
    submit_exp.add_argument("--k", type=int, default=None, help="gemm only")
    submit_exp.add_argument(
        "--dtype",
        default=None,
        help="override experiment default dtype",
    )
    submit_exp.add_argument("--repeats", type=int, default=None)
    submit_exp.add_argument("--warmup", type=int, default=None)
    submit_exp.add_argument("--iters", type=int, default=None)
    submit_exp.add_argument(
        "--liger-mode",
        choices=["baseline", "liger"],
        default=None,
        help="liger-fsdp only: baseline or Liger-patched row",
    )
    submit_exp.add_argument(
        "--model-source",
        choices=["reference", "transformers"],
        default=None,
        help="liger-fsdp only: model implementation source",
    )
    submit_exp.add_argument(
        "--model-preset",
        choices=sorted(LIGER_FSDP_MODEL_PRESETS),
        default=None,
        help="liger-fsdp only: model shape preset",
    )
    submit_exp.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="liger-fsdp only: per-rank batch size",
    )
    submit_exp.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="liger-fsdp only: sequence length",
    )
    submit_exp.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="liger-fsdp only: enable model gradient checkpointing",
    )
    submit_exp.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="liger-fsdp only: torchrun workers per node",
    )
    submit_exp.add_argument(
        "--backend",
        choices=available_gemm_backends(),
        default=None,
        help="gemm only",
    )
    submit_exp.add_argument(
        "--result-root",
        default=None,
        help="PVC directory the in-pod runner writes results to",
    )
    submit_exp.add_argument(
        "--script",
        default=None,
        help="override the in-pod entrypoint script",
    )
    submit_exp.set_defaults(func=_cmd_submit_experiment)

    inspect = sub.add_parser(
        "inspect-run",
        help="fetch a finished rune job's artifacts and open the trace locally "
        "(macOS opens .ncu-rep with ncu-ui, .nsys-rep with nsys-ui)",
    )
    inspect.add_argument("name", help="rune job name (the value printed by `submit-bench`)")
    inspect.add_argument(
        "--profile-mode",
        choices=["ncu", "nsys", "torch"],
        default=None,
        help="profile mode the job was submitted with; required to fetch the trace",
    )
    inspect.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="cache directory for fetched artifacts (default: runs/inspect/<name>/)",
    )
    inspect.add_argument(
        "--namespace",
        default="ray",
        help="kubernetes namespace the job ran in (default: ray)",
    )
    inspect.add_argument(
        "--context",
        default=None,
        help="kubectl context (default: current)",
    )
    inspect.add_argument(
        "--pvc",
        default=None,
        help="override the PVC name when fetching the profile artifact "
        "(default: use the recorded annotation)",
    )
    inspect.add_argument(
        "--overwrite",
        action="store_true",
        help="re-fetch even when local copies already exist",
    )
    inspect.add_argument(
        "--no-open",
        dest="open",
        action="store_false",
        help="do not auto-open the trace after fetch (default: open on macOS)",
    )
    inspect.add_argument(
        "--convert-ncu",
        action="store_true",
        help="when only .ncu-rep is fetched and it can't be read locally "
        "(e.g. Linux dev box without Nsight Compute installed), submit a "
        "CPU-only converter Pod to write a .ncu-summary.csv companion on "
        "the PVC, then re-fetch and print the summary. Mac users with "
        "`brew install --cask nvidia-nsight-compute` don't need this.",
    )
    inspect.add_argument(
        "--convert-ncu-image",
        default="voiceagentcr.azurecr.io/swordfish-bench:latest",
        help="image to use for the converter Pod (default: swordfish-bench)",
    )
    inspect.add_argument(
        "--convert-ncu-pvc",
        default="training-nfs",
        help="PVC name the converter Pod should mount at /data (default: training-nfs)",
    )
    inspect.add_argument(
        "--no-optimize",
        action="store_true",
        help="skip the heuristic optimization report (only print the per-kernel table)",
    )
    inspect.set_defaults(func=_cmd_inspect_run, open=True)

    ncu_summary = sub.add_parser(
        "ncu-summary",
        help="pretty-print a per-kernel summary of an Nsight Compute report "
        "(accepts both `*.ncu.csv` from `ncu --csv` and `*.ncu-rep` binary "
        "via NVIDIA's `ncu_report` Python module)",
    )
    ncu_summary.add_argument(
        "csv",
        type=Path,
        help="path to a .ncu.csv, .ncu-summary.csv, or .ncu-rep file",
    )
    ncu_summary.add_argument(
        "--top",
        type=int,
        default=10,
        help="show top N kernels by total time (default: 10)",
    )
    ncu_summary.add_argument(
        "--name-width",
        type=int,
        default=60,
        help="width of the kernel-name column in the table (default: 60)",
    )
    ncu_summary.add_argument(
        "--no-optimize",
        action="store_true",
        help="skip the heuristic optimization report (only print the per-kernel table)",
    )
    ncu_summary.add_argument(
        "--optimize-top",
        type=int,
        default=5,
        help="how many top kernels (by total time) to give per-kernel advice on (default: 5)",
    )
    ncu_summary.set_defaults(func=_cmd_ncu_summary)

    convert_ncu = sub.add_parser(
        "convert-ncu",
        help="spin up a CPU-only Pod that runs `ncu --import` to convert a "
        "cluster-side `.ncu-rep` into a `.ncu-summary.csv` companion. Useful "
        "for CI / Linux runners without local Nsight Compute installed.",
    )
    convert_ncu.add_argument(
        "name",
        help="rune job name whose .ncu-rep at /data/<name>/profile/profile.ncu-rep should be converted",
    )
    convert_ncu.add_argument(
        "--namespace",
        default="ray",
        help="kubernetes namespace to launch the converter Pod in (default: ray)",
    )
    convert_ncu.add_argument(
        "--pvc",
        default="training-nfs",
        help="PVC name to mount at /data (default: training-nfs)",
    )
    convert_ncu.add_argument(
        "--image",
        default="voiceagentcr.azurecr.io/swordfish-bench:latest",
        help="container image with `ncu` baked in (default: swordfish-bench)",
    )
    convert_ncu.add_argument(
        "--rep-path",
        default=None,
        help="explicit absolute path to the .ncu-rep inside /data "
        "(default: /data/<name>/profile/profile.ncu-rep)",
    )
    convert_ncu.add_argument(
        "--csv-path",
        default=None,
        help="explicit absolute path to write the CSV "
        "(default: same dir, extension `.ncu-summary.csv`)",
    )
    convert_ncu.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="how long to wait for the converter Pod to succeed (default: 180s)",
    )
    convert_ncu.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="leave the converter Pod around after success (default: delete)",
    )
    convert_ncu.add_argument(
        "--context",
        default=None,
        help="kubectl --context value (default: current)",
    )
    convert_ncu.set_defaults(func=_cmd_convert_ncu, cleanup=True)

    profiles_cmd = sub.add_parser(
        "generate-rune-profiles",
        help="regenerate the swordfish rune profile pack from the Python "
        "source of truth in swordfish.dispatch.profiles",
    )
    profiles_cmd.add_argument(
        "--out",
        type=Path,
        default=Path(PACK_YAML_PATH),
        help=f"destination path (default: {PACK_YAML_PATH})",
    )
    profiles_cmd.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the on-disk file does not match (no write)",
    )
    profiles_cmd.set_defaults(func=_cmd_generate_rune_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = ["swordfish.runner", *(argv or sys.argv[1:])]
    return args.func(args)
