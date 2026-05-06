"""Tests for swordfish.dispatch — the project-local Python SDK over `rune submit`."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from swordfish.dispatch import (
    DEFAULT_IMAGE,
    DEFAULT_PVC,
    LigerFsdpRun,
    LigerPerkernelMatrix,
    LigerPerkernelRun,
    RuneProfileSecurityError,
    RuneSubmit,
    RuneSubmitGetMissingAnnotationsError,
    TorchGemmRun,
    VectorSumRun,
    build_run_for_experiment,
    fetch_via_rune_submit_get,
    list_experiments,
    resolve_experiment,
)


def test_rune_submit_requires_either_preset_or_profile():
    with pytest.raises(ValueError, match="preset or profile"):
        RuneSubmit(name="foo", script="run.sh")


def test_rune_submit_rejects_both_preset_and_profile():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RuneSubmit(name="foo", preset="azure.x", profile="my-profile", script="run.sh")


def test_rune_submit_renders_preset_path():
    submit = RuneSubmit(
        name="myjob",
        preset="azure.kernel-mode.training.l",
        image="ghcr.io/me/img:tag",
        script="run.sh",
        volumes=["data=pvc:training-nfs"],
        forwarded_args=["liger-perkernel", "--kernel", "rmsnorm"],
    )
    args = submit.to_args(dry_run="client")
    assert args == [
        "rune",
        "submit",
        "myjob",
        "--preset",
        "azure.kernel-mode.training.l",
        "--image",
        "ghcr.io/me/img:tag",
        "--script",
        "run.sh",
        "--volume",
        "data=pvc:training-nfs",
        "-n",
        "ray",
        "--dry-run",
        "client",
        "--",
        "liger-perkernel",
        "--kernel",
        "rmsnorm",
    ]


def test_rune_submit_to_command_is_shell_safe():
    submit = RuneSubmit(
        name="myjob",
        preset="azure.kernel-mode.training.l",
        script="run.sh with spaces.sh",
    )
    cmd = submit.to_command()
    # spaces in path are quoted
    assert "'run.sh with spaces.sh'" in cmd


def test_rune_submit_renders_profile_mode_flag():
    submit = RuneSubmit(name="j", preset="p", script="s.sh", profile_mode="ncu")
    args = submit.to_args()
    assert "--profile-mode" in args
    assert args[args.index("--profile-mode") + 1] == "ncu"


def test_rune_submit_renders_output_flag():
    submit = RuneSubmit(name="j", preset="p", script="s.sh", output="/data/foo/out.json")
    args = submit.to_args()
    assert "--output" in args
    assert args[args.index("--output") + 1] == "/data/foo/out.json"


def test_rune_submit_renders_container_env_flags_alphabetized():
    submit = RuneSubmit(
        name="j",
        preset="p",
        script="s.sh",
        container_env={"FOO": "1", "BAR": "two", "ZED": "3"},
    )
    args = submit.to_args()
    # alphabetized: BAR, FOO, ZED — three pairs of --env flags in that order
    env_pairs = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert env_pairs == ["BAR=two", "FOO=1", "ZED=3"]


def test_rune_submit_rejects_reserved_container_env_keys():
    with pytest.raises(ValueError, match="reserved"):
        RuneSubmit(name="j", preset="p", script="s.sh", container_env={"RUNE_FOO": "x"})
    with pytest.raises(ValueError, match="reserved"):
        RuneSubmit(name="j", preset="p", script="s.sh", container_env={"AIRUN_BAR": "x"})


def test_rune_submit_rejects_unknown_profile_mode():
    with pytest.raises(ValueError, match="profile_mode"):
        RuneSubmit(name="j", preset="p", script="s.sh", profile_mode="vtune")


def test_a100_ncu_submit_preflights_sys_admin_security_context(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(list(args))

        class P:
            returncode = 0
            stdout = "securityContext:\n  capabilities:\n    add:\n    - SYS_ADMIN\n"
            stderr = ""

        return P()

    monkeypatch.setattr("subprocess.run", fake_run)
    run = RuneSubmit(
        name="a100-ncu",
        profile="swordfish-bench-a100-ncu",
        script="bench.sh",
        profile_mode="ncu",
    )

    result = run.submit()

    assert result.submitted
    assert len(calls) == 2
    assert calls[0][-1] == "client"
    assert "--dry-run" in calls[0]
    assert "--dry-run" not in calls[1]


def test_a100_ncu_submit_blocks_when_rune_drops_sys_admin(monkeypatch):
    def fake_run(args, **_kwargs):
        class P:
            returncode = 0
            stdout = "apiVersion: batch/v1\nkind: Job\n"
            stderr = ""

        return P()

    monkeypatch.setattr("subprocess.run", fake_run)
    run = RuneSubmit(
        name="a100-ncu",
        profile="swordfish-bench-a100-ncu",
        script="bench.sh",
        profile_mode="ncu",
    )

    with pytest.raises(RuneProfileSecurityError, match="SYS_ADMIN"):
        run.submit()


def test_liger_perkernel_run_defaults_to_swordfish_profile_pack():
    """Default submit path uses the swordfish-bench-<arch> profile (not raw preset)
    so edits to swordfish/dispatch/profiles.py flow into actual jobs."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    submit = run.to_rune_submit()
    assert submit.profile == "swordfish-bench-a100"
    assert submit.preset is None
    assert submit.image == DEFAULT_IMAGE
    assert submit.volumes == [f"data=pvc:{DEFAULT_PVC}"]


def test_liger_perkernel_run_rejects_raw_preset_without_escape_hatch():
    with pytest.raises(ValueError, match="kernel-team queue contract"):
        LigerPerkernelRun(kernel="rmsnorm", arch="a100", preset="azure.kernel-mode.training.l")


def test_liger_perkernel_run_explicit_preset_requires_escape_hatch():
    """Callers can opt back to the raw preset shortcut, but only explicitly."""
    run = LigerPerkernelRun(
        kernel="rmsnorm",
        arch="a100",
        preset="azure.kernel-mode.training.l",
        allow_raw_preset=True,
    )
    submit = run.to_rune_submit()
    assert submit.preset == "azure.kernel-mode.training.l"
    assert submit.profile is None


def test_liger_perkernel_run_h200_defaults_to_h200_profile():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="h200")
    assert run.resolved_profile == "swordfish-bench-h200"


def test_liger_perkernel_run_name_is_kebab_case_and_normalized():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    assert run.resolved_name == "sf-liger-rmsnorm-a100"


def test_liger_perkernel_run_custom_name_normalizes_underscores():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", name="my_test_run")
    assert run.resolved_name == "my-test-run"


def test_liger_perkernel_run_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="kernel"):
        LigerPerkernelRun(kernel="bogus", arch="a100")


def test_liger_perkernel_run_rejects_unknown_arch():
    with pytest.raises(ValueError, match="arch"):
        LigerPerkernelRun(kernel="rmsnorm", arch="mi300x")


def test_liger_perkernel_run_forwarded_args_include_full_runner_invocation():
    run = LigerPerkernelRun(kernel="swiglu", arch="h200", batch=8, seq=4096)
    forwarded = run.forwarded_args
    assert forwarded[0] == "liger-perkernel"
    assert "--kernel" in forwarded
    assert "swiglu" in forwarded
    assert "--batch" in forwarded
    assert "8" in forwarded
    assert "--out" in forwarded
    assert "/data/swordfish/week1/liger-perkernel/swiglu-h200.json" in forwarded
    assert "--arch-label" in forwarded
    assert "h200" in forwarded


def test_liger_perkernel_run_to_command_renders_full_invocation():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    cmd = run.to_command(dry_run="client")
    assert "rune" in cmd
    assert "submit" in cmd
    assert "sf-liger-rmsnorm-a100" in cmd
    assert "--profile swordfish-bench-a100" in cmd
    assert "--volume data=pvc:training-nfs" in cmd
    assert "--dry-run client" in cmd
    assert "liger-perkernel" in cmd
    assert "/data/swordfish/week1/liger-perkernel/rmsnorm-a100.json" in cmd


def test_liger_perkernel_run_renders_native_output_flag():
    """The --output flag should be present so `rune submit get` can fetch results."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    args = run.to_rune_submit().to_args()
    assert "--output" in args
    assert (
        args[args.index("--output") + 1]
        == "/data/swordfish/week1/liger-perkernel/rmsnorm-a100.json"
    )


def test_liger_perkernel_run_profile_path_disables_preset():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile="swordfish-bench-a100")
    submit = run.to_rune_submit()
    assert submit.profile == "swordfish-bench-a100"
    assert submit.preset is None


def test_liger_perkernel_matrix_default_kernels_a100():
    matrix = LigerPerkernelMatrix()
    runs = matrix.runs()
    assert len(runs) == 2  # rmsnorm + swiglu
    assert {r.kernel for r in runs} == {"rmsnorm", "swiglu"}
    assert {r.arch for r in runs} == {"a100"}


def test_liger_perkernel_matrix_cross_arch():
    matrix = LigerPerkernelMatrix(kernels=["rmsnorm"], archs=["a100", "h100", "h200"])
    runs = matrix.runs()
    assert len(runs) == 3
    assert [r.arch for r in runs] == ["a100", "h100", "h200"]


def test_liger_perkernel_matrix_to_commands_renders_all():
    matrix = LigerPerkernelMatrix(kernels=["rmsnorm"], archs=["a100", "h200"])
    cmds = matrix.to_commands(dry_run="client")
    assert len(cmds) == 2
    assert any("rmsnorm-a100" in c for c in cmds)
    assert any("rmsnorm-h200" in c for c in cmds)


def test_liger_perkernel_run_name_too_long_rejected():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", name="x" * 60)
    with pytest.raises(ValueError, match="too long"):
        run.resolved_name


def test_liger_perkernel_run_profile_mode_passes_through_native_flag():
    """profile_mode must render as the native --profile-mode CLI flag."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile_mode="ncu")
    assert run.resolved_profile == "swordfish-bench-a100-ncu"
    args = run.to_rune_submit().to_args()
    assert "--profile-mode" in args
    assert args[args.index("--profile-mode") + 1] == "ncu"
    # script should NOT be a tempfile wrapper anymore — it's the bench script verbatim
    script_idx = args.index("--script") + 1
    assert "swordfish-bench.sh" in args[script_idx]


def test_liger_perkernel_run_profile_mode_nsys_path_uses_rune_hardcoded_dir():
    """Rune's renderer hardcodes /data/<job-name>/profile/profile.<ext>."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="h200", profile_mode="nsys")
    assert run.profile_out_dir == "/data/sf-liger-rmsnorm-h200/profile"
    assert run.profile_out_path == "/data/sf-liger-rmsnorm-h200/profile/profile.nsys-rep"
    assert run.profile_out_artifact == "profile.nsys-rep"


def test_liger_perkernel_run_profile_mode_ncu_uses_ncu_rep_extension():
    """Native rune --profile-mode=ncu writes binary .ncu-rep, not the legacy .ncu.csv."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile_mode="ncu")
    assert run.profile_out_path == "/data/sf-liger-rmsnorm-a100/profile/profile.ncu-rep"


def test_liger_perkernel_run_profile_mode_rejects_unknown():
    with pytest.raises(ValueError, match="profile_mode"):
        LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile_mode="vtune")


def test_liger_perkernel_run_profile_mode_torch_does_not_pass_to_rune():
    """torch.profiler is in-process — must NOT pipe through rune's --profile-mode.

    Rune only knows ncu/nsys (external CLI wrappers). Passing --profile-mode=torch
    to rune would error or be silently ignored. Instead, the dispatch SDK injects
    SWORDFISH_PROFILE/SWORDFISH_PROFILE_OUT env vars and the bench main wraps
    itself via swordfish.runner.profile_torch.
    """
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile_mode="torch")
    submit = run.to_rune_submit()
    args = submit.to_args()
    # Rune-side flag must be absent
    assert "--profile-mode" not in args
    # Env vars must be injected so the in-pod bash script + python main can opt in
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert "SWORDFISH_PROFILE=torch" in env_args
    assert "SWORDFISH_PROFILE_OUT=/data/sf-liger-rmsnorm-a100/profile/profile.json" in env_args


def test_liger_perkernel_run_profile_mode_torch_uses_json_extension():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="h200", profile_mode="torch")
    assert run.profile_out_dir == "/data/sf-liger-rmsnorm-h200/profile"
    assert run.profile_out_path == "/data/sf-liger-rmsnorm-h200/profile/profile.json"
    assert run.profile_out_artifact == "profile.json"


def test_torch_gemm_run_profile_mode_torch_does_not_pass_to_rune():
    run = TorchGemmRun(arch="a100", profile_mode="torch")
    submit = run.to_rune_submit()
    args = submit.to_args()
    assert "--profile-mode" not in args
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert "SWORDFISH_PROFILE=torch" in env_args
    # GEMM run uses backend in default name
    assert any("SWORDFISH_PROFILE_OUT=/data/sf-gemm-" in e for e in env_args)
    assert any(e.endswith("/profile/profile.json") for e in env_args)


def test_torch_gemm_run_profile_mode_ncu_still_uses_rune_native():
    """Sanity: only torch bypasses rune; ncu/nsys still go through --profile-mode."""
    run = TorchGemmRun(arch="a100", profile_mode="ncu")
    assert run.resolved_profile == "swordfish-bench-a100-ncu"
    args = run.to_rune_submit().to_args()
    assert "--profile-mode" in args
    assert args[args.index("--profile-mode") + 1] == "ncu"
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert not any(e.startswith("SWORDFISH_PROFILE=") for e in env_args)


def test_vector_sum_run_defaults_to_triton_profile_and_unique_output():
    run = VectorSumRun(arch="a100", size=52_428_800)
    submit = run.to_rune_submit()

    assert run.resolved_name == "sf-vectorsum-v2-triton-52428800-a100"
    assert submit.profile == "swordfish-bench-a100"
    assert submit.output == "/data/swordfish/week1/vectorsum-v2/triton-52428800-a100.json"
    assert "--gpu-class" in submit.extra_args
    assert "a100-nvlink-80gb" in submit.extra_args


def test_vector_sum_run_forwarded_args_include_reduction_contract():
    run = VectorSumRun(arch="h200", backend="torch", size=1_638_400, block_size=2048)
    forwarded = run.forwarded_args

    assert forwarded[0] == "bench-vectorsum"
    assert "--backend" in forwarded
    assert forwarded[forwarded.index("--backend") + 1] == "torch"
    assert "--size" in forwarded
    assert forwarded[forwarded.index("--size") + 1] == "1638400"
    assert "--block-size" in forwarded
    assert forwarded[forwarded.index("--block-size") + 1] == "2048"
    assert "--arch-label" in forwarded
    assert "h200" in forwarded


def test_vector_sum_run_profile_mode_torch_uses_in_process_profiler():
    run = VectorSumRun(arch="a100", profile_mode="torch")
    submit = run.to_rune_submit()
    args = submit.to_args()

    assert "--profile-mode" not in args
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert "SWORDFISH_PROFILE=torch" in env_args
    assert any(e.endswith("/profile/profile.json") for e in env_args)


def test_liger_perkernel_run_profile_mode_allows_custom_script():
    """The 'profile_mode only with default bench script' restriction is gone:
    rune wraps any cmd at the renderer level, so custom scripts work."""
    run = LigerPerkernelRun(
        kernel="rmsnorm",
        arch="a100",
        profile_mode="ncu",
        script="experiments/custom.py",
    )
    submit = run.to_rune_submit()
    assert "experiments/custom.py" in str(submit.script)


def test_liger_perkernel_run_no_profile_mode_no_wrapper():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    submit = run.to_rune_submit()
    assert "infra/rune/scripts/swordfish-bench.sh" in str(submit.script)
    assert run.profile_out_path is None
    assert run.profile_out_dir is None


def test_liger_fsdp_run_defaults_to_eight_gpu_a100_profile():
    run = LigerFsdpRun(arch="a100", mode="baseline")
    submit = run.to_rune_submit()

    assert run.resolved_name == "sf-liger-fsdp-llama3-8b-baseline-a100"
    assert submit.profile == "swordfish-fsdp-a100"
    assert submit.preset is None
    assert submit.output == "/data/swordfish/week1/liger-fsdp/llama3-8b-baseline-a100.json"
    assert "--gpu-class" in submit.extra_args
    assert "a100-nvlink-80gb" in submit.extra_args


def test_liger_fsdp_run_rejects_raw_preset_without_escape_hatch():
    with pytest.raises(ValueError, match="kernel-team queue contract"):
        LigerFsdpRun(arch="a100", mode="baseline", preset="azure.research.training.xl")


def test_liger_fsdp_run_forwarded_args_include_torchrun_contract():
    run = LigerFsdpRun(
        arch="a100",
        mode="liger",
        model_source="transformers",
        micro_batch_size=1,
        seq_len=2048,
        nproc_per_node=8,
    )
    forwarded = run.forwarded_args

    assert forwarded[0] == "liger-fsdp-step"
    assert "--liger-mode" in forwarded
    assert "liger" in forwarded
    assert "--model-preset" in forwarded
    assert "llama3-8b" in forwarded
    assert "--nproc-per-node" in forwarded
    assert "8" in forwarded
    assert "--out" in forwarded
    assert "/data/swordfish/week1/liger-fsdp/llama3-8b-liger-a100.json" in forwarded


def test_liger_fsdp_run_custom_name_uses_unique_output_path():
    run = LigerFsdpRun(
        arch="a100",
        mode="liger",
        name="sf-fsdp-liger-knob-tb-no-limit-05031248-a100",
        fsdp_wrap_policy="transformer-block",
        fsdp_limit_all_gathers=False,
    )
    submit = run.to_rune_submit()

    assert (
        run.out_path
        == "/data/swordfish/week1/liger-fsdp/sf-fsdp-liger-knob-tb-no-limit-05031248-a100.json"
    )
    assert submit.output == run.out_path
    assert run.out_path in run.forwarded_args


def test_liger_fsdp_run_to_command_renders_dry_run():
    run = LigerFsdpRun(arch="h100", mode="baseline", name="fsdp_smoke")
    cmd = run.to_command(dry_run="client")

    assert "rune submit fsdp-smoke" in cmd
    assert "--profile swordfish-fsdp-h100" in cmd
    assert "--gpu-class h100-standalone-95gb" in cmd
    assert "liger-fsdp-step" in cmd
    assert "--nproc-per-node 8" in cmd


def test_liger_fsdp_run_profile_mode_torch_uses_in_process_profiler():
    run = LigerFsdpRun(arch="a100", mode="baseline", profile_mode="torch")
    submit = run.to_rune_submit()
    args = submit.to_args()

    assert "--profile-mode" not in args
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert "SWORDFISH_PROFILE=torch" in env_args
    assert any(e.endswith("/profile/profile.json") for e in env_args)


def test_liger_fsdp_run_profile_mode_ncu_uses_a100_ncu_profile():
    run = LigerFsdpRun(arch="a100", mode="baseline", profile_mode="ncu")
    submit = run.to_rune_submit()

    assert submit.profile == "swordfish-fsdp-a100-ncu"
    assert "--profile-mode" in submit.to_args()


def test_liger_fsdp_run_profile_steady_state_sets_runner_and_nsys_capture_env():
    run = LigerFsdpRun(
        arch="a100",
        mode="baseline",
        profile_mode="nsys",
        profile_steady_state=True,
    )
    submit = run.to_rune_submit()
    args = submit.to_args()

    assert "--profile-steady-state" in run.forwarded_args
    env_args = [args[i + 1] for i, arg in enumerate(args) if arg == "--env"]
    assert "NSYS_CAPTURE_RANGE=cudaProfilerApi" in env_args
    assert "NSYS_CAPTURE_RANGE_END=stop" in env_args


def test_liger_fsdp_run_forwarded_args_include_fsdp_overlap_knobs():
    run = LigerFsdpRun(
        arch="a100",
        mode="liger",
        fsdp_wrap_policy="transformer-block",
        fsdp_backward_prefetch="backward-post",
        fsdp_forward_prefetch=True,
        fsdp_limit_all_gathers=False,
    )
    forwarded = run.forwarded_args

    assert "--fsdp-wrap-policy" in forwarded
    assert forwarded[forwarded.index("--fsdp-wrap-policy") + 1] == "transformer-block"
    assert "--fsdp-backward-prefetch" in forwarded
    assert forwarded[forwarded.index("--fsdp-backward-prefetch") + 1] == "backward-post"
    assert "--fsdp-forward-prefetch" in forwarded
    assert "--no-fsdp-limit-all-gathers" in forwarded


# ---------------------------------------------------------------------------
# experiment registry
# ---------------------------------------------------------------------------


def test_experiment_registry_lists_current_workloads():
    specs = {spec.name: spec for spec in list_experiments()}

    assert set(specs) == {
        "gemm",
        "vectorsum-v2",
        "liger-fsdp",
        "liger-rmsnorm",
        "liger-swiglu",
    }
    assert specs["gemm"].profile_family == "bench"
    assert specs["vectorsum-v2"].profile_family == "bench"
    assert specs["liger-fsdp"].profile_family == "fsdp"


def test_resolve_experiment_grounds_profile_and_queue():
    resolved = resolve_experiment("liger-fsdp", "a100")

    assert resolved.profile == "swordfish-fsdp-a100"
    assert resolved.queue_summary.cluster_queue == "team-kernel-mode-reserved-cq"
    assert resolved.queue_summary.local_queue == "kernel-mode-training"
    assert resolved.queue_summary.gpu_class == "a100-nvlink-80gb"
    assert resolved.queue_summary.gpus_per_node == 8
    assert resolved.queue_summary.claim_template == "ds-8gpus"


def test_experiment_registry_rejects_unknown_experiment():
    with pytest.raises(ValueError, match="unknown experiment"):
        resolve_experiment("secret-side-queue", "a100")


def test_build_run_for_experiment_uses_resolved_profile():
    run = build_run_for_experiment("gemm", "h100", {"m": 1024, "n": 2048, "k": 4096})
    submit = run.to_rune_submit()

    assert isinstance(run, TorchGemmRun)
    assert submit.profile == "swordfish-bench-h100"
    assert submit.preset is None
    assert run.m == 1024 and run.n == 2048 and run.k == 4096


def test_build_run_for_vectorsum_experiment_uses_bench_profile_and_overrides():
    run = build_run_for_experiment(
        "vectorsum-v2",
        "h200",
        {
            "backend": "triton",
            "size": 52_428_800,
            "dtype": "fp32",
            "block_size": 2048,
        },
    )
    submit = run.to_rune_submit()

    assert isinstance(run, VectorSumRun)
    assert submit.profile == "swordfish-bench-h200"
    assert submit.preset is None
    assert run.size == 52_428_800
    assert run.block_size == 2048
    assert "--size" in run.forwarded_args
    assert "52428800" in run.forwarded_args


def test_build_run_for_vectorsum_experiment_uses_a100_ncu_profile():
    run = build_run_for_experiment(
        "vectorsum-v2",
        "a100",
        {"profile_mode": "ncu"},
    )
    submit = run.to_rune_submit()

    assert isinstance(run, VectorSumRun)
    assert submit.profile == "swordfish-bench-a100-ncu"
    assert "--profile-mode" in submit.to_args()


def test_build_run_for_liger_fsdp_experiment_uses_a100_ncu_profile():
    run = build_run_for_experiment(
        "liger-fsdp",
        "a100",
        {"mode": "liger", "profile_mode": "ncu"},
    )
    submit = run.to_rune_submit()

    assert isinstance(run, LigerFsdpRun)
    assert submit.profile == "swordfish-fsdp-a100-ncu"
    assert "--profile-mode" in submit.to_args()


def test_build_run_for_liger_fsdp_experiment_uses_fsdp_profile_and_overrides():
    run = build_run_for_experiment(
        "liger-fsdp",
        "a100",
        {
            "mode": "liger",
            "repeats": 1,
            "warmup": 0,
            "iters": 1,
            "profile_steady_state": True,
            "fsdp_wrap_policy": "transformer-block",
            "fsdp_backward_prefetch": "backward-pre",
            "fsdp_forward_prefetch": True,
            "fsdp_limit_all_gathers": False,
            "context": "voice-agent-flex",
            "image": "voiceagentcr.azurecr.io/airun/swordfish-bench:bf92726-dirty",
        },
    )
    submit = run.to_rune_submit()

    assert isinstance(run, LigerFsdpRun)
    assert submit.profile == "swordfish-fsdp-a100"
    assert submit.preset is None
    assert "--liger-mode" in run.forwarded_args
    assert "liger" in run.forwarded_args
    assert "--profile-steady-state" in run.forwarded_args
    assert "--fsdp-wrap-policy" in run.forwarded_args
    assert "transformer-block" in run.forwarded_args
    assert "--fsdp-backward-prefetch" in run.forwarded_args
    assert "backward-pre" in run.forwarded_args
    assert "--fsdp-forward-prefetch" in run.forwarded_args
    assert "--no-fsdp-limit-all-gathers" in run.forwarded_args
    assert submit.context == "voice-agent-flex"
    assert submit.image == "voiceagentcr.azurecr.io/airun/swordfish-bench:bf92726-dirty"
    assert "--context" in submit.to_args()


def test_every_registered_experiment_resolves_to_generated_profile_pack():
    from swordfish.dispatch.profiles import all_profiles

    generated = {profile.name for profile in all_profiles()}
    for spec in list_experiments():
        for arch in spec.allowed_arches:
            resolved = resolve_experiment(spec.name, arch)
            run = build_run_for_experiment(spec.name, arch)
            submit = run.to_rune_submit()

            assert resolved.profile in generated
            assert submit.profile == resolved.profile
            assert submit.preset is None
            assert resolved.queue_summary.local_queue.startswith("kernel-mode-")


def test_build_run_for_experiment_rejects_invalid_override_for_workload():
    with pytest.raises(ValueError, match="overrides not valid"):
        build_run_for_experiment("gemm", "a100", {"seq_len": 2048})


# --- retrieval semantics ---


def _mock_subprocess_run(stdout: bytes, returncode: int = 0, stderr: bytes = b""):
    """Build a CompletedProcess-like return value for subprocess.run mocks."""

    class P:
        pass

    p = P()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_fetch_via_rune_submit_get_returns_bytes_unchanged(monkeypatch):
    """Binary roundtrip: NULs and 0xFF must survive verbatim."""
    raw = b"\x00\xff\x01\x02NCUREP\x7f\x80\xfe\xff" * 16

    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _mock_subprocess_run(stdout=raw)

    monkeypatch.setattr("swordfish.dispatch.results.subprocess.run", fake_run)
    monkeypatch.setattr("swordfish.dispatch.results.shutil.which", lambda _: "/usr/local/bin/rune")

    out = fetch_via_rune_submit_get(name="j1")
    assert out == raw
    # text=True must NOT be passed (would corrupt binary)
    assert "text" not in captured["kwargs"] or captured["kwargs"]["text"] is False


def test_fetch_via_rune_submit_get_passes_path_pvc_artifact_overrides(monkeypatch):
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _mock_subprocess_run(stdout=b"ok")

    monkeypatch.setattr("swordfish.dispatch.results.subprocess.run", fake_run)
    monkeypatch.setattr("swordfish.dispatch.results.shutil.which", lambda _: "/usr/local/bin/rune")

    fetch_via_rune_submit_get(
        name="j1",
        path="/data/j1/profile",
        pvc="training-nfs",
        artifact="profile.ncu-rep",
        context="ctx",
    )
    args = captured["args"]
    assert "--path" in args and args[args.index("--path") + 1] == "/data/j1/profile"
    assert "--pvc" in args and args[args.index("--pvc") + 1] == "training-nfs"
    assert "--artifact" in args and args[args.index("--artifact") + 1] == "profile.ncu-rep"
    assert "--context" in args and args[args.index("--context") + 1] == "ctx"
    assert "--output" in args and args[args.index("--output") + 1] == "raw"


def test_fetch_via_rune_submit_get_raises_typed_error_on_missing_annotations(monkeypatch):
    monkeypatch.setattr(
        "swordfish.dispatch.results.subprocess.run",
        lambda *a, **kw: _mock_subprocess_run(
            stdout=b"",
            returncode=1,
            stderr=b"job ray/legacy-job has no airun.aks.io/result-path; resubmit with --output, or pass --path",
        ),
    )
    monkeypatch.setattr("swordfish.dispatch.results.shutil.which", lambda _: "/usr/local/bin/rune")

    with pytest.raises(RuneSubmitGetMissingAnnotationsError):
        fetch_via_rune_submit_get(name="legacy-job")


def test_fetch_via_rune_submit_get_other_errors_raise_base_class(monkeypatch):
    """Auth/missing-job/etc. errors must NOT be confused with missing-annotations."""
    from swordfish.dispatch import ResultFetchError

    monkeypatch.setattr(
        "swordfish.dispatch.results.subprocess.run",
        lambda *a, **kw: _mock_subprocess_run(
            stdout=b"",
            returncode=1,
            stderr=b'Error from server (NotFound): jobs.batch "ghost" not found',
        ),
    )
    monkeypatch.setattr("swordfish.dispatch.results.shutil.which", lambda _: "/usr/local/bin/rune")

    with pytest.raises(ResultFetchError) as excinfo:
        fetch_via_rune_submit_get(name="ghost")
    # And NOT the subclass — caller should propagate, not fall back.
    assert not isinstance(excinfo.value, RuneSubmitGetMissingAnnotationsError)


def test_liger_perkernel_run_fetch_result_uses_rune_submit_get_by_default(monkeypatch, tmp_path):
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    payload = b'{"hello":"world"}'

    rune_called = {"count": 0}

    def fake_rune_get(*, name, **kwargs):
        rune_called["count"] += 1
        rune_called["kwargs"] = kwargs
        rune_called["name"] = name
        return payload

    def fake_kubectl_cp(**kwargs):
        raise AssertionError("kubectl-cp must not be called when rune-get succeeds")

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr("swordfish.dispatch.results.fetch_result", fake_kubectl_cp)

    target = tmp_path / "out.json"
    result = run.fetch_result(local_path=target)
    assert rune_called["count"] == 1
    assert rune_called["name"] == run.resolved_name
    assert target.read_bytes() == payload
    assert result.local_path == target


def test_liger_perkernel_run_fetch_result_falls_back_when_annotations_missing(
    monkeypatch, tmp_path
):
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")

    def fake_rune_get(**kwargs):
        raise RuneSubmitGetMissingAnnotationsError("legacy job")

    kubectl_called = {"count": 0}

    def fake_kubectl_cp(**kwargs):
        kubectl_called["count"] += 1
        kubectl_called["kwargs"] = kwargs
        from swordfish.dispatch.results import FetchedResult

        target = Path(kwargs["local_path"])
        target.write_bytes(b'{"legacy":1}')
        return FetchedResult(
            name=kwargs["job_name"],
            pod="legacy-pod",
            remote_path=kwargs["remote_path"],
            local_path=target,
        )

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr("swordfish.dispatch.results.fetch_result", fake_kubectl_cp)

    target = tmp_path / "legacy.json"
    result = run.fetch_result(local_path=target)
    assert kubectl_called["count"] == 1
    assert result.pod == "legacy-pod"
    assert target.read_bytes() == b'{"legacy":1}'


def test_liger_perkernel_run_fetch_result_does_not_fall_back_on_other_errors(monkeypatch, tmp_path):
    """Auth/missing-job errors must propagate, not silently retry kubectl-cp."""
    from swordfish.dispatch import ResultFetchError

    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")

    def fake_rune_get(**kwargs):
        raise ResultFetchError("rune submit get failed: NotFound")

    def fake_kubectl_cp(**kwargs):
        raise AssertionError("must not fall back to kubectl-cp on non-annotation errors")

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr("swordfish.dispatch.results.fetch_result", fake_kubectl_cp)

    with pytest.raises(ResultFetchError, match="NotFound"):
        run.fetch_result(local_path=tmp_path / "x.json")


def test_liger_perkernel_run_fetch_result_explicit_pod_skips_rune(monkeypatch, tmp_path):
    """Caller-specified pod should bypass rune-submit-get entirely."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")

    def fake_rune_get(**kwargs):
        raise AssertionError("rune-submit-get must not be called when pod= is set")

    kubectl_called = {"count": 0}

    def fake_kubectl_cp(**kwargs):
        kubectl_called["count"] += 1
        kubectl_called["pod"] = kwargs.get("pod")
        from swordfish.dispatch.results import FetchedResult

        target = Path(kwargs["local_path"])
        target.write_bytes(b"{}")
        return FetchedResult(
            name=kwargs["job_name"],
            pod=kwargs.get("pod") or "auto-pod",
            remote_path=kwargs["remote_path"],
            local_path=target,
        )

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr("swordfish.dispatch.results.fetch_result", fake_kubectl_cp)

    run.fetch_result(local_path=tmp_path / "x.json", pod="my-debug-pod")
    assert kubectl_called["count"] == 1
    assert kubectl_called["pod"] == "my-debug-pod"


def test_liger_perkernel_run_fetch_result_traces_use_explicit_path_pvc_artifact(
    monkeypatch, tmp_path
):
    """Trace fetch must use --path/--pvc/--artifact overrides because the
    profile artifact lives at /data/<name>/profile/, NOT at result-path."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", profile_mode="ncu")

    rune_calls: list[dict] = []
    payload_json = b'{"x":1}'
    payload_trace = b"\x00NCUREP\xff" * 64

    def fake_rune_get(**kwargs):
        rune_calls.append(kwargs)
        # First call (no path/pvc) → JSON. Second call (with path) → trace.
        if "path" in kwargs and kwargs["path"]:
            return payload_trace
        return payload_json

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr(
        "swordfish.dispatch.results.fetch_result",
        lambda **kw: (_ for _ in ()).throw(AssertionError("kubectl must not be used")),
    )

    target = tmp_path / "out.json"
    run.fetch_result(local_path=target, include_traces=True)

    assert len(rune_calls) == 2
    json_call, trace_call = rune_calls
    # JSON call: no overrides
    assert json_call.get("path") is None
    assert json_call.get("pvc") is None
    assert json_call.get("artifact") is None
    # Trace call: explicit overrides pointing at rune's hardcoded location
    assert trace_call["path"] == "/data/sf-liger-rmsnorm-a100/profile"
    assert trace_call["pvc"] == DEFAULT_PVC
    assert trace_call["artifact"] == "profile.ncu-rep"
    # Local files
    assert target.read_bytes() == payload_json
    trace_local = target.with_suffix(".ncu-rep")
    assert trace_local.read_bytes() == payload_trace


# ---------------------------------------------------------------------------
# fetch_run_artifacts (name-only inspection helper)
# ---------------------------------------------------------------------------


def _patch_rune_get_returning(monkeypatch, payloads: dict[str, bytes]):
    """Make fetch_via_rune_submit_get return payloads keyed by 'has overrides'.

    payloads keys: 'json' (no path/pvc/artifact) and 'trace' (with overrides).
    Records each call's kwargs in returned list for assertion.
    """
    calls: list[dict] = []

    def fake_rune_get(**kwargs):
        calls.append(kwargs)
        if kwargs.get("path"):
            return payloads["trace"]
        return payloads["json"]

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    return calls


def test_fetch_run_artifacts_json_only_when_profile_mode_unset(monkeypatch, tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    calls = _patch_rune_get_returning(monkeypatch, {"json": b'{"k":1}', "trace": b"NA"})

    fetched = fetch_run_artifacts(name="job1", local_dir=tmp_path)

    assert len(calls) == 1, "no profile_mode → only one rune call (json)"
    assert calls[0].get("path") is None
    assert fetched.result_json == tmp_path / "job1.json"
    assert fetched.profile_artifact is None
    assert fetched.parsed_json == {"k": 1}


def test_fetch_run_artifacts_pulls_ncu_rep_with_explicit_path_and_artifact(monkeypatch, tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    payloads = {"json": b'{"ok":true}', "trace": b"\x00NCUREP\xff" * 32}
    calls = _patch_rune_get_returning(monkeypatch, payloads)

    fetched = fetch_run_artifacts(
        name="sf-liger-rmsnorm-h100",
        profile_mode="ncu",
        local_dir=tmp_path,
        pvc="training-nfs",
    )

    assert len(calls) == 2
    json_call, trace_call = calls
    assert json_call.get("path") is None and json_call.get("artifact") is None
    # Profile fetch uses the rune-hardcoded /data/<name>/profile path:
    assert trace_call["path"] == "/data/sf-liger-rmsnorm-h100/profile"
    assert trace_call["pvc"] == "training-nfs"
    assert trace_call["artifact"] == "profile.ncu-rep"

    assert fetched.result_json.read_bytes() == payloads["json"]
    assert fetched.profile_artifact is not None
    assert fetched.profile_artifact.name == "sf-liger-rmsnorm-h100.ncu-rep"
    assert fetched.profile_artifact.read_bytes() == payloads["trace"]
    assert fetched.profile_mode == "ncu"


def test_fetch_run_artifacts_skips_refetch_when_files_exist(monkeypatch, tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    payloads = {"json": b'{"first":true}', "trace": b"NCUFIRST"}
    calls = _patch_rune_get_returning(monkeypatch, payloads)

    fetch_run_artifacts(name="j", profile_mode="ncu", local_dir=tmp_path)
    assert len(calls) == 2

    # Second call: nothing fetched, files reused
    fetch_run_artifacts(name="j", profile_mode="ncu", local_dir=tmp_path)
    assert len(calls) == 2, "cached files must short-circuit"

    # overwrite=True forces re-fetch
    fetch_run_artifacts(name="j", profile_mode="ncu", local_dir=tmp_path, overwrite=True)
    assert len(calls) == 4


def test_fetch_run_artifacts_uses_nsys_extension_for_nsys_mode(monkeypatch, tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    calls = _patch_rune_get_returning(monkeypatch, {"json": b"{}", "trace": b"NSYSREP"})
    fetched = fetch_run_artifacts(name="j", profile_mode="nsys", local_dir=tmp_path)
    assert calls[1]["artifact"] == "profile.nsys-rep"
    assert fetched.profile_artifact is not None
    assert fetched.profile_artifact.name == "j.nsys-rep"


def test_fetch_run_artifacts_rejects_unknown_profile_mode(tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    with pytest.raises(ValueError, match="profile_mode"):
        fetch_run_artifacts(name="j", profile_mode="bogus", local_dir=tmp_path)


def test_fetch_run_artifacts_propagates_missing_annotations_error(monkeypatch, tmp_path):
    from swordfish.dispatch import fetch_run_artifacts

    def fake_rune_get(**kwargs):
        raise RuneSubmitGetMissingAnnotationsError("legacy job has no result-path")

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    with pytest.raises(RuneSubmitGetMissingAnnotationsError):
        fetch_run_artifacts(name="legacy", local_dir=tmp_path)


# ---------------------------------------------------------------------------
# TorchGemmRun
# ---------------------------------------------------------------------------


def test_torch_gemm_run_defaults_to_swordfish_profile_pack():
    run = TorchGemmRun(arch="a100")
    submit = run.to_rune_submit()
    assert submit.profile == "swordfish-bench-a100"
    assert submit.preset is None


def test_torch_gemm_run_h200_defaults_to_h200_profile():
    run = TorchGemmRun(arch="h200")
    assert run.resolved_profile == "swordfish-bench-h200"


def test_torch_gemm_run_rejects_raw_preset_without_escape_hatch():
    with pytest.raises(ValueError, match="kernel-team queue contract"):
        TorchGemmRun(arch="a100", preset="azure.kernel-mode.training.l")


def test_torch_gemm_run_explicit_preset_requires_escape_hatch():
    run = TorchGemmRun(
        arch="a100",
        preset="azure.kernel-mode.training.l",
        allow_raw_preset=True,
    )
    submit = run.to_rune_submit()
    assert submit.preset == "azure.kernel-mode.training.l"
    assert submit.profile is None


def test_torch_gemm_run_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown arch"):
        TorchGemmRun(arch="mi300x")


def test_torch_gemm_run_forwarded_args_match_runner_subcommand():
    run = TorchGemmRun(arch="h100", m=8192, n=4096, k=2048, dtype="bf16")
    forwarded = run.forwarded_args
    assert forwarded[0] == "run-gemm"
    for flag in ("--backend", "--m", "--n", "--k", "--dtype", "--arch-label", "--out"):
        assert flag in forwarded
    assert "8192" in forwarded
    assert "4096" in forwarded
    assert "2048" in forwarded
    assert "bf16" in forwarded
    assert "h100" in forwarded
    assert "/data/swordfish/week1/torch-gemm-h100.json" in forwarded


def test_torch_gemm_run_renders_native_output_flag():
    run = TorchGemmRun(arch="h100")
    args = run.to_rune_submit().to_args()
    assert "--output" in args
    assert args[args.index("--output") + 1] == "/data/swordfish/week1/torch-gemm-h100.json"


def test_torch_gemm_run_to_command_renders_full_invocation():
    run = TorchGemmRun(arch="a100")
    cmd = run.to_command(dry_run="client")
    assert "rune" in cmd
    assert "submit" in cmd
    assert "sf-gemm-torch-a100" in cmd
    assert "--volume data=pvc:training-nfs" in cmd
    assert "--dry-run client" in cmd
    assert "run-gemm" in cmd


def test_torch_gemm_run_profile_mode_passes_through_native_flag():
    run = TorchGemmRun(arch="h100", profile_mode="ncu")
    args = run.to_rune_submit().to_args()
    assert "--profile-mode" in args
    assert args[args.index("--profile-mode") + 1] == "ncu"


def test_torch_gemm_run_custom_name_normalized():
    run = TorchGemmRun(arch="a100", name="My_Gemm_Run")
    assert run.resolved_name == "my-gemm-run"


def test_torch_gemm_run_preset_and_profile_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchGemmRun(arch="a100", preset="x", profile="y")


def test_torch_gemm_run_image_and_pvc_inherit_dispatch_defaults():
    run = TorchGemmRun(arch="a100")
    submit = run.to_rune_submit()
    assert submit.image == DEFAULT_IMAGE
    assert "data=pvc:" + DEFAULT_PVC in submit.volumes


# ---------------------------------------------------------------------------
# profile YAML sync — the on-disk pack must equal Python source of truth
# ---------------------------------------------------------------------------


def test_swordfish_pack_yaml_in_sync_with_python_source():
    """
    The committed `infra/rune/profiles/swordfish-pack.yaml` is generated from
    `swordfish.dispatch.profiles`. This test fails (with a clear hint) if anyone
    edits one without regenerating the other.
    """
    from swordfish.dispatch.profiles import PACK_YAML_PATH, render_pack_yaml

    expected = render_pack_yaml()
    actual = Path(PACK_YAML_PATH).read_text()
    assert expected == actual, (
        f"{PACK_YAML_PATH} is out of sync with swordfish.dispatch.profiles. Run: make rune-profiles"
    )


def test_swordfish_pack_contains_a100_ncu_profiles_only_for_elevated_capability():
    from swordfish.dispatch.profiles import all_profiles, render_pack_yaml

    generated = {profile.name: profile for profile in all_profiles()}
    assert generated["swordfish-bench-a100-ncu"].sys_admin is True
    assert generated["swordfish-fsdp-a100-ncu"].sys_admin is True
    assert generated["swordfish-bench-a100"].sys_admin is False

    yaml = render_pack_yaml()
    assert "name: swordfish-bench-a100-ncu" in yaml
    assert "name: swordfish-fsdp-a100-ncu" in yaml
    assert "capabilities:\n        add:\n          - SYS_ADMIN" in yaml


# ---------------------------------------------------------------------------
# CLI subcommands: submit-bench and generate-rune-profiles
# ---------------------------------------------------------------------------


def test_submit_bench_cli_constructs_torch_gemm_run(monkeypatch):
    from swordfish.runner.cli import _build_submit_run, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "submit-bench",
            "--workload",
            "gemm",
            "--arch",
            "h100",
            "--m",
            "1024",
            "--n",
            "2048",
            "--k",
            "4096",
            "--dtype",
            "bf16",
            "--name",
            "my-test",
        ]
    )
    run = _build_submit_run(args)
    assert isinstance(run, TorchGemmRun)
    assert run.arch == "h100"
    assert run.m == 1024 and run.n == 2048 and run.k == 4096
    assert run.dtype == "bf16"
    assert run.name == "my-test"


def test_submit_bench_cli_constructs_liger_run_for_rmsnorm(monkeypatch):
    from swordfish.runner.cli import _build_submit_run, build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["submit-bench", "--workload", "liger-rmsnorm", "--arch", "a100", "--profile-mode", "ncu"]
    )
    run = _build_submit_run(args)
    assert isinstance(run, LigerPerkernelRun)
    assert run.kernel == "rmsnorm"
    assert run.dtype == "bf16"
    assert run.profile_mode == "ncu"


def test_submit_bench_cli_propagates_result_root_and_script(monkeypatch):
    from swordfish.runner.cli import _build_submit_run, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "submit-bench",
            "--workload",
            "gemm",
            "--arch",
            "h100",
            "--result-root",
            "/data/custom",
            "--script",
            "/tmp/my.sh",
        ]
    )
    run = _build_submit_run(args)
    assert run.result_root == "/data/custom"
    assert str(run.script) == "/tmp/my.sh"
    # Output path should follow the new result_root
    assert run.out_path == "/data/custom/torch-gemm-h100.json"


def test_submit_bench_cli_dry_run_invokes_submit(monkeypatch, capsys):
    from swordfish.runner import cli

    captured: dict = {}

    def fake_submit(self, *, dry_run=None, **kwargs):
        captured["arch"] = self.arch
        captured["dry_run"] = dry_run
        from swordfish.dispatch.rune import RuneSubmitResult

        return RuneSubmitResult(
            name=self.resolved_name,
            args=["rune"],
            rendered_yaml="kind: Job",
            stdout="kind: Job",
            stderr="",
        )

    monkeypatch.setattr(TorchGemmRun, "submit", fake_submit)
    rc = cli.main(["submit-bench", "--workload", "gemm", "--arch", "a100", "--dry-run", "client"])
    assert rc == 0
    assert captured == {"arch": "a100", "dry_run": "client"}


def test_list_experiments_cli_prints_registered_experiments(capsys):
    from swordfish.runner import cli

    rc = cli.main(["list-experiments"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "liger-fsdp" in out
    assert "profile-family" in out


def test_explain_experiment_cli_prints_profile_and_queue(capsys):
    from swordfish.runner import cli

    rc = cli.main(["explain-experiment", "liger-fsdp", "--arch", "a100"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "profile:        swordfish-fsdp-a100" in out
    assert "queue:          team-kernel-mode-reserved-cq/kernel-mode-training" in out


def test_submit_experiment_cli_invokes_resolved_run(monkeypatch):
    from swordfish.runner import cli

    captured: dict = {}

    def fake_submit(self, *, dry_run=None, **kwargs):
        captured["profile"] = self.to_rune_submit().profile
        captured["dry_run"] = dry_run
        captured["mode"] = self.mode
        captured["context"] = self.context
        captured["image"] = self.image
        from swordfish.dispatch.rune import RuneSubmitResult

        return RuneSubmitResult(
            name=self.resolved_name,
            args=["rune"],
            rendered_yaml="kind: Job",
            stdout="kind: Job",
            stderr="",
        )

    monkeypatch.setattr(LigerFsdpRun, "submit", fake_submit)
    rc = cli.main(
        [
            "submit-experiment",
            "liger-fsdp",
            "--arch",
            "a100",
            "--liger-mode",
            "liger",
            "--context",
            "voice-agent-flex",
            "--image",
            "voiceagentcr.azurecr.io/airun/swordfish-bench:bf92726-dirty",
            "--dry-run",
            "client",
        ]
    )

    assert rc == 0
    assert captured == {
        "profile": "swordfish-fsdp-a100",
        "dry_run": "client",
        "mode": "liger",
        "context": "voice-agent-flex",
        "image": "voiceagentcr.azurecr.io/airun/swordfish-bench:bf92726-dirty",
    }


def test_submit_experiment_cli_does_not_accept_raw_preset():
    from swordfish.runner.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["submit-experiment", "liger-fsdp", "--arch", "a100", "--preset", "azure.x"]
        )


def test_generate_rune_profiles_check_passes_when_in_sync(tmp_path):
    from swordfish.dispatch.profiles import render_pack_yaml
    from swordfish.runner import cli

    target = tmp_path / "pack.yaml"
    target.write_text(render_pack_yaml())
    rc = cli.main(["generate-rune-profiles", "--check", "--out", str(target)])
    assert rc == 0


def test_generate_rune_profiles_check_fails_when_drifted(tmp_path):
    from swordfish.runner import cli

    target = tmp_path / "pack.yaml"
    target.write_text("# stale\n")
    rc = cli.main(["generate-rune-profiles", "--check", "--out", str(target)])
    assert rc == 1


def test_generate_rune_profiles_writes_when_no_check(tmp_path):
    from swordfish.dispatch.profiles import render_pack_yaml
    from swordfish.runner import cli

    target = tmp_path / "subdir" / "pack.yaml"
    rc = cli.main(["generate-rune-profiles", "--out", str(target)])
    assert rc == 0
    assert target.read_text() == render_pack_yaml()


# ---------------------------------------------------------------------------
# inspect-run CLI
# ---------------------------------------------------------------------------


def _patch_inspect_helpers(monkeypatch, payloads):
    """Stub the rune fetch and macOS `open` so inspect-run can run on a Mac
    with no rune binary installed.

    payloads maps 'json' / 'trace' to bytes returned by fetch_via_rune_submit_get.
    Returns (rune_calls, open_calls) lists for assertion.
    """
    rune_calls: list[dict] = []
    open_calls: list[list[str]] = []

    def fake_rune_get(**kwargs):
        rune_calls.append(kwargs)
        return payloads["trace"] if kwargs.get("path") else payloads["json"]

    def fake_subprocess_run(args, **kwargs):
        open_calls.append(list(args))

        class P:
            returncode = 0

        return P()

    monkeypatch.setattr("swordfish.dispatch.results.fetch_via_rune_submit_get", fake_rune_get)
    monkeypatch.setattr("subprocess.run", fake_subprocess_run)
    return rune_calls, open_calls


def test_inspect_run_cli_fetches_json_only_when_no_profile_mode(monkeypatch, tmp_path):
    from swordfish.runner import cli

    rune_calls, open_calls = _patch_inspect_helpers(
        monkeypatch, {"json": b'{"x":1}', "trace": b"NOPE"}
    )

    rc = cli.main(["inspect-run", "myjob", "--local-dir", str(tmp_path), "--no-open"])

    assert rc == 0
    assert len(rune_calls) == 1
    assert (tmp_path / "myjob.json").read_bytes() == b'{"x":1}'
    # No profile artifact, no open call regardless of --open
    assert open_calls == []


def test_inspect_run_cli_fetches_ncu_rep_and_opens_on_macos(monkeypatch, tmp_path):
    from swordfish.runner import cli

    payloads = {"json": b'{"k":1}', "trace": b"\x00NCUREP\xff" * 8}
    rune_calls, open_calls = _patch_inspect_helpers(monkeypatch, payloads)
    monkeypatch.setattr("sys.platform", "darwin")

    rc = cli.main(
        [
            "inspect-run",
            "sf-liger-rmsnorm-h100",
            "--profile-mode",
            "ncu",
            "--local-dir",
            str(tmp_path),
            "--pvc",
            "training-nfs",
        ]
    )

    assert rc == 0
    assert len(rune_calls) == 2
    json_call, trace_call = rune_calls
    assert json_call.get("path") is None
    assert trace_call["path"] == "/data/sf-liger-rmsnorm-h100/profile"
    assert trace_call["pvc"] == "training-nfs"
    assert trace_call["artifact"] == "profile.ncu-rep"

    trace_local = tmp_path / "sf-liger-rmsnorm-h100.ncu-rep"
    assert trace_local.read_bytes() == payloads["trace"]
    assert open_calls == [["open", str(trace_local)]]


def test_inspect_run_cli_no_open_flag_skips_subprocess(monkeypatch, tmp_path):
    from swordfish.runner import cli

    rune_calls, open_calls = _patch_inspect_helpers(monkeypatch, {"json": b"{}", "trace": b"X"})
    monkeypatch.setattr("sys.platform", "darwin")

    rc = cli.main(
        [
            "inspect-run",
            "j",
            "--profile-mode",
            "ncu",
            "--local-dir",
            str(tmp_path),
            "--no-open",
        ]
    )

    assert rc == 0
    assert open_calls == [], "--no-open must NOT shell out to `open`"


def test_inspect_run_cli_skips_open_on_non_darwin(monkeypatch, tmp_path, capsys):
    from swordfish.runner import cli

    rune_calls, open_calls = _patch_inspect_helpers(monkeypatch, {"json": b"{}", "trace": b"X"})
    monkeypatch.setattr("sys.platform", "linux")

    rc = cli.main(
        [
            "inspect-run",
            "j",
            "--profile-mode",
            "ncu",
            "--local-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert open_calls == [], "non-Mac platforms must not invoke `open`"
    err = capsys.readouterr().err
    assert "open" in err.lower()  # printed a hint instead


def test_inspect_run_cli_default_local_dir_is_runs_inspect_name(monkeypatch, tmp_path):
    """Without --local-dir, defaults to runs/inspect/<name>/ relative to cwd."""
    from swordfish.runner import cli

    rune_calls, open_calls = _patch_inspect_helpers(monkeypatch, {"json": b"{}", "trace": b""})
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["inspect-run", "myjob", "--no-open"])

    assert rc == 0
    assert (tmp_path / "runs" / "inspect" / "myjob" / "myjob.json").exists()


def test_inspect_run_cli_auto_prints_ncu_summary_when_csv_companion_present(
    monkeypatch, tmp_path, capsys
):
    """If a `.ncu.csv` lands in local_dir alongside the fetched .ncu-rep,
    inspect-run auto-prints the per-kernel summary to stdout. This is the
    happy path once cluster-side dual-emit (or local `ncu --import`) lands.
    """
    from swordfish.runner import cli

    payloads = {"json": b'{"k":1}', "trace": b"\x00NCUREP\xff"}
    _patch_inspect_helpers(monkeypatch, payloads)
    monkeypatch.setattr("sys.platform", "darwin")

    # Pre-seed a CSV companion in the local_dir as if a prior step had
    # converted .ncu-rep → .ncu.csv. inspect-run must pick it up.
    csv_companion = tmp_path / "smoke.ncu.csv"
    csv_companion.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Block Size","Grid Size",'
                '"Metric Name","Metric Unit","Metric Value"',
                '"0","my_kernel","(1,1,1)","(1,1,1)","gpu__time_duration.sum","ns","500"',
                '"0","my_kernel","(1,1,1)","(1,1,1)","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.0"',
            ]
        )
    )

    rc = cli.main(
        [
            "inspect-run",
            "myjob",
            "--profile-mode",
            "ncu",
            "--local-dir",
            str(tmp_path),
            "--no-open",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "NCU summary:" in captured.out
    assert "my_kernel" in captured.out


def test_parse_ncu_csv_full_handles_friendly_metric_names_from_default_ncu():
    """The cluster-side `convert-ncu` runs `ncu --import REP --csv --page details`
    against a .ncu-rep produced by rune's default ncu invocation (no `--section`).
    That path emits human-readable metric names like "Duration" (us) and
    "Compute (SM) Throughput" (%) instead of engine names like
    `gpu__time_duration.sum` (ns) and
    `sm__throughput.avg.pct_of_peak_sustained_elapsed` (%). The parser must
    canonicalize these so total_time, mean/max time, and the SM%/MEM%/DRAM%
    display columns are populated regardless of which capture path was used.
    """
    from swordfish.runner.ncu_summary import parse_ncu_csv_full

    csv_text = "\n".join(
        [
            '"ID","Kernel Name","Block Size","Grid Size",'
            '"Metric Name","Metric Unit","Metric Value"',
            # Duration in us → must be normalized to 500_000 ns.
            '"0","my_kernel","(1,1,1)","(1,1,1)","Duration","us","500"',
            '"0","my_kernel","(1,1,1)","(1,1,1)","Compute (SM) Throughput","%","42.0"',
            '"0","my_kernel","(1,1,1)","(1,1,1)","Memory Throughput","%","17.5"',
            '"0","my_kernel","(1,1,1)","(1,1,1)","DRAM Throughput","%","8.25"',
        ]
    )
    p = Path(tempfile.mkdtemp()) / "friendly.ncu.csv"
    p.write_text(csv_text)

    summary = parse_ncu_csv_full(p)

    assert summary.unique_kernels == 1
    assert summary.total_invocations == 1
    # 500 us -> 500_000 ns
    assert summary.total_time_ns == 500_000.0
    k = summary.kernels[0]
    assert k.short_name == "my_kernel"
    sm = k.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    mem = k.metrics["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]
    dram = k.metrics["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert sm.mean == 42.0
    assert mem.mean == 17.5
    assert dram.mean == 8.25


def test_inspect_run_cli_prints_install_hint_when_only_ncu_rep_is_fetched(
    monkeypatch, tmp_path, capsys
):
    """When --profile-mode=ncu fetches only the .ncu-rep AND we can't parse
    it locally (e.g., ncu_report module missing OR file corrupted), inspect-run
    prints a stderr hint pointing the user at `brew install` + ncu-summary
    instead of silently skipping the readable summary path.
    """
    from swordfish.runner import cli

    payloads = {"json": b"{}", "trace": b"\x00NCUREP\xff"}
    _patch_inspect_helpers(monkeypatch, payloads)
    monkeypatch.setattr("sys.platform", "darwin")
    # Force the parser to raise NcuReportUnavailableError so we exercise the
    # hint path deterministically (otherwise the test depends on whether
    # ncu_report is installed on the dev box).
    from swordfish.runner import ncu_summary as nsum

    def _raise_unavailable(_path):
        raise nsum.NcuReportUnavailableError("test: simulating missing module")

    monkeypatch.setattr(nsum, "summarize_ncu_file", _raise_unavailable)
    monkeypatch.setattr(cli, "summarize_ncu_file", _raise_unavailable, raising=False)

    rc = cli.main(
        [
            "inspect-run",
            "myjob",
            "--profile-mode",
            "ncu",
            "--local-dir",
            str(tmp_path),
            "--no-open",
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "brew install" in err and "ncu-summary" in err


# ---------------------------------------------------------------------------
# bundle-traces CLI / Hermes handoff bundle
# ---------------------------------------------------------------------------


def test_parse_trace_job_spec_accepts_inline_profile_mode():
    from swordfish.runner.trace_bundle import parse_trace_job_spec

    spec = parse_trace_job_spec("sf-job:nsys")

    assert spec.name == "sf-job"
    assert spec.profile_mode == "nsys"


def test_bundle_traces_fetches_jobs_and_writes_manifest_archive(monkeypatch, tmp_path):
    from swordfish.dispatch.results import FetchedRunArtifacts
    from swordfish.runner.trace_bundle import TraceJobSpec, bundle_traces

    calls: list[dict] = []

    def fake_fetch_run_artifacts(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        name = kwargs["name"]
        result_json = local_dir / f"{name}.json"
        result_json.write_text('{"ok": true}\n')
        profile_artifact = None
        if kwargs["profile_mode"] == "ncu":
            profile_artifact = local_dir / f"{name}.ncu-rep"
            profile_artifact.write_bytes(b"NCU")
        return FetchedRunArtifacts(
            name=name,
            local_dir=local_dir,
            result_json=result_json,
            profile_artifact=profile_artifact,
            profile_mode=kwargs["profile_mode"],
        )

    monkeypatch.setattr(
        "swordfish.runner.trace_bundle.fetch_run_artifacts", fake_fetch_run_artifacts
    )

    result = bundle_traces(
        [TraceJobSpec("job-a", "ncu"), TraceJobSpec("job-b", None)],
        bundle_name="handoff",
        local_root=tmp_path,
        namespace="ray",
        context="voice-agent-flex",
        pvc="training-nfs",
    )

    assert [c["name"] for c in calls] == ["job-a", "job-b"]
    assert result.archive_path == tmp_path / "handoff.tar.gz"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schema_version"] == "swordfish.trace-bundle.v1"
    assert manifest["jobs"][0]["remote_profile_path"] == "/data/job-a/profile/profile.ncu-rep"
    assert manifest["jobs"][1]["profile_artifact"] is None
    with tarfile.open(result.archive_path) as archive:
        names = set(archive.getnames())
    assert "handoff/manifest.json" in names
    assert "handoff/job-a/job-a.ncu-rep" in names


def test_bundle_traces_cli_wires_args(monkeypatch, tmp_path, capsys):
    from swordfish.runner import cli

    captured: dict = {}

    def fake_bundle_traces(jobs, **kwargs):
        captured["jobs"] = jobs
        captured.update(kwargs)
        from swordfish.runner.trace_bundle import TraceBundleResult

        bundle_dir = tmp_path / "b"
        manifest = bundle_dir / "manifest.json"
        archive = tmp_path / "b.tar.gz"
        return TraceBundleResult(
            bundle_name="b",
            bundle_dir=bundle_dir,
            manifest_path=manifest,
            archive_path=archive,
            jobs=tuple(jobs),
        )

    monkeypatch.setattr(cli, "bundle_traces", fake_bundle_traces)

    rc = cli.main(
        [
            "bundle-traces",
            "job1:ncu",
            "job2",
            "--profile-mode",
            "nsys",
            "--bundle-name",
            "b",
            "--local-root",
            str(tmp_path),
            "--context",
            "voice-agent-flex",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert [j.name for j in captured["jobs"]] == ["job1", "job2"]
    assert [j.profile_mode for j in captured["jobs"]] == ["ncu", "nsys"]
    assert captured["bundle_name"] == "b"
    assert captured["context"] == "voice-agent-flex"
    assert captured["overwrite"] is True
    assert "archive:" in capsys.readouterr().err


def test_a100_ncu_window_cli_pause_wires_helpers(monkeypatch, capsys):
    from swordfish.runner import cli
    from swordfish.runner.dcgm_window import DcgmWindowStatus

    captured: dict = {}

    def fake_pause(**kwargs):
        captured.update(kwargs)
        return DcgmWindowStatus(
            a100_nodes=("a100-node",),
            a100_exporter_pods=(),
            desired=4,
            ready=4,
            updated=4,
            available=4,
        )

    monkeypatch.setattr(cli, "pause_a100_dcgm", fake_pause)

    rc = cli.main(
        [
            "a100-ncu-window",
            "pause",
            "--context",
            "voice-agent-flex",
            "--timeout-seconds",
            "123",
        ]
    )

    assert rc == 0
    assert captured["context"] == "voice-agent-flex"
    assert captured["timeout_seconds"] == 123
    assert "A100 exporter pods: none" in capsys.readouterr().out


# -----------------------------------------------------------------------------
# Cluster-side .ncu-rep -> .ncu-summary.csv converter (swordfish.dispatch.ncu_convert)
# -----------------------------------------------------------------------------


def _patch_kubectl(monkeypatch, scripted_responses):
    """Stub subprocess.run so the converter sees a sequence of canned kubectl
    responses without ever hitting a real cluster.

    scripted_responses is a list of dicts:
      [{"argv_substr": "apply",          "rc": 0, "stdout": "...", "stderr": ""},
       {"argv_substr": "get pod",        "rc": 0, "stdout": "{json}", "stderr": ""},
       ...]
    Returned in order; if the call doesn't match argv_substr we still return
    the next entry but the test assertion will pick up the mismatch.

    Returns the list of (argv, input) tuples actually called for inspection.
    """
    calls: list[tuple[list[str], str | None]] = []
    iterator = iter(scripted_responses)

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("input")))
        try:
            resp = next(iterator)
        except StopIteration:
            resp = {"rc": 0, "stdout": "", "stderr": ""}

        class P:
            returncode = resp["rc"]
            stdout = resp.get("stdout", "")
            stderr = resp.get("stderr", "")

        return P()

    # Patch BOTH the global subprocess.run and the alias used inside ncu_convert.
    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def test_submit_ncu_convert_happy_path_creates_pod_and_waits_for_succeeded(monkeypatch):
    from swordfish.dispatch import submit_ncu_convert

    # Make the function think kubectl exists.
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    # Make the pod-name suffix deterministic for assertion.
    monkeypatch.setattr("time.time", lambda: 1234567890)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)

    scripted = [
        # apply -f -
        {"rc": 0, "stdout": "pod/sf-ncu-convert-myjob-67890 created\n", "stderr": ""},
        # get pod ... -o json (Succeeded)
        {"rc": 0, "stdout": '{"status":{"phase":"Succeeded"}}', "stderr": ""},
        # delete pod ... --wait=false
        {"rc": 0, "stdout": "pod deleted", "stderr": ""},
    ]
    calls = _patch_kubectl(monkeypatch, scripted)

    result = submit_ncu_convert(
        job_name="myjob",
        namespace="ray",
        pvc="training-nfs",
        timeout_seconds=10,
        poll_interval_seconds=0.0,
    )

    assert result.pod_name.startswith("sf-ncu-convert-myjob-")
    assert result.rep_path == "/data/myjob/profile/profile.ncu-rep"
    assert result.csv_path == "/data/myjob/profile/profile.ncu-summary.csv"

    # Apply was called with stdin containing the rendered Pod YAML.
    apply_argv, apply_stdin = calls[0]
    assert "apply" in apply_argv
    assert apply_stdin is not None
    assert "kind: Pod" in apply_stdin
    assert "claimName: training-nfs" in apply_stdin
    assert "/data/myjob/profile/profile.ncu-rep" in apply_stdin
    assert "/data/myjob/profile/profile.ncu-summary.csv" in apply_stdin

    # Cleanup happened on success.
    assert any("delete" in argv for argv, _ in calls)


def test_submit_ncu_convert_raises_when_kubectl_apply_fails(monkeypatch):
    from swordfish.dispatch import NcuConvertError, submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    scripted = [
        {"rc": 1, "stdout": "", "stderr": "Error from server (NotFound): pvc not found"},
    ]
    _patch_kubectl(monkeypatch, scripted)

    with pytest.raises(NcuConvertError) as excinfo:
        submit_ncu_convert(job_name="myjob", timeout_seconds=5, poll_interval_seconds=0.0)
    assert "kubectl apply failed" in str(excinfo.value)
    assert "pvc not found" in str(excinfo.value)


def test_submit_ncu_convert_raises_on_pod_failed_and_includes_logs(monkeypatch):
    from swordfish.dispatch import NcuConvertError, submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    monkeypatch.setattr("time.time", lambda: 100)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)

    scripted = [
        # apply ok
        {"rc": 0, "stdout": "pod created", "stderr": ""},
        # get pod -> Failed
        {"rc": 0, "stdout": '{"status":{"phase":"Failed"}}', "stderr": ""},
        # logs
        {"rc": 0, "stdout": "ncu: failed to import: file truncated\n", "stderr": ""},
        # delete (cleanup)
        {"rc": 0, "stdout": "pod deleted", "stderr": ""},
    ]
    _patch_kubectl(monkeypatch, scripted)

    with pytest.raises(NcuConvertError) as excinfo:
        submit_ncu_convert(job_name="myjob", timeout_seconds=10, poll_interval_seconds=0.0)
    msg = str(excinfo.value)
    assert "did not succeed" in msg
    assert "file truncated" in msg


def test_submit_ncu_convert_times_out_when_pod_never_terminal(monkeypatch):
    from swordfish.dispatch import NcuConvertError, submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    monkeypatch.setattr("time.time", lambda: 100)
    # Deterministic clock: monotonic returns 0, then 999 (past timeout).
    times = iter([0.0, 0.0, 999.0, 999.0])
    monkeypatch.setattr("time.monotonic", lambda: next(times, 999.0))
    monkeypatch.setattr("time.sleep", lambda _: None)

    scripted = [
        # apply ok
        {"rc": 0, "stdout": "pod created", "stderr": ""},
        # get pod -> Pending (never terminal)
        {"rc": 0, "stdout": '{"status":{"phase":"Pending"}}', "stderr": ""},
        # logs probe after timeout
        {"rc": 0, "stdout": "still pulling image", "stderr": ""},
        # delete
        {"rc": 0, "stdout": "pod deleted", "stderr": ""},
    ]
    _patch_kubectl(monkeypatch, scripted)

    with pytest.raises(NcuConvertError) as excinfo:
        submit_ncu_convert(job_name="myjob", timeout_seconds=5, poll_interval_seconds=0.0)
    assert "did not reach Succeeded" in str(excinfo.value) or "did not succeed" in str(
        excinfo.value
    )


def test_submit_ncu_convert_honors_explicit_rep_and_csv_paths(monkeypatch):
    from swordfish.dispatch import submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    monkeypatch.setattr("time.time", lambda: 100)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)

    scripted = [
        {"rc": 0, "stdout": "pod created", "stderr": ""},
        {"rc": 0, "stdout": '{"status":{"phase":"Succeeded"}}', "stderr": ""},
        {"rc": 0, "stdout": "pod deleted", "stderr": ""},
    ]
    calls = _patch_kubectl(monkeypatch, scripted)

    result = submit_ncu_convert(
        job_name="myjob",
        rep_path="/mnt/scratch/old.ncu-rep",
        csv_path="/data/exports/old.csv",
        timeout_seconds=5,
        poll_interval_seconds=0.0,
    )

    assert result.rep_path == "/mnt/scratch/old.ncu-rep"
    assert result.csv_path == "/data/exports/old.csv"
    apply_stdin = calls[0][1]
    assert "/mnt/scratch/old.ncu-rep" in apply_stdin
    assert "/data/exports/old.csv" in apply_stdin


def test_submit_ncu_convert_skips_cleanup_when_disabled(monkeypatch):
    from swordfish.dispatch import submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: "/usr/local/bin/kubectl")
    monkeypatch.setattr("time.time", lambda: 100)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)

    scripted = [
        {"rc": 0, "stdout": "pod created", "stderr": ""},
        {"rc": 0, "stdout": '{"status":{"phase":"Succeeded"}}', "stderr": ""},
    ]
    calls = _patch_kubectl(monkeypatch, scripted)

    submit_ncu_convert(
        job_name="myjob",
        cleanup=False,
        timeout_seconds=5,
        poll_interval_seconds=0.0,
    )

    assert all("delete" not in argv for argv, _ in calls)


def test_submit_ncu_convert_raises_when_kubectl_not_on_path(monkeypatch):
    from swordfish.dispatch import NcuConvertError, submit_ncu_convert

    monkeypatch.setattr("shutil.which", lambda _b: None)
    with pytest.raises(NcuConvertError) as excinfo:
        submit_ncu_convert(job_name="myjob")
    assert "kubectl" in str(excinfo.value).lower()


def test_convert_ncu_cli_subcommand_invokes_submit(monkeypatch, capsys):
    """The `convert-ncu` CLI subcommand wires args to submit_ncu_convert and
    pretty-prints the result to stderr.
    """
    from swordfish.runner import cli

    captured_kwargs: dict = {}

    def fake_submit(**kwargs):
        captured_kwargs.update(kwargs)
        from swordfish.dispatch.ncu_convert import NcuConvertResult

        return NcuConvertResult(
            pod_name="sf-ncu-convert-myjob-12345",
            rep_path="/data/myjob/profile/profile.ncu-rep",
            csv_path="/data/myjob/profile/profile.ncu-summary.csv",
            elapsed_seconds=12.3,
        )

    monkeypatch.setattr("swordfish.dispatch.submit_ncu_convert", fake_submit)
    monkeypatch.setattr("swordfish.dispatch.ncu_convert.submit_ncu_convert", fake_submit)

    rc = cli.main(
        [
            "convert-ncu",
            "myjob",
            "--namespace",
            "ray",
            "--pvc",
            "custom-pvc",
            "--image",
            "ghcr.io/me/img:latest",
        ]
    )

    assert rc == 0
    assert captured_kwargs["job_name"] == "myjob"
    assert captured_kwargs["namespace"] == "ray"
    assert captured_kwargs["pvc"] == "custom-pvc"
    assert captured_kwargs["image"] == "ghcr.io/me/img:latest"
    err = capsys.readouterr().err
    assert "sf-ncu-convert-myjob-12345" in err
    assert "12.3s" in err
