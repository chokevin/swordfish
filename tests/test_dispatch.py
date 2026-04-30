"""Tests for swordfish.dispatch — the project-local Python SDK over `rune submit`."""

from __future__ import annotations

import pytest

from swordfish.dispatch import (
    DEFAULT_IMAGE,
    DEFAULT_PVC,
    LigerPerkernelMatrix,
    LigerPerkernelRun,
    RuneSubmit,
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


def test_liger_perkernel_run_defaults_to_kernel_mode_preset():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    submit = run.to_rune_submit()
    assert submit.preset == "azure.kernel-mode.training.l"
    assert submit.profile is None
    assert submit.image == DEFAULT_IMAGE
    assert submit.volumes == [f"data=pvc:{DEFAULT_PVC}"]


def test_liger_perkernel_run_h200_uses_large_memory_preset():
    run = LigerPerkernelRun(kernel="rmsnorm", arch="h200")
    assert run.resolved_preset == "azure.kernel-mode.large-memory.xl"


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
    assert "--preset azure.kernel-mode.training.l" in cmd
    assert "--volume data=pvc:training-nfs" in cmd
    assert "--dry-run client" in cmd
    assert "liger-perkernel" in cmd
    assert "/data/swordfish/week1/liger-perkernel/rmsnorm-a100.json" in cmd


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
