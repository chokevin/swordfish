"""Tests for swordfish.dispatch — the project-local Python SDK over `rune submit`."""

from __future__ import annotations

from pathlib import Path

import pytest

from swordfish.dispatch import (
    DEFAULT_IMAGE,
    DEFAULT_PVC,
    LigerPerkernelMatrix,
    LigerPerkernelRun,
    RuneSubmit,
    RuneSubmitGetMissingAnnotationsError,
    fetch_via_rune_submit_get,
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
