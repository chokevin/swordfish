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
    TorchGemmRun,
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


def test_liger_perkernel_run_defaults_to_swordfish_profile_pack():
    """Default submit path uses the swordfish-bench-<arch> profile (not raw preset)
    so edits to swordfish/dispatch/profiles.py flow into actual jobs."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100")
    submit = run.to_rune_submit()
    assert submit.profile == "swordfish-bench-a100"
    assert submit.preset is None
    assert submit.image == DEFAULT_IMAGE
    assert submit.volumes == [f"data=pvc:{DEFAULT_PVC}"]


def test_liger_perkernel_run_explicit_preset_overrides_profile_default():
    """Callers can opt back to the raw preset shortcut."""
    run = LigerPerkernelRun(kernel="rmsnorm", arch="a100", preset="azure.kernel-mode.training.l")
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
    args = run.to_rune_submit().to_args()
    assert "--profile-mode" in args
    assert args[args.index("--profile-mode") + 1] == "ncu"
    env_args = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
    assert not any(e.startswith("SWORDFISH_PROFILE=") for e in env_args)


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


def test_torch_gemm_run_explicit_preset_overrides_profile_default():
    run = TorchGemmRun(arch="a100", preset="azure.kernel-mode.training.l")
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
