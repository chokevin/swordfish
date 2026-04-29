from __future__ import annotations

import json

import pytest

from swordfish.runner.airun import (
    AirunGemmConfig,
    render_airun_gemm_job,
    render_airun_preflight_script,
    write_airun_gemm_manifests,
)
from swordfish.kernels.cute import BUILD_COMMAND, cutlass_matmul
from swordfish.kernels.ptx import (
    PTX_VECTOR_ADD_F32,
    ptx_vector_add,
    raw_ptx_blocker,
    torch_vector_add_reference,
)
from swordfish.quant.marlin_triton import (
    dequantize_weight_int4,
    pack_int4_signed,
    quantize_weight_int4_per_group,
    reference_w4a16_matmul,
    run_w4a16_benchmark,
    triton_w4a16_matmul,
    unpack_int4_signed,
)
from swordfish.runner.backends import available_gemm_backends, get_gemm_backend
from swordfish.runner.compare import render_results_comparison
from swordfish.runner.index import build_result_index
from swordfish.runner.liger_perkernel import KERNEL_NAMES as LIGER_KERNEL_NAMES, run_liger_perkernel
from swordfish.runner.matrix import run_gemm_matrix, validate_gemm_matrix_results
from swordfish.runner.schema import (
    TRAINING_SCHEMA_VERSION,
    gpu_class_from_name,
    parse_ncu_csv,
    validate_result_protocol,
    validate_training_result_protocol,
)
from swordfish.runner.status import render_completion_report
from swordfish.runner.torch_gemm import (
    _reference_check,
    run_gemm_benchmark,
    run_torch_gemm,
    write_result,
)
from swordfish.runner.upstream import render_upstream_packet


def test_gpu_class_from_name():
    assert gpu_class_from_name("NVIDIA A100-SXM4-80GB") == "a100"
    assert gpu_class_from_name("NVIDIA H100 80GB HBM3") == "h100"
    assert gpu_class_from_name("NVIDIA H200 NVL") == "h200"
    assert gpu_class_from_name("cpu", fallback="h200") == "h200"
    assert gpu_class_from_name("Apple M3") == "unknown"


def test_run_torch_gemm_cpu_smoke(tmp_path):
    out = tmp_path / "gemm.json"
    result = run_torch_gemm(
        m=8,
        n=8,
        k=8,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
    )
    write_result(result, out)
    saved = json.loads(out.read_text())

    assert saved["schema_version"] == "swordfish.runner.v1"
    assert saved["benchmark"] == "torch_gemm"
    assert validate_result_protocol(saved) == []
    assert saved["config"]["scope"] == "gemm"
    assert saved["config"]["backend"] == "torch"
    assert saved["config"]["shape"] == {"m": 8, "n": 8, "k": 8}
    assert saved["config"]["m"] == 8
    assert saved["env"]["gpu_class"] == "a100"
    assert "git_sha" in saved["env"]
    assert saved["correctness"]["finite_output"] is True
    assert saved["correctness"]["reference_backend"] == "torch"
    assert saved["correctness"]["matches_reference"] is True
    assert saved["correctness"]["max_abs_error"] == 0.0
    assert saved["metrics"]["flops"] == 2 * 8 * 8 * 8
    assert saved["metrics"]["latency"]["mean_ms"] > 0


def test_render_upstream_packet_from_result_json(tmp_path):
    result_path = tmp_path / "gemm.json"
    result = run_torch_gemm(
        m=4,
        n=4,
        k=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
    )
    result["command"] = ["python", "-m", "swordfish.runner", "run-gemm"]
    write_result(result, result_path)

    packet = render_upstream_packet(
        result_path=result_path,
        target="triton",
        ask="Is this benchmark packet useful as a docs/example artifact?",
    )

    assert "# Triton repro:" in packet
    assert "**Target:** Triton" in packet
    assert "**Backend:** torch" in packet
    assert "m=4, n=4, k=4" in packet
    assert "finite_output=True" in packet
    assert "## Result protocol validation\n\n- OK" in packet


def test_render_results_comparison_from_result_jsons(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path, arch in ((first, "a100"), (second, "h100")):
        result = run_torch_gemm(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
            arch_label=arch,
        )
        write_result(result, path)

    table = render_results_comparison([first, second])

    assert "| file | benchmark | backend | gpu |" in table
    assert "first.json" in table
    assert "second.json" in table
    assert "torch_gemm" in table
    assert "speedup_vs_first" in table
    assert "OK" in table


def test_build_result_index_skips_non_results(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    write_result(
        run_torch_gemm(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
            arch_label="a100",
        ),
        result_dir / "torch-gemm-a100.json",
    )
    (result_dir / "config.json").write_text('{"not": "a result"}\n')
    write_result(
        run_torch_gemm(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
            arch_label="a100",
        ),
        result_dir / "torch-gemm-a100.raw.json",
    )

    index = build_result_index(result_dir)

    assert index["schema_version"] == "swordfish.result_index.v1"
    assert index["count"] == 1
    assert index["skipped_count"] == 2
    assert {item["reason"] for item in index["skipped"]} == {
        "not a swordfish result",
        "raw intermediate result",
    }
    row = index["results"][0]
    assert row["file"] == "torch-gemm-a100.json"
    assert row["benchmark"] == "torch_gemm"
    assert row["backend"] == "torch"
    assert row["gpu_class"] == "a100"
    assert row["shape"] == {"k": 4, "m": 4, "n": 4}
    assert row["protocol_errors"] == []

    with_raw = build_result_index(result_dir, include_raw=True)
    assert with_raw["count"] == 2


def test_render_completion_report_blocks_on_missing_arch(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    write_result(
        run_torch_gemm(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
            arch_label="a100",
        ),
        result_dir / "torch-gemm-a100.json",
    )

    report, errors = render_completion_report(
        result_dir=result_dir,
        arch_labels=("a100", "h100"),
        prefix="torch-gemm",
        backend="torch",
        dtype="fp32",
        m=4,
        n=4,
        k=4,
    )

    assert errors == [f"h100: missing result file {result_dir / 'torch-gemm-h100.json'}"]
    assert "**Status:** BLOCKED" in report
    assert "torch-gemm-a100.json" in report
    assert "h100: missing result file" in report


def test_gemm_backend_registry():
    assert available_gemm_backends() == ("cutlass", "torch", "triton")
    assert get_gemm_backend("torch").name == "torch"
    assert get_gemm_backend("cutlass").name == "cutlass"
    with pytest.raises(ValueError, match="unknown GEMM backend"):
        get_gemm_backend("raw-ptx")


def test_result_protocol_reports_missing_fields():
    assert validate_result_protocol({}) == [
        "result.schema_version",
        "result.benchmark",
        "result.config",
        "result.env",
        "result.correctness",
        "result.metrics",
    ]


def test_reference_check_requires_abs_and_relative_tolerance():
    import torch

    from swordfish.runner.backends import GemmState

    state = GemmState(
        a=torch.tensor([[1000.0]], dtype=torch.float32),
        b=torch.tensor([[1000.0]], dtype=torch.float32),
        out=torch.tensor([[1_000_100.0]], dtype=torch.float32),
    )

    check = _reference_check(state, backend_name="triton", dtype="fp16")

    assert check["max_abs_error"] == 100.0
    assert check["max_rel_error"] < check["rtol"]
    assert check["max_abs_error"] > check["atol"]
    assert check["matches_reference"] is False


def test_triton_backend_rejects_cpu_before_importing_triton():
    import torch

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        get_gemm_backend("triton").prepare(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            device=torch.device("cpu"),
            seed=0,
        )


def test_cutlass_backend_rejects_cpu_before_loading_extension():
    import torch

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        get_gemm_backend("cutlass").prepare(
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            device=torch.device("cpu"),
            seed=0,
        )


def test_cutlass_extension_fails_with_build_instructions():
    assert "swordfish.kernels.cute.build" in BUILD_COMMAND
    with pytest.raises(RuntimeError, match="CuTe/CUTLASS extension is not built"):
        cutlass_matmul(None, None, None)  # type: ignore[arg-type]


def test_raw_ptx_vector_add_artifact_and_blocker():
    import torch

    assert ".entry vector_add_f32" in PTX_VECTOR_ADD_F32
    assert "add.rn.f32" in PTX_VECTOR_ADD_F32
    assert "not implemented" in raw_ptx_blocker()

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert torch.equal(torch_vector_add_reference(a, b), torch.tensor([5.0, 7.0, 9.0]))
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        ptx_vector_add(a, b, torch.empty_like(a))


def test_marlin_int4_pack_round_trip_odd_columns():
    import torch

    values = torch.tensor([[-8, -1, 0, 7, 5], [3, -4, 6, -7, 0]], dtype=torch.int8)
    packed = pack_int4_signed(values)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3)
    assert torch.equal(unpack_int4_signed(packed, n=5), values)


def test_marlin_int4_reference_matmul_matches_dequantized_weight():
    import torch

    a = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float32)
    b = torch.tensor(
        [
            [1.0, -2.0],
            [0.25, 3.0],
            [-4.0, 0.5],
            [2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    weight = quantize_weight_int4_per_group(b, group_size=2)
    dequant = dequantize_weight_int4(weight)

    assert weight.packed.shape == (4, 1)
    assert weight.scales.shape == (2, 2)
    assert torch.allclose(reference_w4a16_matmul(a, weight), a @ dequant)

    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        triton_w4a16_matmul(a, weight)


def test_w4a16_reference_benchmark_cpu_smoke():
    result = run_w4a16_benchmark(
        backend="reference",
        m=4,
        n=6,
        k=8,
        group_size=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
    )

    assert result["benchmark"] == "marlin_w4a16_matmul"
    assert validate_result_protocol(result) == []
    assert result["config"]["backend"] == "reference"
    assert result["config"]["shape"] == {"m": 4, "n": 6, "k": 8}
    assert result["env"]["gpu_class"] == "a100"
    assert result["correctness"]["finite_output"] is True
    assert result["correctness"]["matches_reference"] is True
    assert result["correctness"]["output_shape"] == [4, 6]
    assert result["metrics"]["latency"]["mean_ms"] > 0


def test_run_gemm_backend_cpu_smoke(tmp_path):
    result = run_gemm_benchmark(
        backend="torch",
        m=8,
        n=8,
        k=8,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
    )

    assert result["benchmark"] == "torch_gemm"
    assert result["config"]["backend"] == "torch"
    assert result["env"]["gpu_class"] == "h100"
    assert result["correctness"]["finite_output"] is True
    assert result["correctness"]["matches_reference"] is True


def test_run_gemm_matrix_cpu_smoke(tmp_path):
    written = run_gemm_matrix(
        arch_labels=["a100", "h100", "h200"],
        out_dir=tmp_path,
        prefix="torch-gemm",
        m=4,
        n=4,
        k=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        seed=0,
        backend="torch",
        command=["swordfish.runner", "run-gemm-matrix"],
    )

    assert [path.name for path in written] == [
        "torch-gemm-a100.json",
        "torch-gemm-h100.json",
        "torch-gemm-h200.json",
    ]
    assert json.loads((tmp_path / "torch-gemm-h200.json").read_text())["env"]["gpu_class"] == "h200"
    assert (
        validate_gemm_matrix_results(
            arch_labels=["a100", "h100", "h200"],
            result_dir=tmp_path,
            prefix="torch-gemm",
            backend="torch",
            dtype="fp32",
            m=4,
            n=4,
            k=4,
        )
        == []
    )


def test_validate_gemm_matrix_reports_missing_and_wrong_backend(tmp_path):
    run_gemm_matrix(
        arch_labels=["a100"],
        out_dir=tmp_path,
        prefix="torch-gemm",
        m=4,
        n=4,
        k=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        seed=0,
        backend="torch",
    )

    errors = validate_gemm_matrix_results(
        arch_labels=["a100", "h200"],
        result_dir=tmp_path,
        prefix="torch-gemm",
        backend="triton",
        dtype="fp32",
        m=4,
        n=4,
        k=4,
    )

    assert "a100: config.backend must be triton" in errors
    assert f"h200: missing result file {tmp_path / 'torch-gemm-h200.json'}" in errors


def test_validate_gemm_matrix_can_search_recursively(tmp_path):
    nested = tmp_path / "run-001"
    run_gemm_matrix(
        arch_labels=["h100"],
        out_dir=nested,
        prefix="torch-gemm",
        m=4,
        n=4,
        k=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        seed=0,
        backend="torch",
    )

    assert (
        validate_gemm_matrix_results(
            arch_labels=["h100"],
            result_dir=tmp_path,
            prefix="torch-gemm",
            backend="torch",
            dtype="fp32",
            m=4,
            n=4,
            k=4,
            recursive=True,
        )
        == []
    )


def test_validate_gemm_matrix_reports_recursive_duplicates(tmp_path):
    for dirname in ("run-001", "run-002"):
        run_gemm_matrix(
            arch_labels=["h100"],
            out_dir=tmp_path / dirname,
            prefix="torch-gemm",
            m=4,
            n=4,
            k=4,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
            seed=0,
            backend="torch",
        )

    errors = validate_gemm_matrix_results(
        arch_labels=["h100"],
        result_dir=tmp_path,
        prefix="torch-gemm",
        backend="torch",
        recursive=True,
    )

    assert len(errors) == 1
    assert errors[0].startswith("h100: multiple result files named torch-gemm-h100.json")


def test_render_airun_gemm_job_uses_kueue_and_ncu():
    config = AirunGemmConfig.from_mapping(
        {
            "namespace": "ray",
            "image": "example/swordfish:test",
            "output_dir": "/data-nfs/swordfish/week1",
            "python_command": ["python"],
            "resources": {
                "cpu_request": "8",
                "memory_request": "64Gi",
                "cpu_limit": "16",
                "memory_limit": "128Gi",
            },
            "pvc": {"claim_name": "results-pvc", "mount_path": "/data-nfs"},
            "archs": {
                "a100": {"queue": "a100-queue"},
                "h100": {
                    "queue": "h100-queue",
                    "resource_claim_template": "h100-template",
                    "container_security_context": {
                        "capabilities": {"add": ["SYS_ADMIN"]},
                    },
                },
                "h200": {"queue": "h200-queue"},
            },
        }
    )

    manifest = render_airun_gemm_job(config, config.archs[1])

    assert 'kueue.x-k8s.io/queue-name: "h100-queue"' in manifest
    assert "--arch-label h100" in manifest
    assert "--backend torch" in manifest
    assert "import swordfish.runner" in manifest
    assert "python -m swordfish.runner run-gemm" in manifest
    assert "ncu --csv" in manifest
    assert "torch-gemm-h100.profile.raw.json" in manifest
    assert "WARNING: ncu failed; attaching partial profiler output" in manifest
    assert 'resourceClaimTemplateName: "h100-template"' in manifest
    assert "securityContext:" in manifest
    assert "SYS_ADMIN" in manifest
    assert "torch-gemm-h100.ncu.csv" in manifest
    assert "torch-gemm-h100.json" in manifest
    assert 'cpu: "8"' in manifest
    assert 'memory: "128Gi"' in manifest
    assert 'swordfish.dev/backend: "torch"' in manifest


def test_render_airun_preflight_script_checks_nodes_and_blocker():
    config = AirunGemmConfig.from_mapping(
        {
            "namespace": "ray",
            "image": "example/swordfish:test",
            "output_dir": "/data-nfs/swordfish/week1",
            "kubectl_context": "voice-agent-flex",
            "archs": {
                "a100": {"queue": "a100-queue"},
                "h100": {"queue": "h100-queue"},
                "h200": {
                    "queue": "h200-queue",
                    "resource_claim_template": "full-gpu",
                    "node_selector": {"gpu": "h200"},
                },
            },
        }
    )

    script = render_airun_preflight_script(
        config,
        arch_label="h200",
        blocker_pod="sf-gemm-133050-h200-8wr7p",
    )

    assert "KUBECTL=(kubectl --context voice-agent-flex)" in script
    assert "get localqueue h200-queue" in script
    assert "get resourceclaimtemplate full-gpu" in script
    assert "get nodes -l gpu=h200" in script
    assert "Ready and schedulable" in script
    assert "ERROR: no live nodes match selector gpu=h200" in script
    assert "known blocker pod sf-gemm-133050-h200-8wr7p still exists" in script
    assert "is Failed and deletion-marked; continuing because live nodes passed" in script
    assert "do not submit benchmark jobs" in script


def test_render_airun_preflight_script_blocks_a100_ncu_dcgm_conflict():
    config = AirunGemmConfig.from_mapping(
        {
            "namespace": "ray",
            "image": "example/swordfish:test",
            "output_dir": "/data-nfs/swordfish/week1",
            "kubectl_context": "voice-agent-flex",
            "archs": {
                "a100": {
                    "queue": "a100-queue",
                    "node_selector": {"gpu": "a100"},
                    "container_security_context": {
                        "capabilities": {"add": ["SYS_ADMIN"]},
                    },
                },
                "h100": {"queue": "h100-queue"},
                "h200": {"queue": "h200-queue"},
            },
        }
    )

    script = render_airun_preflight_script(config, arch_label="a100")

    assert "== DCGM exporter / NCU profiler conflict ==" in script
    assert "-n gpu-operator get pods" in script
    assert "app=nvidia-dcgm-exporter" in script
    assert "DCGM profiling metrics contend with Nsight Compute" in script
    assert "rerun this preflight, run NCU, then restore DCGM immediately" in script
    assert "ERROR: A100 NCU profiling requires" not in script


def test_render_airun_preflight_script_requires_a100_sys_admin_for_ncu():
    config = AirunGemmConfig.from_mapping(
        {
            "namespace": "ray",
            "image": "example/swordfish:test",
            "output_dir": "/data-nfs/swordfish/week1",
            "archs": {
                "a100": {"queue": "a100-queue", "node_selector": {"gpu": "a100"}},
                "h100": {"queue": "h100-queue"},
                "h200": {"queue": "h200-queue"},
            },
        }
    )

    script = render_airun_preflight_script(config, arch_label="a100")

    assert "ERROR: A100 NCU profiling requires container_security_context" in script
    assert 'capabilities.add: ["SYS_ADMIN"]' in script


def test_render_airun_preflight_cli_returns_script_exit_code(tmp_path, monkeypatch):
    import subprocess

    from swordfish.runner import cli as runner_cli

    config_path = tmp_path / "airun.json"
    config_path.write_text(
        json.dumps(
            {
                "namespace": "ray",
                "image": "example/swordfish:test",
                "output_dir": "/data-nfs/swordfish/week1",
                "archs": {
                    "a100": {"queue": "a100-queue"},
                    "h100": {"queue": "h100-queue"},
                    "h200": {"queue": "h200-queue"},
                },
            }
        )
    )

    def fake_run(cmd, *, check, timeout):
        assert cmd == ["bash", str(tmp_path / "preflight.sh")]
        assert check is False
        assert timeout == runner_cli.KUBECTL_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(cmd, 2)

    monkeypatch.setattr(runner_cli.subprocess, "run", fake_run)

    status = runner_cli.main(
        [
            "render-airun-preflight",
            "--config",
            str(config_path),
            "--arch-label",
            "h200",
            "--out",
            str(tmp_path / "preflight.sh"),
            "--run",
        ]
    )

    assert status == 2


def test_airun_arch_config_rejects_zero_gpu_count():
    with pytest.raises(ValueError, match="at least one GPU"):
        AirunGemmConfig.from_mapping(
            {
                "namespace": "ray",
                "image": "example/swordfish:test",
                "output_dir": "/data-nfs/swordfish/week1",
                "archs": {
                    "a100": {"queue": "a100-queue", "gpu_count": 0},
                    "h100": {"queue": "h100-queue"},
                    "h200": {"queue": "h200-queue"},
                },
            }
        )


def test_write_airun_gemm_manifests_can_filter_arches(tmp_path):
    config = AirunGemmConfig.from_mapping(
        {
            "namespace": "ray",
            "image": "example/swordfish:test",
            "output_dir": "/data-nfs/swordfish/week1",
            "archs": {
                "a100": {"queue": "a100-queue"},
                "h100": {"queue": "h100-queue"},
                "h200": {"queue": "h200-queue"},
            },
        }
    )

    written = write_airun_gemm_manifests(config, tmp_path, arch_labels=["a100", "h100"])

    assert [path.name for path in written] == [
        "swordfish-gemm-a100.yaml",
        "swordfish-gemm-h100.yaml",
    ]
    assert not (tmp_path / "swordfish-gemm-h200.yaml").exists()


def test_parse_ncu_csv_metric_name_format(tmp_path):
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process",
                '"Metric Name","Metric Unit","Metric Value"',
                '"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","77.5"',
                '"dram__throughput.avg.pct_of_peak_sustained_elapsed","%","42.25"',
            ]
        )
    )

    parsed = parse_ncu_csv(csv_path)
    assert parsed["rows"] == 2
    assert parsed["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 77.5
    assert parsed["metrics"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 42.25
    assert parsed["complete"] is False
    assert parsed["missing_metrics"] == [
        "gpu__time_duration.sum",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]


def test_parse_ncu_csv_wide_format(tmp_path):
    csv_path = tmp_path / "ncu-wide.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ID,Kernel Name,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "1,ampere_sgemm_128x64,12345,65.5",
            ]
        )
    )

    parsed = parse_ncu_csv(csv_path)
    assert parsed["rows"] == 1
    assert parsed["metrics"]["gpu__time_duration.sum"] == 12345
    assert parsed["metrics"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"] == 65.5
    assert parsed["complete"] is False
    assert parsed["missing_metrics"] == [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]


def test_parse_ncu_csv_reports_missing_metrics_for_malformed_csv(tmp_path):
    csv_path = tmp_path / "malformed-ncu.csv"
    csv_path.write_text("==PROF== permission denied\nERR_NVGPUCTRPERM\n")

    parsed = parse_ncu_csv(csv_path)

    assert parsed["rows"] == 0
    assert parsed["metrics"] == {}
    assert parsed["missing_metrics"] == [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    assert parsed["complete"] is False


def test_liger_perkernel_rmsnorm_cpu_smoke_skips_liger_half(tmp_path):
    """CPU smoke: baseline runs; Liger half is reported as skipped."""
    result = run_liger_perkernel(
        kernel="rmsnorm",
        batch=2,
        seq=4,
        hidden=8,
        intermediate=16,
        eps=1e-6,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
    )

    assert result["schema_version"] == TRAINING_SCHEMA_VERSION
    assert result["benchmark"] == "liger_perkernel_rmsnorm"
    assert validate_training_result_protocol(result) == []
    assert result["config"]["scope"] == "liger_perkernel"
    assert result["config"]["kernel"] == "rmsnorm"
    assert result["config"]["shape"] == {"batch": 2, "seq": 4, "hidden": 8, "intermediate": 16}
    assert result["config"]["liger"]["applied"] is False

    baseline = result["metrics"]["modes"]["baseline"]
    liger = result["metrics"]["modes"]["liger"]
    assert baseline["skipped"] is False
    assert baseline["forward_ms"]["mean_ms"] >= 0
    assert baseline["backward_ms"]["mean_ms"] >= 0
    assert baseline["finite_output"] is True

    assert liger["skipped"] is True
    assert liger["skip_reason"] is not None
    assert result["metrics"]["deltas"]["forward_speedup"] is None
    assert result["correctness"]["baseline_finite"] is True
    assert result["correctness"]["liger_finite"] is None


def test_liger_perkernel_swiglu_cpu_smoke():
    result = run_liger_perkernel(
        kernel="swiglu",
        batch=2,
        seq=4,
        hidden=8,
        intermediate=16,
        eps=1e-6,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
    )

    assert validate_training_result_protocol(result) == []
    assert result["config"]["kernel"] == "swiglu"
    baseline = result["metrics"]["modes"]["baseline"]
    assert baseline["finite_output"] is True
    assert baseline["forward_ms"]["mean_ms"] >= 0


def test_liger_perkernel_rope_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="rope"):
        run_liger_perkernel(
            kernel="rope",
            batch=1,
            seq=4,
            hidden=8,
            intermediate=16,
            eps=1e-6,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
        )


def test_liger_perkernel_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unknown kernel"):
        run_liger_perkernel(
            kernel="bogus",
            batch=1,
            seq=4,
            hidden=8,
            intermediate=16,
            eps=1e-6,
            dtype="fp32",
            repeats=1,
            warmup=0,
            iters=1,
            device_name="cpu",
            allow_cpu=True,
        )


def test_liger_perkernel_kernel_names_match_advertised_set():
    assert LIGER_KERNEL_NAMES == ("rmsnorm", "swiglu", "rope", "fused_linear_ce")


def test_validate_training_result_protocol_reports_missing_fields():
    bare = {"schema_version": TRAINING_SCHEMA_VERSION}
    errors = validate_training_result_protocol(bare)
    assert any("benchmark" in e for e in errors)
    assert any("config" in e for e in errors)
    assert any("env" in e for e in errors)
    assert any("metrics" in e for e in errors)
    assert any("correctness" in e for e in errors)
