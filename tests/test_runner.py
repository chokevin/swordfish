from __future__ import annotations

import json
import sys
import types

import pytest

from swordfish.kernels.cute import BUILD_COMMAND, cutlass_matmul
from swordfish.kernels.ptx import (
    PTX_VECTOR_ADD_F32,
    ptx_vector_add,
    raw_ptx_blocker,
    torch_vector_add_reference,
)
from swordfish.kernels.vector_sum import torch_vector_sum_reference, triton_vector_sum
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
from swordfish.runner import liger_fsdp as liger_fsdp_module
from swordfish.runner.liger_perkernel import KERNEL_NAMES as LIGER_KERNEL_NAMES, run_liger_perkernel
from swordfish.runner.liger_fsdp import run_liger_fsdp_step
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
from swordfish.runner.vector_sum import (
    VECTOR_SUM_BENCHMARK_SIZES,
    run_vector_sum_benchmark,
)


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


def test_liger_fsdp_reference_train_step_cpu_smoke():
    result = run_liger_fsdp_step(
        mode="baseline",
        model_source="reference",
        model_preset="tiny",
        micro_batch_size=1,
        seq_len=3,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
        seed=0,
        gradient_checkpointing=False,
    )

    assert result is not None
    assert result["schema_version"] == TRAINING_SCHEMA_VERSION
    assert validate_training_result_protocol(result) == []
    assert result["benchmark"] == "liger_fsdp_train_step"
    assert result["config"]["scope"] == "fsdp_train_step"
    assert result["config"]["distributed_strategy"] == "single_process"
    assert result["config"]["shape"]["global_batch_size"] == 1
    assert result["config"]["shape"]["world_size"] == 1
    assert result["config"]["liger"]["applied"] is False
    assert result["config"]["fsdp"] == {
        "wrap_policy": "root",
        "backward_prefetch": "default",
        "forward_prefetch": False,
        "limit_all_gathers": True,
    }
    assert result["config"]["profile"]["nvtx_ranges"] is True
    assert result["config"]["profile"]["steady_state_cuda_profiler_api"] is False
    assert result["config"]["profile"]["step_phases"] == [
        "zero_grad",
        "forward",
        "loss",
        "backward",
        "optimizer",
    ]
    assert result["correctness"]["finite_loss"] is True
    assert result["metrics"]["tokens_per_second"] > 0
    assert "baseline" in result["metrics"]["modes"]


def test_transformers_liger_fsdp_uses_non_reentrant_checkpointing(monkeypatch):
    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLlamaForCausalLM:
        def __init__(self, config):
            self.config = config
            self.gradient_checkpointing_kwargs = None
            self.to_kwargs = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        def to(self, **kwargs):
            self.to_kwargs = kwargs
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.LlamaConfig = FakeConfig
    fake_transformers.LlamaForCausalLM = FakeLlamaForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model = liger_fsdp_module._build_transformers_llama(
        liger_fsdp_module.MODEL_PRESETS["tiny"],
        device=liger_fsdp_module.torch.device("cpu"),
        dtype=liger_fsdp_module.torch.float32,
        gradient_checkpointing=True,
    )

    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.to_kwargs == {
        "device": liger_fsdp_module.torch.device("cpu"),
        "dtype": liger_fsdp_module.torch.float32,
    }


def test_build_result_index_includes_training_results(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    result = run_liger_fsdp_step(
        mode="baseline",
        model_source="reference",
        model_preset="tiny",
        micro_batch_size=1,
        seq_len=3,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
        seed=0,
        gradient_checkpointing=False,
    )
    assert result is not None
    write_result(result, result_dir / "liger-fsdp-h100.json")

    index = build_result_index(result_dir)

    assert index["count"] == 1
    row = index["results"][0]
    assert row["benchmark"] == "liger_fsdp_train_step"
    assert row["scope"] == "fsdp_train_step"
    assert row["gpu_class"] == "h100"
    assert row["tokens_per_second"] > 0
    assert row["protocol_errors"] == []


def test_render_upstream_packet_accepts_liger_training_result(tmp_path):
    result_path = tmp_path / "liger-fsdp.json"
    result = run_liger_fsdp_step(
        mode="baseline",
        model_source="reference",
        model_preset="tiny",
        micro_batch_size=1,
        seq_len=3,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
        seed=0,
        gradient_checkpointing=False,
    )
    assert result is not None
    result["command"] = ["python", "-m", "swordfish.runner", "liger-fsdp-step"]
    write_result(result, result_path)

    packet = render_upstream_packet(result_path=result_path, target="liger")

    assert "**Target:** Liger Kernel" in packet
    assert "**Benchmark:** liger_fsdp_train_step" in packet
    assert "## Result protocol validation\n\n- OK" in packet


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


def test_vectorsum_v2_benchmark_sizes_match_target_shapes():
    assert VECTOR_SUM_BENCHMARK_SIZES == (
        1_638_400,
        3_276_800,
        6_553_600,
        13_107_200,
        26_214_400,
        52_428_800,
    )


def test_vectorsum_v2_torch_reference_sums_to_fp32_scalar():
    import torch

    x = torch.tensor([1.0, 16_777_216.0, -16_777_216.0], dtype=torch.float32)
    out = torch.empty((1,), dtype=torch.float32)

    result = torch_vector_sum_reference(x, out)

    assert result is out
    assert out.shape == (1,)
    assert out.item() == pytest.approx(1.0)


def test_vectorsum_v2_triton_backend_rejects_cpu_before_launch():
    import torch

    x = torch.ones((8,), dtype=torch.float32)
    out = torch.empty((), dtype=torch.float32)
    partials = torch.empty((1,), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires.*CUDA|requires the triton package"):
        triton_vector_sum(x, out, partials)


def test_submission_exports_custom_kernel():
    import importlib

    submission = importlib.import_module("submission")

    assert callable(submission.custom_kernel)


def test_submission_custom_kernel_returns_scalar_view(monkeypatch):
    import importlib
    import math
    import torch

    submission = importlib.import_module("submission")

    class FakeTriton:
        @staticmethod
        def cdiv(a, b):
            return math.ceil(a / b)

        @staticmethod
        def next_power_of_2(value):
            return 1 << (value - 1).bit_length()

    class FakePartialKernel:
        def __getitem__(self, grid):
            return self

        def __call__(self, *args, **kwargs):
            return None

    class FakeFinalKernel:
        def __getitem__(self, grid):
            return self

        def __call__(self, partials, output, *args, **kwargs):
            output.reshape(-1)[0].fill_(3.0)

    monkeypatch.setattr(submission, "triton", FakeTriton)
    monkeypatch.setattr(submission, "_partial_sum_kernel", FakePartialKernel())
    monkeypatch.setattr(submission, "_final_sum_kernel", FakeFinalKernel())
    monkeypatch.setattr(submission, "_PARTIALS", None)
    monkeypatch.setattr(submission, "_PARTIALS_DEVICE", None)
    monkeypatch.setattr(submission, "_PARTIALS_N", 0)
    monkeypatch.setattr(submission, "_N_PARTIALS", 0)
    monkeypatch.setattr(submission, "_FINAL_BLOCK_SIZE", 0)
    monkeypatch.setattr(submission, "_GRAPH", None)
    monkeypatch.setattr(submission, "_GRAPH_X", None)
    monkeypatch.setattr(submission, "_GRAPH_OUTPUT", None)
    monkeypatch.setattr(submission, "_GRAPH_DATA", None)
    monkeypatch.setattr(submission, "_GRAPH_PARTIALS", None)
    monkeypatch.setattr(submission, "_GRAPH_N", 0)
    monkeypatch.setattr(submission, "_GRAPH_REPLAY", None)
    monkeypatch.setattr(submission, "_GRAPH_RESULT", None)

    output = torch.empty(1, dtype=torch.float32)
    result = submission.custom_kernel((torch.ones(4, dtype=torch.float32), output))
    cached_partials = submission._PARTIALS
    result_again = submission.custom_kernel((torch.ones(4, dtype=torch.float32), output))

    assert result.shape == torch.Size([])
    assert result.item() == pytest.approx(3.0)
    assert result_again.shape == torch.Size([])
    assert result_again.item() == pytest.approx(3.0)
    assert submission._PARTIALS is cached_partials


def test_submission_does_not_capture_graph_for_new_output(monkeypatch):
    import importlib
    import math
    import types

    submission = importlib.import_module("submission")

    class FakeDevice:
        type = "cuda"
        index = 0

    class FakeTensor:
        def __init__(self, name, numel=1):
            self.name = name
            self.device = FakeDevice()
            self.value = 0.0
            self._numel = numel

        def numel(self):
            return self._numel

        def reshape(self, *args):
            return self

        def __getitem__(self, index):
            return self

    class FakeTriton:
        @staticmethod
        def cdiv(a, b):
            return math.ceil(a / b)

        @staticmethod
        def next_power_of_2(value):
            return 1 << (value - 1).bit_length()

    class FakePartialKernel:
        def __getitem__(self, grid):
            return self

        def __call__(self, *args, **kwargs):
            return None

    class FakeFinalKernel:
        def __getitem__(self, grid):
            return self

        def __call__(self, partials, output, *args, **kwargs):
            output.value = 3.0

    class FakeGraph:
        captures = 0

        def __init__(self):
            FakeGraph.captures += 1

        def replay(self):
            return None

    class FakeGraphContext:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            return self.graph

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(submission, "triton", FakeTriton)
    monkeypatch.setattr(submission, "_partial_sum_kernel", FakePartialKernel())
    monkeypatch.setattr(submission, "_final_sum_kernel", FakeFinalKernel())
    monkeypatch.setattr(
        submission.torch,
        "empty",
        lambda shape, device=None, dtype=None: FakeTensor(
            "empty", shape[0] if isinstance(shape, tuple) else shape
        ),
    )
    monkeypatch.setattr(
        submission.torch,
        "cuda",
        types.SimpleNamespace(
            CUDAGraph=FakeGraph,
            graph=lambda graph: FakeGraphContext(graph),
            synchronize=lambda: None,
        ),
    )

    kernel = submission._make_custom_kernel()
    x = FakeTensor("x", numel=4)
    first_output = FakeTensor("first_output")
    second_output = FakeTensor("second_output")

    kernel((x, first_output))
    kernel((x, second_output))

    assert FakeGraph.captures == 0

    kernel((x, second_output))

    assert FakeGraph.captures == 1


def test_vectorsum_v2_torch_benchmark_cpu_smoke():
    result = run_vector_sum_benchmark(
        backend="torch",
        size=64,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
    )

    assert result["benchmark"] == "vectorsum_v2"
    assert validate_result_protocol(result) == []
    assert result["config"]["scope"] == "vector_sum"
    assert result["config"]["backend"] == "torch"
    assert result["config"]["shape"] == {"size": 64}
    assert result["env"]["gpu_class"] == "a100"
    assert result["correctness"]["finite_output"] is True
    assert result["correctness"]["matches_reference"] is True
    assert result["correctness"]["output_shape"] == [1]
    assert result["metrics"]["elements"] == 64
    assert result["metrics"]["latency"]["mean_ms"] > 0


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


# ---------------------------------------------------------------------------
# ncu_summary: rich per-kernel parsing of NCU CSV exports
#
# The legacy schema.parse_ncu_csv returns 4 aggregate metrics. ncu_summary
# returns per-kernel detail (top-N by time, per-metric distribution stats).
# The week-1 GEMM CSV fixtures are the canonical ground truth: 1236 rows =
# 309 invocations × 4 metrics, with 9 unique kernels per arch.
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402  — placed near the ncu_summary block for locality

from swordfish.runner.ncu_summary import (  # noqa: E402
    _percentile,
    _short_name,
    format_summary_text,
    parse_ncu_csv_full,
)


def _write_ncu_gemm_fixture(tmp_path: Path, name: str, top_kernel: str) -> Path:
    """Write a small long-form NCU CSV shaped like the real GEMM captures.

    The real week-1 NCU CSVs are large run artifacts and are intentionally not
    required for unit tests. This fixture preserves the contracts the parser
    needs to support: multiple invocations per kernel, canonical metric names,
    cuBLAS-style kernel names, and a dominant matmul kernel.
    """

    path = tmp_path / name
    rows = [
        [
            '"ID"',
            '"Kernel Name"',
            '"Block Size"',
            '"Grid Size"',
            '"Metric Name"',
            '"Metric Unit"',
            '"Metric Value"',
        ]
    ]

    def add_invocation(
        inv_id: int,
        kernel: str,
        *,
        duration_ns: float,
        sm: float,
        mem: float,
        dram: float,
        block: str = "(384,1,1)",
        grid: str = "(2,66,1)",
    ) -> None:
        for metric, unit, value in [
            ("gpu__time_duration.sum", "ns", duration_ns),
            ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", sm),
            ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "%", mem),
            ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", dram),
        ]:
            rows.append(
                [
                    f'"{inv_id}"',
                    f'"{kernel}"',
                    f'"{block}"',
                    f'"{grid}"',
                    f'"{metric}"',
                    f'"{unit}"',
                    f'"{value}"',
                ]
            )

    add_invocation(0, top_kernel, duration_ns=1_000_000, sm=90.0, mem=70.0, dram=16.0)
    add_invocation(1, top_kernel, duration_ns=1_010_000, sm=91.0, mem=71.0, dram=15.0)
    add_invocation(2, top_kernel, duration_ns=990_000, sm=89.0, mem=69.0, dram=17.0)
    add_invocation(
        3, "at::vectorized_elementwise_kernel", duration_ns=3_000, sm=20, mem=72, dram=72
    )
    add_invocation(4, "at::reduce_kernel", duration_ns=2_000, sm=18, mem=75, dram=75)
    add_invocation(
        5,
        "at::distribution_elementwise_grid_stride_kernel",
        duration_ns=1_000,
        sm=74,
        mem=12,
        dram=3,
    )

    path.write_text("\n".join(",".join(row) for row in rows))
    return path


def test_short_name_strips_void_return_type_and_template_args():
    raw = (
        "void at::native::vectorized_elementwise_kernel<2, "
        "at::native::AbsFunctor<float>, at::detail::Array<char *, 2>>"
        "(int, T2, T3)"
    )
    assert _short_name(raw) == "at::native::vectorized_elementwise_kernel"


def test_short_name_strips_unnamed_namespace_inserts():
    raw = (
        "void at::<unnamed>::distribution_elementwise_grid_stride_kernel<float, 4>"
        "(long, at::PhiloxCudaState, T3, T4)"
    )
    assert _short_name(raw) == "at::distribution_elementwise_grid_stride_kernel"


def test_short_name_passes_through_cublas_kernel_names():
    """cuBLAS / cuDNN kernels have no template params and should round-trip."""
    raw = "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN"
    assert _short_name(raw) == raw


def test_short_name_falls_back_to_truncated_prefix_on_garbage():
    # No matchable identifier and no parens — exercise the fallback.
    raw = "<<<unparseable>>>"
    out = _short_name(raw)
    assert out  # non-empty
    assert len(out) <= 80


def test_percentile_handles_single_element_and_endpoints():
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([1.0, 2.0, 3.0], 0) == 1.0
    assert _percentile([1.0, 2.0, 3.0], 100) == 3.0


def test_percentile_linear_interpolation_matches_numpy_default():
    # Standard inclusive percentile: p50 of [1,2,3,4] = 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_parse_ncu_csv_full_against_h100_gemm_fixture(tmp_path):
    """cuBLAS-via-nvjet should dominate at roughly 90% SM throughput."""
    csv_path = _write_ncu_gemm_fixture(
        tmp_path,
        "torch-gemm-h100.ncu.csv",
        "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN",
    )
    summary = parse_ncu_csv_full(csv_path)
    assert summary.rows == 24
    assert summary.unique_kernels == 4
    assert summary.total_invocations == 6
    assert summary.total_time_ns > 0
    # Top kernel must be the cuBLAS H100 SXM5 SGEMM and ~99% of time.
    top = summary.kernels[0]
    assert top.short_name.startswith("nvjet_hsh_")
    assert top.invocations == 3
    pct_top = top.total_time_ns / summary.total_time_ns
    assert pct_top > 0.99, f"expected nvjet to dominate; got {pct_top:.2%}"
    # Per-metric SoL means must be in the right ballpark.
    sm = top.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert 80 < sm.mean < 100
    assert sm.samples == 3


def test_parse_ncu_csv_full_against_a100_gemm_fixture(tmp_path):
    """A100 GEMM: dominated by `ampere_fp16_s16816gemm_*` (cuBLAS pre-Hopper)."""
    csv_path = _write_ncu_gemm_fixture(
        tmp_path,
        "torch-gemm-a100.ncu.csv",
        "ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_nn",
    )
    summary = parse_ncu_csv_full(csv_path)
    assert summary.rows == 24
    top = summary.kernels[0]
    assert "ampere" in top.short_name and "gemm" in top.short_name
    assert top.invocations == 3


def test_parse_ncu_csv_full_against_h200_gemm_fixture_uses_different_nvjet_variant(tmp_path):
    """H200 picks a different cuBLAS tile shape than H100 (256x128 vs 128x256).

    This test exists because catching that difference is exactly the kind of
    insight the tool is supposed to enable.
    """
    h100 = parse_ncu_csv_full(
        _write_ncu_gemm_fixture(
            tmp_path,
            "torch-gemm-h100.ncu.csv",
            "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN",
        )
    )
    h200 = parse_ncu_csv_full(
        _write_ncu_gemm_fixture(
            tmp_path,
            "torch-gemm-h200.ncu.csv",
            "nvjet_hsh_256x128_64x4_1x2_h_bz_coopA_NNT",
        )
    )
    h100_top = h100.kernels[0].short_name
    h200_top = h200.kernels[0].short_name
    assert h100_top.startswith("nvjet_hsh_") and h200_top.startswith("nvjet_hsh_")
    assert h100_top != h200_top, "expected H200 to pick a different cuBLAS variant than H100"


def test_parse_ncu_csv_full_returns_empty_summary_on_no_header(tmp_path):
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("==PROF== permission denied\nERR_NVGPUCTRPERM\n")
    summary = parse_ncu_csv_full(csv_path)
    assert summary.rows == 0
    assert summary.kernels == []
    assert summary.parse_warnings == ["no header row found in CSV"]


def test_parse_ncu_csv_full_records_warnings_for_non_numeric_metric_values(tmp_path):
    csv_path = tmp_path / "fuzz.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Block Size","Grid Size",'
                '"Metric Name","Metric Unit","Metric Value"',
                '"0","kern_a","(1,1,1)","(1,1,1)","gpu__time_duration.sum","ns","100"',
                '"0","kern_a","(1,1,1)","(1,1,1)","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","n/a"',
                '"0","kern_a","(1,1,1)","(1,1,1)","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","not-a-number"',
            ]
        )
    )
    summary = parse_ncu_csv_full(csv_path)
    assert summary.unique_kernels == 1
    # n/a is a known sentinel and is silently skipped (no warning).
    # "not-a-number" is unparseable and IS a warning.
    assert any("not-a-number" in w for w in summary.parse_warnings)
    assert summary.kernels[0].total_time_ns == 100


def test_parse_ncu_csv_full_pivots_multiple_invocations_into_one_kernel_row(tmp_path):
    """Two invocations of the same kernel collapse to one KernelStats row
    with samples=2 and the right mean/max."""
    csv_path = tmp_path / "two-invs.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Block Size","Grid Size",'
                '"Metric Name","Metric Unit","Metric Value"',
                '"0","kern_x","(256,1,1)","(8,1,1)","gpu__time_duration.sum","ns","1000"',
                '"0","kern_x","(256,1,1)","(8,1,1)","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","50.0"',
                '"1","kern_x","(256,1,1)","(8,1,1)","gpu__time_duration.sum","ns","3000"',
                '"1","kern_x","(256,1,1)","(8,1,1)","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","70.0"',
            ]
        )
    )
    summary = parse_ncu_csv_full(csv_path)
    assert summary.unique_kernels == 1
    k = summary.kernels[0]
    assert k.invocations == 2
    assert k.total_time_ns == 4000
    assert k.mean_time_ns == 2000
    assert k.max_time_ns == 3000
    sm = k.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert sm.mean == 60.0
    assert sm.max == 70.0


def test_format_summary_text_renders_top_n_table_and_truncation_notice(tmp_path):
    summary = parse_ncu_csv_full(
        _write_ncu_gemm_fixture(
            tmp_path,
            "torch-gemm-h100.ncu.csv",
            "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN",
        )
    )
    out = format_summary_text(summary, top_n=3)
    # Header lines.
    assert "NCU summary:" in out
    assert "rows=24" in out
    assert "unique_kernels=4" in out
    # Column header.
    assert "kernel" in out and "SM%" in out and "DRAM%" in out
    # Top kernel rendered.
    assert "nvjet_hsh_" in out
    # Truncation notice for the 1 kernel not shown.
    assert "1 more kernels not shown" in out


def test_format_summary_text_handles_empty_summary_gracefully(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")  # no header at all
    summary = parse_ncu_csv_full(csv_path)
    out = format_summary_text(summary, top_n=10)
    assert "rows=0" in out
    assert "unique_kernels=0" in out


# ---------------------------------------------------------------------------
# ncu-summary CLI
# ---------------------------------------------------------------------------


def test_ncu_summary_cli_prints_table_and_returns_zero(tmp_path, capsys):
    from swordfish.runner import cli

    csv_path = _write_ncu_gemm_fixture(
        tmp_path,
        "torch-gemm-h100.ncu.csv",
        "nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_NNN",
    )
    rc = cli.main(
        [
            "ncu-summary",
            str(csv_path),
            "--top",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "NCU summary:" in out
    assert "nvjet_hsh_" in out
    # --top 3 means 1 kernel not shown (4 total).
    assert "1 more kernels not shown" in out


def test_ncu_summary_cli_returns_nonzero_on_unparseable_csv(tmp_path, capsys):
    from swordfish.runner import cli

    bad = tmp_path / "bad.csv"
    bad.write_text("==PROF== permission denied\nERR_NVGPUCTRPERM\n")
    rc = cli.main(["ncu-summary", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no kernels found" in err


# ---------------------------------------------------------------------------
# .ncu-rep binary parser (parse_ncu_rep + summarize_ncu_file)
#
# We can't ship a real .ncu-rep fixture (proprietary NVIDIA format, generated
# by GPU work), so the binary path is exercised against a fake ncu_report
# module that mimics the SWIG wrapper's IContext → IRange → IAction → IMetric
# tree. The fake covers the exact methods parse_ncu_rep calls so contract
# drift in ncu_report would surface as a test failure.
# ---------------------------------------------------------------------------


from swordfish.runner import ncu_summary as _nsum_mod  # noqa: E402


class _FakeMetric:
    def __init__(self, value, unit=""):
        self._v = value
        self._u = unit

    def as_double(self):
        return float(self._v)

    def as_uint64(self):
        return int(self._v)

    def unit(self):
        return self._u


class _FakeAction:
    """Mirrors ncu_report.IAction: name(NameBase) + metric_by_name."""

    NameBase_DEMANGLED = 0  # the constant ncu_summary reads off the class

    def __init__(self, kernel_name, metrics):
        self._name = kernel_name
        self._metrics = metrics  # dict[name -> _FakeMetric]

    def name(self, _base=None):
        return self._name

    def metric_by_name(self, name):
        return self._metrics.get(name)


class _FakeRange:
    def __init__(self, actions):
        self._actions = actions

    def num_actions(self):
        return len(self._actions)

    def action_by_idx(self, i):
        return self._actions[i]


class _FakeContext:
    def __init__(self, ranges):
        self._ranges = ranges

    def num_ranges(self):
        return len(self._ranges)

    def range_by_idx(self, i):
        return self._ranges[i]


class _FakeNcuReport:
    """Stands in for the real ncu_report module."""

    IAction = _FakeAction  # parse_ncu_rep reads NameBase_DEMANGLED off this

    def __init__(self, ctx):
        self._ctx = ctx

    def load_report(self, _path):
        return self._ctx


def _patch_ncu_report(monkeypatch, ctx):
    """Force _import_ncu_report to return a fake module backed by `ctx`."""
    fake = _FakeNcuReport(ctx)
    monkeypatch.setattr(_nsum_mod, "_import_ncu_report", lambda: fake)
    return fake


def test_parse_ncu_rep_pivots_actions_into_per_kernel_summary(monkeypatch, tmp_path):
    """Two kernels, two invocations of one and one of the other. Output
    must match the same shape parse_ncu_csv_full produces for the same
    semantic data."""
    rep_path = tmp_path / "fake.ncu-rep"
    rep_path.write_bytes(b"")  # parse_ncu_rep checks existence, not content

    ctx = _FakeContext(
        [
            _FakeRange(
                [
                    _FakeAction(
                        "void my_kernel<float>(int)",
                        {
                            "gpu__time_duration.sum": _FakeMetric(1000, "ns"),
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                                50.0, "%"
                            ),
                        },
                    ),
                    _FakeAction(
                        "void my_kernel<float>(int)",
                        {
                            "gpu__time_duration.sum": _FakeMetric(3000, "ns"),
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                                70.0, "%"
                            ),
                        },
                    ),
                    _FakeAction(
                        "other_kernel",
                        {
                            "gpu__time_duration.sum": _FakeMetric(500, "ns"),
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                                40.0, "%"
                            ),
                        },
                    ),
                ]
            )
        ]
    )
    _patch_ncu_report(monkeypatch, ctx)

    summary = _nsum_mod.parse_ncu_rep(rep_path)
    assert summary.unique_kernels == 2
    assert summary.total_invocations == 3
    # Sorted by total_time desc — my_kernel is 4000ns vs other_kernel's 500ns.
    top = summary.kernels[0]
    assert top.short_name == "my_kernel"
    assert top.invocations == 2
    assert top.total_time_ns == 4000
    assert top.mean_time_ns == 2000
    sm = top.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    assert sm.mean == 60.0
    assert sm.samples == 2


def test_parse_ncu_rep_returns_warning_on_empty_report(monkeypatch, tmp_path):
    rep_path = tmp_path / "empty.ncu-rep"
    rep_path.write_bytes(b"")
    _patch_ncu_report(monkeypatch, _FakeContext([]))
    summary = _nsum_mod.parse_ncu_rep(rep_path)
    assert summary.unique_kernels == 0
    assert summary.total_invocations == 0
    assert any("empty profile" in w for w in summary.parse_warnings)


def test_parse_ncu_rep_skips_actions_with_blank_kernel_name(monkeypatch, tmp_path):
    rep_path = tmp_path / "fake.ncu-rep"
    rep_path.write_bytes(b"")
    ctx = _FakeContext(
        [
            _FakeRange(
                [
                    _FakeAction("", {"gpu__time_duration.sum": _FakeMetric(99, "ns")}),
                    _FakeAction("real_kernel", {"gpu__time_duration.sum": _FakeMetric(5, "ns")}),
                ]
            )
        ]
    )
    _patch_ncu_report(monkeypatch, ctx)
    summary = _nsum_mod.parse_ncu_rep(rep_path)
    assert summary.unique_kernels == 1
    assert summary.kernels[0].short_name == "real_kernel"


def test_parse_ncu_rep_raises_file_not_found_for_missing_file(monkeypatch, tmp_path):
    _patch_ncu_report(monkeypatch, _FakeContext([]))
    with pytest.raises(FileNotFoundError):
        _nsum_mod.parse_ncu_rep(tmp_path / "does-not-exist.ncu-rep")


def test_summarize_ncu_file_dispatches_on_extension(monkeypatch, tmp_path):
    """`.ncu-rep` and `.ncu-repz` go to the binary parser; everything else
    falls through to the CSV parser."""
    rep_path = tmp_path / "x.ncu-rep"
    rep_path.write_bytes(b"")
    repz_path = tmp_path / "x.ncu-repz"
    repz_path.write_bytes(b"")
    csv_path = tmp_path / "x.ncu.csv"
    csv_path.write_text("")

    calls: list[str] = []
    monkeypatch.setattr(
        _nsum_mod,
        "parse_ncu_rep",
        lambda p: (
            calls.append(("rep", str(p)))
            or _nsum_mod.NcuSummary(
                path=p, rows=0, unique_kernels=0, total_invocations=0, total_time_ns=0, kernels=[]
            )
        ),
    )
    monkeypatch.setattr(
        _nsum_mod,
        "parse_ncu_csv_full",
        lambda p: (
            calls.append(("csv", str(p)))
            or _nsum_mod.NcuSummary(
                path=p, rows=0, unique_kernels=0, total_invocations=0, total_time_ns=0, kernels=[]
            )
        ),
    )

    _nsum_mod.summarize_ncu_file(rep_path)
    _nsum_mod.summarize_ncu_file(repz_path)
    _nsum_mod.summarize_ncu_file(csv_path)

    assert [c[0] for c in calls] == ["rep", "rep", "csv"]


def test_import_ncu_report_raises_actionable_error_when_missing(monkeypatch):
    """When the module isn't importable and no install path matches, raise
    NcuReportUnavailableError with brew/install instructions in the message."""
    # Block the bare import path.
    import builtins

    real_import = builtins.__import__

    def blocking_import(name, *a, **kw):
        if name == "ncu_report":
            raise ImportError("blocked by test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    # Block all default install paths.
    monkeypatch.setattr(_nsum_mod, "_NCU_REPORT_DEFAULT_PATHS", ())
    monkeypatch.delenv("NCU_REPORT_PYTHON_DIR", raising=False)

    with pytest.raises(_nsum_mod.NcuReportUnavailableError) as excinfo:
        _nsum_mod._import_ncu_report()
    msg = str(excinfo.value)
    assert "brew install" in msg
    assert "NCU_REPORT_PYTHON_DIR" in msg


def test_import_ncu_report_respects_env_var_override(monkeypatch, tmp_path):
    """When NCU_REPORT_PYTHON_DIR points at a directory containing an
    ncu_report.py, the override is used. Importlib.spec_from_file_location
    means we don't pollute sys.modules / sys.path across tests.
    """
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "ncu_report.py").write_text("OVERRIDE_MARKER = 'env-var-stub'\n")

    monkeypatch.setenv("NCU_REPORT_PYTHON_DIR", str(stub_dir))

    mod = _nsum_mod._import_ncu_report()
    assert getattr(mod, "OVERRIDE_MARKER", None) == "env-var-stub"
