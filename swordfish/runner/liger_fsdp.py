"""End-to-end Liger Llama FSDP train-step reproduction runner.

This is the Thursday stretch path from Week 1: reproduce Liger's published
Llama-3-8B FSDP1 training-step shape on 8xA100, then reuse the same contract on
other GPU classes when capacity exists. The production path uses
``model_source="transformers"`` plus the in-cluster image that bakes
``transformers`` and ``liger-kernel``. Local tests use ``model_source="reference"``
with the tiny GPT reference so the CLI and result schema stay smoke-testable on
CPU-only machines.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from collections.abc import Iterator
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from swordfish.runner.backends import TORCH_DTYPES
from swordfish.runner.schema import TRAINING_SCHEMA_VERSION, latency_stats
from swordfish.runner.torch_gemm import _resolve_device, capture_env
from swordfish.transformer.config import GPTConfig
from swordfish.transformer.model import GPTLanguageModel

LigerMode = Literal["baseline", "liger"]
ModelSource = Literal["reference", "transformers"]
ModelPreset = Literal["tiny", "llama3-8b"]


@dataclass(frozen=True)
class LlamaTrainSpec:
    """Shape knobs for the train-step reproduction."""

    name: str
    vocab_size: int
    block_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


MODEL_PRESETS: dict[str, LlamaTrainSpec] = {
    "tiny": LlamaTrainSpec(
        name="tiny",
        vocab_size=128,
        block_size=16,
        hidden_size=48,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=12,
        num_key_value_heads=12,
    ),
    # Llama-3 8B architecture shape. The benchmark initializes random weights;
    # no Meta checkpoint download is needed.
    "llama3-8b": LlamaTrainSpec(
        name="llama3-8b",
        vocab_size=128_256,
        block_size=8192,
        hidden_size=4096,
        intermediate_size=14_336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
}


@dataclass(frozen=True)
class DistributedState:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool


def _liger_version() -> str | None:
    try:
        return metadata.version("liger-kernel")
    except metadata.PackageNotFoundError:
        return None


def _apply_liger_llama_patch() -> tuple[str | None, str]:
    """Apply Liger's Llama monkey patch, returning (version, function path)."""

    try:
        import liger_kernel.transformers as liger_transformers  # noqa: PLC0415

        patch = getattr(liger_transformers, "apply_liger_kernel_to_llama")
        patch_path = "liger_kernel.transformers.apply_liger_kernel_to_llama"
    except (ImportError, AttributeError):
        try:
            from liger_kernel.transformers.monkey_patch import (  # noqa: PLC0415
                apply_liger_kernel_to_llama as patch,
            )

            patch_path = "liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_llama"
        except ImportError as exc:
            raise RuntimeError(
                "liger mode requires liger-kernel with apply_liger_kernel_to_llama; "
                "use the swordfish-bench image or install liger-kernel"
            ) from exc

    patch()
    return _liger_version(), patch_path


def _distributed_state(*, device_name: str, allow_cpu: bool) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed FSDP runs require CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            initialized_here = True
        return DistributedState(rank, local_rank, world_size, device, initialized_here)

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedState(
        rank=0, local_rank=0, world_size=1, device=device, initialized_here=False
    )


def _build_reference_model(
    spec: LlamaTrainSpec, *, device: torch.device, dtype: torch.dtype
) -> nn.Module:
    ratio = max(1, round(spec.intermediate_size / spec.hidden_size))
    config = GPTConfig(
        vocab_size=spec.vocab_size,
        block_size=spec.block_size,
        n_layer=spec.num_hidden_layers,
        n_head=spec.num_attention_heads,
        n_embd=spec.hidden_size,
        mlp_ratio=ratio,
        dropout=0.0,
    )
    return GPTLanguageModel(config).to(device=device, dtype=dtype)


def _build_transformers_llama(
    spec: LlamaTrainSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
) -> nn.Module:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "model_source='transformers' requires transformers; use the swordfish-bench "
            "image or pass --model-source reference for local smoke tests"
        ) from exc

    config = LlamaConfig(
        vocab_size=spec.vocab_size,
        max_position_embeddings=spec.block_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
        rope_theta=500_000.0,
    )
    model = LlamaForCausalLM(config)
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError as exc:
            raise RuntimeError(
                "gradient checkpointing for the Liger FSDP reproduction requires "
                "a transformers build that supports non-reentrant checkpoint kwargs"
            ) from exc
    return model.to(device=device, dtype=dtype)


def _maybe_wrap_fsdp(
    model: nn.Module,
    *,
    state: DistributedState,
    dtype: torch.dtype,
) -> tuple[nn.Module, str]:
    if state.world_size == 1:
        return model, "single_process"

    from torch.distributed.fsdp import (  # noqa: PLC0415
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    mixed_precision = None
    if dtype in {torch.float16, torch.bfloat16}:
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    return (
        FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            use_orig_params=True,
        ),
        "FSDP1",
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _reduce_float(value: float, *, device: torch.device, op: Any) -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=op)
    return float(tensor.item())


def _mean_across_ranks(value: float, *, device: torch.device, world_size: int) -> float:
    total = _reduce_float(value, device=device, op=torch.distributed.ReduceOp.SUM)
    return total / world_size


def _max_across_ranks(value: float, *, device: torch.device) -> float:
    return _reduce_float(value, device=device, op=torch.distributed.ReduceOp.MAX)


def _peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024**3)


@contextmanager
def _nvtx_range(name: str, *, device: torch.device) -> Iterator[None]:
    if device.type != "cuda":
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


@contextmanager
def _cuda_profiler_capture(*, device: torch.device, enabled: bool) -> Iterator[None]:
    if not enabled or device.type != "cuda":
        yield
        return
    _sync(device)
    start_result = torch.cuda.cudart().cudaProfilerStart()
    if start_result not in (0, None):
        raise RuntimeError(f"cudaProfilerStart failed with {start_result!r}")
    try:
        yield
    finally:
        _sync(device)
        stop_result = torch.cuda.cudart().cudaProfilerStop()
        if stop_result not in (0, None):
            raise RuntimeError(f"cudaProfilerStop failed with {stop_result!r}")


def run_liger_fsdp_step(
    *,
    mode: LigerMode,
    model_source: ModelSource,
    model_preset: ModelPreset,
    micro_batch_size: int,
    seq_len: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    gradient_checkpointing: bool = True,
    profile_steady_state: bool = False,
) -> dict[str, Any] | None:
    """Run one baseline or Liger-patched training-step benchmark.

    In distributed runs every rank executes the function, but only rank 0
    returns the result JSON. Non-zero ranks return ``None`` after participating
    in all collectives.
    """

    if mode not in {"baseline", "liger"}:
        raise ValueError("mode must be 'baseline' or 'liger'")
    if model_source not in {"reference", "transformers"}:
        raise ValueError("model_source must be 'reference' or 'transformers'")
    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"model_preset must be one of {sorted(MODEL_PRESETS)}")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if min(micro_batch_size, seq_len, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "micro_batch_size, seq_len, repeats, and iters must be positive; "
            "warmup must be non-negative"
        )
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay must be non-negative")

    spec = MODEL_PRESETS[model_preset]
    if seq_len > spec.block_size:
        raise ValueError(f"seq_len {seq_len} exceeds block_size {spec.block_size}")

    state = _distributed_state(device_name=device_name, allow_cpu=allow_cpu)
    torch.manual_seed(seed + state.rank)
    torch_dtype = TORCH_DTYPES[dtype]

    liger_version: str | None = None
    liger_patch = None
    if mode == "liger":
        if model_source != "transformers":
            raise ValueError("mode='liger' requires model_source='transformers'")
        liger_version, liger_patch = _apply_liger_llama_patch()

    try:
        if model_source == "transformers":
            model = _build_transformers_llama(
                spec,
                device=state.device,
                dtype=torch_dtype,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            model = _build_reference_model(spec, device=state.device, dtype=torch_dtype)

        model, distributed_strategy = _maybe_wrap_fsdp(model, state=state, dtype=torch_dtype)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        inputs = torch.randint(
            0,
            spec.vocab_size,
            (micro_batch_size, seq_len),
            device=state.device,
            dtype=torch.long,
        )
        targets = torch.randint(
            0,
            spec.vocab_size,
            (micro_batch_size, seq_len),
            device=state.device,
            dtype=torch.long,
        )

        last_loss: list[torch.Tensor] = []

        def step_once(*, phase: Literal["warmup", "measure"]) -> torch.Tensor:
            with _nvtx_range(f"swordfish.fsdp.{phase}.step", device=state.device):
                with _nvtx_range("swordfish.fsdp.zero_grad", device=state.device):
                    optimizer.zero_grad(set_to_none=True)
                with _nvtx_range("swordfish.fsdp.forward", device=state.device):
                    output = model(inputs)
                    logits = output.logits if hasattr(output, "logits") else output
                with _nvtx_range("swordfish.fsdp.loss", device=state.device):
                    loss = F.cross_entropy(
                        logits.float().view(-1, spec.vocab_size),
                        targets.view(-1),
                    )
                with _nvtx_range("swordfish.fsdp.backward", device=state.device):
                    loss.backward()
                with _nvtx_range("swordfish.fsdp.optimizer", device=state.device):
                    optimizer.step()
            detached = loss.detach()
            if last_loss:
                last_loss[0] = detached
            else:
                last_loss.append(detached)
            return detached

        if state.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(state.device)

        samples_ms: list[float] = []
        for repeat_idx in range(repeats):
            with _nvtx_range(f"swordfish.fsdp.repeat_{repeat_idx}.warmup", device=state.device):
                for _ in range(warmup):
                    step_once(phase="warmup")
            _sync(state.device)
            with _cuda_profiler_capture(device=state.device, enabled=profile_steady_state):
                with _nvtx_range(
                    f"swordfish.fsdp.repeat_{repeat_idx}.steady_state",
                    device=state.device,
                ):
                    start = time.perf_counter()
                    for _ in range(iters):
                        step_once(phase="measure")
                    _sync(state.device)
            sample = (time.perf_counter() - start) * 1000.0 / iters
            samples_ms.append(_max_across_ranks(sample, device=state.device))

        loss_value = float(last_loss[0].float().item())
        mean_loss = _mean_across_ranks(loss_value, device=state.device, world_size=state.world_size)
        finite_loss = bool(torch.isfinite(last_loss[0]).item())
        finite_loss_all = bool(
            _reduce_float(
                1.0 if finite_loss else 0.0,
                device=state.device,
                op=torch.distributed.ReduceOp.MIN,
            )
        )
        local_peak = _peak_memory_gb(state.device)
        peak_gpu_mem_gb = (
            _max_across_ranks(local_peak, device=state.device) if local_peak is not None else None
        )

        if state.rank != 0:
            return None

        stats = latency_stats(samples_ms)
        mean_ms = stats["mean_ms"]
        global_batch_size = micro_batch_size * state.world_size
        tokens_per_second = (
            global_batch_size * seq_len / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")
        )
        mode_metrics = {
            "latency": stats,
            "tokens_per_second": tokens_per_second,
            "optimizer_steps_per_second": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "final_loss": mean_loss,
            "finite_loss": finite_loss_all,
        }
        env = capture_env(state.device, arch_label=arch_label)
        env.update(
            {
                "rank": state.rank,
                "world_size": state.world_size,
                "local_rank": state.local_rank,
                "liger": liger_version or _liger_version(),
            }
        )

        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "benchmark": "liger_fsdp_train_step",
            "config": {
                "scope": "fsdp_train_step",
                "kernel": f"{model_preset}_fsdp1_step",
                "model_source": model_source,
                "model_preset": model_preset,
                "shape": {
                    "micro_batch_size": micro_batch_size,
                    "global_batch_size": global_batch_size,
                    "seq_len": seq_len,
                    "hidden_size": spec.hidden_size,
                    "intermediate_size": spec.intermediate_size,
                    "num_hidden_layers": spec.num_hidden_layers,
                    "num_attention_heads": spec.num_attention_heads,
                    "num_key_value_heads": spec.num_key_value_heads,
                    "vocab_size": spec.vocab_size,
                    "world_size": state.world_size,
                },
                "model": asdict(spec),
                "dtype": dtype,
                "repeats": repeats,
                "warmup": warmup,
                "iters": iters,
                "seed": seed,
                "optimizer": "AdamW",
                "lr": lr,
                "weight_decay": weight_decay,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_checkpointing_use_reentrant": (False if gradient_checkpointing else None),
                "profile": {
                    "nvtx_ranges": True,
                    "steady_state_cuda_profiler_api": profile_steady_state,
                    "steady_state_range": "swordfish.fsdp.repeat_<n>.steady_state",
                    "step_phases": [
                        "zero_grad",
                        "forward",
                        "loss",
                        "backward",
                        "optimizer",
                    ],
                },
                "distributed_strategy": distributed_strategy,
                "liger": {
                    "applied": mode == "liger",
                    "version": liger_version,
                    "kernel_module": liger_patch,
                    "mode": mode,
                },
            },
            "env": env,
            "correctness": {
                "finite_output": finite_loss_all,
                "finite_loss": finite_loss_all,
                "final_loss": mean_loss,
            },
            "metrics": {
                "latency": stats,
                "tokens_per_second": tokens_per_second,
                "optimizer_steps_per_second": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
                "peak_gpu_mem_gb": peak_gpu_mem_gb,
                "modes": {
                    mode: mode_metrics,
                },
            },
        }
    finally:
        if state.initialized_here and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
