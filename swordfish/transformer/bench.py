"""Benchmark harness for the PyTorch GPT reference modules."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

from swordfish.runner.backends import TORCH_DTYPES
from swordfish.runner.schema import SCHEMA_VERSION, latency_stats
from swordfish.runner.torch_gemm import (
    _resolve_device,
    _time_cpu,
    _time_cuda,
    capture_env,
    write_result,
)
from swordfish.transformer.config import GPTConfig, gpt1_config, tiny_test_config
from swordfish.transformer.model import GPTDecoderBlock, GPTLanguageModel

TransformerScope = Literal["block", "model"]
TransformerPreset = Literal["tiny", "gpt1"]


def _config_from_preset(
    preset: TransformerPreset,
    *,
    vocab_size: int | None,
    block_size: int | None,
    n_layer: int | None,
    n_head: int | None,
    n_embd: int | None,
) -> GPTConfig:
    factory = tiny_test_config if preset == "tiny" else gpt1_config
    overrides = {
        key: value
        for key, value in {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
        }.items()
        if value is not None
    }
    return factory(**overrides)


def run_transformer_forward_benchmark(
    *,
    scope: TransformerScope,
    preset: TransformerPreset,
    batch_size: int,
    seq_len: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    vocab_size: int | None = None,
    block_size: int | None = None,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
) -> dict[str, Any]:
    if scope not in {"block", "model"}:
        raise ValueError("scope must be 'block' or 'model'")
    if preset not in {"tiny", "gpt1"}:
        raise ValueError("preset must be 'tiny' or 'gpt1'")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if min(batch_size, seq_len, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "batch_size, seq_len, repeats, and iters must be positive; warmup must be non-negative"
        )

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    config = _config_from_preset(
        preset,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    if seq_len > config.block_size:
        raise ValueError(f"seq_len {seq_len} exceeds block_size {config.block_size}")

    torch.manual_seed(seed)
    torch_dtype = TORCH_DTYPES[dtype]
    if scope == "block":
        module = GPTDecoderBlock(config).to(device=device, dtype=torch_dtype)
        inputs = torch.randn((batch_size, seq_len, config.n_embd), device=device, dtype=torch_dtype)
    else:
        module = GPTLanguageModel(config).to(device=device, dtype=torch_dtype)
        inputs = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
    module.eval()

    last_output: list[torch.Tensor] = []

    def forward_once() -> torch.Tensor:
        with torch.no_grad():
            output = module(inputs)
        if last_output:
            last_output[0] = output
        else:
            last_output.append(output)
        return output

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    samples_ms = [timer(forward_once, warmup=warmup, iters=iters) for _ in range(repeats)]
    stats = latency_stats(samples_ms)
    output = last_output[0]

    finite = bool(torch.isfinite(output).all().item())
    checksum = float(output.float().sum().item())
    mean_ms = stats["mean_ms"]
    tokens_per_second = batch_size * seq_len / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": f"transformer_{scope}_forward",
        "config": {
            "scope": scope,
            "backend": "torch",
            "preset": preset,
            "shape": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
            },
            "model": asdict(config),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dtype": dtype,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        "env": capture_env(device, arch_label=arch_label),
        "correctness": {
            "finite_output": finite,
            "checksum_fp32_sum": checksum,
            "output_shape": list(output.shape),
        },
        "metrics": {
            "latency": stats,
            "tokens_per_second": tokens_per_second,
        },
    }


def run_transformer_train_step_benchmark(
    *,
    scope: TransformerScope,
    preset: TransformerPreset,
    batch_size: int,
    seq_len: int,
    dtype: str,
    repeats: int,
    warmup: int,
    iters: int,
    device_name: str = "auto",
    allow_cpu: bool = False,
    arch_label: str | None = None,
    seed: int = 0,
    vocab_size: int | None = None,
    block_size: int | None = None,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
) -> dict[str, Any]:
    if scope not in {"block", "model"}:
        raise ValueError("scope must be 'block' or 'model'")
    if preset not in {"tiny", "gpt1"}:
        raise ValueError("preset must be 'tiny' or 'gpt1'")
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}")
    if min(batch_size, seq_len, repeats, iters) <= 0 or warmup < 0:
        raise ValueError(
            "batch_size, seq_len, repeats, and iters must be positive; warmup must be non-negative"
        )
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay must be non-negative")

    device = _resolve_device(device_name, allow_cpu=allow_cpu)
    config = _config_from_preset(
        preset,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    if seq_len > config.block_size:
        raise ValueError(f"seq_len {seq_len} exceeds block_size {config.block_size}")

    torch.manual_seed(seed)
    torch_dtype = TORCH_DTYPES[dtype]
    if scope == "block":
        module = GPTDecoderBlock(config).to(device=device, dtype=torch_dtype)
        inputs = torch.randn((batch_size, seq_len, config.n_embd), device=device, dtype=torch_dtype)
        targets = torch.randn(
            (batch_size, seq_len, config.n_embd), device=device, dtype=torch_dtype
        )
    else:
        module = GPTLanguageModel(config).to(device=device, dtype=torch_dtype)
        inputs = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        targets = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)

    last_output: list[torch.Tensor] = []
    last_loss: list[torch.Tensor] = []

    def train_step_once() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        output = module(inputs)
        if scope == "block":
            loss = F.mse_loss(output.float(), targets.float())
        else:
            loss = F.cross_entropy(output.float().view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        if last_output:
            last_output[0] = output.detach()
            last_loss[0] = loss.detach()
        else:
            last_output.append(output.detach())
            last_loss.append(loss.detach())
        return loss

    timer = _time_cuda if device.type == "cuda" else _time_cpu
    samples_ms = [timer(train_step_once, warmup=warmup, iters=iters) for _ in range(repeats)]
    stats = latency_stats(samples_ms)
    output = last_output[0]
    loss = last_loss[0]

    finite_output = bool(torch.isfinite(output).all().item())
    finite_loss = bool(torch.isfinite(loss).item())
    checksum = float(output.float().sum().item())
    final_loss = float(loss.float().item())
    mean_ms = stats["mean_ms"]
    tokens_per_second = batch_size * seq_len / (mean_ms / 1000.0) if mean_ms > 0 else float("nan")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": f"transformer_{scope}_train_step",
        "config": {
            "scope": scope,
            "backend": "torch",
            "preset": preset,
            "shape": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
            },
            "model": asdict(config),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dtype": dtype,
            "repeats": repeats,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "env": capture_env(device, arch_label=arch_label),
        "correctness": {
            "finite_output": finite_output,
            "finite_loss": finite_loss,
            "checksum_fp32_sum": checksum,
            "final_loss": final_loss,
            "output_shape": list(output.shape),
        },
        "metrics": {
            "latency": stats,
            "tokens_per_second": tokens_per_second,
            "optimizer_steps_per_second": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        },
    }


def write_transformer_result(result: dict[str, Any], out_path: Path) -> None:
    write_result(result, out_path)
