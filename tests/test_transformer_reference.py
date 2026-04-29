from __future__ import annotations

import pytest
import torch

from swordfish.runner.schema import validate_result_protocol
from swordfish.transformer import (
    GPTDecoderBlock,
    GPTLanguageModel,
    gpt1_config,
    run_transformer_forward_benchmark,
    run_transformer_train_step_benchmark,
    tiny_test_config,
)


def test_gpt1_config_uses_twelve_attention_heads():
    config = gpt1_config()

    assert config.n_head == 12
    assert config.n_embd == 768
    assert config.head_dim == 64
    assert config.mlp_hidden_dim == 4 * 768


def test_config_rejects_hidden_size_that_does_not_split_across_heads():
    with pytest.raises(ValueError, match="n_embd must divide"):
        gpt1_config(n_embd=770)


def test_decoder_block_forward_supports_gpt1_style_shape_on_cpu():
    config = gpt1_config(n_layer=1, block_size=8, dropout=0.0)
    block = GPTDecoderBlock(config)
    x = torch.randn(1, 4, config.n_embd)

    y = block(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_decoder_block_preserves_float64_dtype():
    config = tiny_test_config(n_layer=1)
    block = GPTDecoderBlock(config).to(dtype=torch.float64)
    x = torch.randn(2, 5, config.n_embd, dtype=torch.float64)

    y = block(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float64


def test_language_model_forward_shape_and_dtype():
    config = tiny_test_config(vocab_size=97, block_size=8, n_layer=2)
    model = GPTLanguageModel(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 6))

    logits = model(token_ids)

    assert logits.shape == (2, 6, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_language_model_rejects_too_long_sequence():
    config = tiny_test_config(block_size=4)
    model = GPTLanguageModel(config)
    token_ids = torch.randint(0, config.vocab_size, (1, 5))

    with pytest.raises(ValueError, match="exceeds block_size"):
        model(token_ids)


def test_transformer_block_benchmark_cpu_smoke(tmp_path):
    out = tmp_path / "block-bench.json"

    result = run_transformer_forward_benchmark(
        scope="block",
        preset="tiny",
        batch_size=2,
        seq_len=4,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="a100",
        seed=0,
    )
    out.write_text(str(result))

    assert result["benchmark"] == "transformer_block_forward"
    assert validate_result_protocol(result) == []
    assert result["config"]["backend"] == "torch"
    assert result["config"]["shape"]["seq_len"] == 4
    assert result["config"]["model"]["n_head"] == 12
    assert result["env"]["gpu_class"] == "a100"
    assert result["correctness"]["finite_output"] is True
    assert result["correctness"]["output_shape"] == [2, 4, 48]
    assert result["metrics"]["latency"]["mean_ms"] > 0
    assert result["metrics"]["tokens_per_second"] > 0


def test_transformer_model_benchmark_cpu_smoke():
    result = run_transformer_forward_benchmark(
        scope="model",
        preset="tiny",
        batch_size=1,
        seq_len=3,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
        seed=0,
        vocab_size=97,
        block_size=8,
    )

    assert result["benchmark"] == "transformer_model_forward"
    assert validate_result_protocol(result) == []
    assert result["correctness"]["output_shape"] == [1, 3, 97]
    assert result["correctness"]["finite_output"] is True


def test_transformer_train_step_benchmark_cpu_smoke():
    result = run_transformer_train_step_benchmark(
        scope="model",
        preset="tiny",
        batch_size=1,
        seq_len=3,
        dtype="fp32",
        repeats=1,
        warmup=0,
        iters=1,
        device_name="cpu",
        allow_cpu=True,
        arch_label="h100",
        seed=0,
        vocab_size=97,
        block_size=8,
        lr=1e-3,
        weight_decay=0.0,
    )

    assert result["benchmark"] == "transformer_model_train_step"
    assert validate_result_protocol(result) == []
    assert result["config"]["optimizer"] == "AdamW"
    assert result["correctness"]["output_shape"] == [1, 3, 97]
    assert result["correctness"]["finite_output"] is True
    assert result["correctness"]["finite_loss"] is True
    assert result["correctness"]["final_loss"] > 0
    assert result["metrics"]["latency"]["mean_ms"] > 0
    assert result["metrics"]["optimizer_steps_per_second"] > 0
