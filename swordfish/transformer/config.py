"""Shape/config definitions for the reference GPT-style transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Minimal decoder-only transformer config.

    The defaults intentionally mirror the important GPT-1 scale knobs: 12 layers,
    12 attention heads, 768 hidden size, learned positional embeddings, and a
    4x hidden MLP. This is "GPT-1-ish" for kernel learning, not a historical
    reproduction of GPT-1 training or tokenization.
    """

    vocab_size: int = 40_478
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("vocab_size, block_size, n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must divide evenly across n_head")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden_dim(self) -> int:
        return self.mlp_ratio * self.n_embd


def gpt1_config(**overrides: int | float | bool) -> GPTConfig:
    """Return the default GPT-1-ish config with optional field overrides."""
    return GPTConfig(**overrides)


def tiny_test_config(**overrides: int | float | bool) -> GPTConfig:
    """Return a fast CPU config that keeps the same 12-head attention structure."""
    values: dict[str, int | float | bool] = {
        "vocab_size": 128,
        "block_size": 16,
        "n_layer": 2,
        "n_head": 12,
        "n_embd": 48,
        "dropout": 0.0,
    }
    values.update(overrides)
    return GPTConfig(**values)
