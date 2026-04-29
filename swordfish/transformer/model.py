"""Small PyTorch GPT-style reference model.

This file is the semantic source of truth for future Triton/CuTe/PTX kernels.
The code deliberately spells out QKV projection, causal masking, softmax, and
the MLP instead of hiding the block behind a high-level library module.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from swordfish.transformer.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for a decoder-only transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Shape: [1, 1, T, T], so it broadcasts across batch and heads.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        if channels != self.config.n_embd:
            raise ValueError(f"expected hidden size {self.config.n_embd}, got {channels}")
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        # One GEMM produces Q, K, and V together: [B, T, 3*C].
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split hidden dim into heads: [B, T, C] -> [B, H, T, D].
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # QK^T creates one score for every query-token/key-token pair.
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Softmax is computed in fp32 for stability, then cast back to the model dtype.
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(dtype=q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # AV mixes values using the causal attention weights, then heads are merged.
        y = attn_weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.out_proj(y))


class GPTMLP(nn.Module):
    """The feed-forward half of the decoder block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, config.mlp_hidden_dim, bias=config.bias)
        self.fc_out = nn.Linear(config.mlp_hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x)


class GPTDecoderBlock(nn.Module):
    """One decoder-only transformer block: attention, MLP, and residual paths."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Tiny decoder-only language model wrapper around the reference block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTDecoderBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"expected token_ids shape [batch, seq], got {tuple(token_ids.shape)}")
        _, seq_len = token_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
