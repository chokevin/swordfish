"""PyTorch reference transformer used as the correctness oracle for kernels."""

from swordfish.transformer.config import GPTConfig, gpt1_config, tiny_test_config
from swordfish.transformer.model import (
    CausalSelfAttention,
    GPTDecoderBlock,
    GPTLanguageModel,
    GPTMLP,
)
from swordfish.transformer.bench import (
    run_transformer_forward_benchmark,
    run_transformer_train_step_benchmark,
)

__all__ = [
    "CausalSelfAttention",
    "GPTConfig",
    "GPTDecoderBlock",
    "GPTLanguageModel",
    "GPTMLP",
    "gpt1_config",
    "run_transformer_forward_benchmark",
    "run_transformer_train_step_benchmark",
    "tiny_test_config",
]
