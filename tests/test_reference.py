"""Reference implementation correctness against a pure-FP16 matmul.

When we implement the Triton kernel, this same structure will validate it
against the reference.
"""

import pytest
import torch

from bench.shapes import SHAPE_SETS
from swordfish.pack import random_quantized_weights
from swordfish.reference import dequantize_int4, reference_w4a16_matmul


@pytest.mark.parametrize("shape", SHAPE_SETS["voice"], ids=lambda s: s.name)
def test_reference_matches_dequant_then_matmul(shape):
    if not torch.cuda.is_available():
        pytest.skip("no cuda")

    M, N, K, g = shape.M, shape.N, shape.K, shape.group_size
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    packed, scales = random_quantized_weights(K, N, group_size=g, seed=1)

    # Two paths that should agree:
    # Path A: the helper (dequant -> fp16 matmul)
    out_a = reference_w4a16_matmul(a, packed, scales, group_size=g)

    # Path B: manual dequant + matmul
    w_fp = dequantize_int4(packed, scales, group_size=g)
    out_b = (a.float() @ w_fp.float()).half()

    # bit-for-bit should not be expected due to accumulation order, but very close
    torch.testing.assert_close(out_a, out_b, rtol=1e-2, atol=1e-2)
