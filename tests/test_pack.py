"""Pack/unpack round-trip correctness."""

import pytest
import torch

from swordfish.pack import quantize_symmetric_int4, random_quantized_weights
from swordfish.reference import dequantize_int4


def _dequant_error(w, group_size):
    packed, scales = quantize_symmetric_int4(w, group_size=group_size)
    w_hat = dequantize_int4(packed, scales, group_size=group_size)
    abs_err = (w - w_hat).abs()
    rel_err = abs_err / (w.abs().clamp(min=1e-4))
    return abs_err.max().item(), rel_err.median().item()


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("K,N", [(128, 128), (4096, 4096), (4096, 8192)])
def test_pack_unpack_roundtrip(K, N, group_size):
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(K, N, device="cuda", dtype=torch.float16, generator=gen) * 0.02

    max_abs, median_rel = _dequant_error(w, group_size)

    # INT4 symmetric has max quant error ~ scale/2 = absmax/14; with normally
    # distributed weights the max abs error is bounded roughly by that.
    assert max_abs < 1e-2, f"max_abs_err={max_abs} too large"
    assert median_rel < 0.3, f"median_rel_err={median_rel} too large"


def test_random_quantized_weights_shapes():
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    packed, scales = random_quantized_weights(K=512, N=256, group_size=128)
    assert packed.shape == (512, 128)
    assert packed.dtype == torch.uint8
    assert scales.shape == (4, 256)  # 512/128 = 4 groups
    assert scales.dtype == torch.float16
