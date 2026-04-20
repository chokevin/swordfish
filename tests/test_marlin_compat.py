"""CPU-runnable contract tests for marlin_compat.

The full Marlin-output cross-check is GPU-only and lives in the bench
harness on the A100 box (it is gated by ``import marlin`` at runtime).
This file just guards the boundary so swordfish.pack changes can't
silently break the dequant path that feeds Marlin.
"""

from __future__ import annotations

import pytest
import torch

from swordfish.marlin_compat import dequantized_for_marlin, to_marlin_layout
from swordfish.pack import quantize_symmetric_int4


def _make_w(K: int, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(K, N, dtype=torch.float16, generator=g) * 0.02


def test_dequantized_for_marlin_shape_and_dtype():
    K, N, group_size = 256, 128, 128
    w = _make_w(K, N)
    packed, scales = quantize_symmetric_int4(w, group_size=group_size)

    out = dequantized_for_marlin(packed, scales, group_size=group_size)

    # Marlin expects nn.Linear weight layout: [out_features=N, in_features=K]
    assert out.shape == (N, K), f"expected [{N},{K}], got {tuple(out.shape)}"
    assert out.dtype == torch.float16
    assert out.is_contiguous()


def test_dequantized_for_marlin_roundtrip_close_to_original():
    """Dequant should be close to the original fp16 weights (INT4 quant error
    bounded by scale/2 = absmax/14 per group)."""
    K, N, group_size = 512, 256, 128
    w = _make_w(K, N, seed=7)
    packed, scales = quantize_symmetric_int4(w, group_size=group_size)

    w_hat_NK = dequantized_for_marlin(packed, scales, group_size=group_size)
    w_hat_KN = w_hat_NK.t().contiguous()

    # absmax / 14 is the symmetric INT4 worst-case error per element
    per_group_absmax = w.reshape(K // group_size, group_size, N).abs().amax(dim=1)
    bound = (per_group_absmax / 14.0).repeat_interleave(group_size, dim=0).max().item()

    assert (w - w_hat_KN).abs().max().item() <= bound + 1e-3


def test_to_marlin_layout_raises_without_cuda():
    """On the Mac dev box this is the contract: clean error, not a segfault."""
    K, N, group_size = 128, 64, 128
    w = _make_w(K, N)
    packed, scales = quantize_symmetric_int4(w, group_size=group_size)

    if torch.cuda.is_available():
        pytest.skip("CUDA available — non-CUDA-error path not exercised here")

    with pytest.raises(RuntimeError, match="CUDA"):
        to_marlin_layout(packed, scales, group_size=group_size)


def test_workspace_cache_reuses_storage_per_N():
    """Cache contract: identical (N, device) -> same tensor object reused.
    Different N -> different tensor. This is the W2-prep guarantee that
    captured-marlin callers can rely on a stable device offset for the
    workspace tensor (rvLLM §5.2 trap: alloc inside capture binds to a
    stale offset on replay)."""
    from swordfish.marlin_compat import _get_workspace, clear_workspace_cache

    clear_workspace_cache()
    device = torch.device("cpu")
    w_a = _get_workspace(N=4096, device=device)
    w_b = _get_workspace(N=4096, device=device)
    assert w_a.data_ptr() == w_b.data_ptr(), "same (N, device) must reuse storage"
    assert w_a.shape == (4096 // 128 * 16,) and w_a.dtype == torch.int32

    w_c = _get_workspace(N=8192, device=device)
    assert w_c.data_ptr() != w_a.data_ptr(), "different N must allocate fresh"
    assert w_c.shape == (8192 // 128 * 16,)


def test_workspace_cache_zeros_on_reuse():
    """Marlin uses the workspace as inter-CTA tile counter / sync barrier.
    Stale values from a previous call must NOT leak into the next call."""
    from swordfish.marlin_compat import _get_workspace, clear_workspace_cache

    clear_workspace_cache()
    device = torch.device("cpu")
    w = _get_workspace(N=2048, device=device)
    w.fill_(42)  # simulate marlin writing tile counters
    w2 = _get_workspace(N=2048, device=device)
    assert w2.data_ptr() == w.data_ptr(), "cache hit"
    assert int(w2.sum().item()) == 0, "must be re-zeroed on reuse"


def test_marlin_matmul_signature_accepts_out_kwarg():
    """W2 captured-marlin callers will pre-allocate `out` once (rvLLM §5.2:
    no allocation inside capture). The signature must accept `out=` and
    pass it through to marlin.mul. We can't actually invoke marlin on Mac,
    but we CAN check the signature contract via inspect — a regression
    here breaks the W2 captured path."""
    import inspect

    from swordfish.marlin_compat import marlin_matmul

    sig = inspect.signature(marlin_matmul)
    assert "out" in sig.parameters, "marlin_matmul must accept `out=` for capture"
    assert sig.parameters["out"].default is None
    assert sig.parameters["out"].kind == inspect.Parameter.KEYWORD_ONLY \
        or sig.parameters["out"].default is None  # either keyword-only or default-None positional
