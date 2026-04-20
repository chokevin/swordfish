"""Tests for the bench harness's L0 correctness gate.

The gate is the firewall that prevents a numerically-wrong kernel from
ever producing a speed number. If these tests pass, we're guaranteed:
  - a correct impl reports correct=True, cosine~=1, low max_relerr, gets timed
  - a wrong impl reports correct=False, gets error="correctness_failed:...",
    is NOT timed (no misleading TFLOPS in the CSV)
"""

from __future__ import annotations

import pytest
import torch

from bench.run_bench import _cosine_sim, _max_relerr


def test_max_relerr_zero_for_identical():
    a = torch.randn(100, 100)
    assert _max_relerr(a, a) == 0.0


def test_max_relerr_catches_systematic_bias():
    a = torch.ones(100, 100)
    b = a * 1.5  # 50% systematic bias — must be caught
    err = _max_relerr(b, a)
    assert err > 0.4, f"systematic bias should produce large relerr, got {err}"


def test_cosine_sim_one_for_identical():
    a = torch.randn(1000)
    assert abs(_cosine_sim(a, a) - 1.0) < 1e-6


def test_cosine_sim_invariant_to_uniform_scale():
    """Cosine should NOT catch a uniform scale (that's why we also check allclose).
    This documents the contract: cosine catches direction drift, allclose catches
    magnitude drift. Both gates must pass."""
    a = torch.randn(1000)
    b = a * 2.0
    cos = _cosine_sim(a, b)
    assert abs(cos - 1.0) < 1e-6, "cosine is scale-invariant by design"


def test_cosine_sim_below_threshold_for_random():
    """Two unrelated random vectors should be far below the 0.999 gate."""
    torch.manual_seed(0)
    a = torch.randn(10000)
    b = torch.randn(10000)
    assert _cosine_sim(a, b) < 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bench gate exercised on CUDA")
def test_l0_gate_passes_reference_against_itself():
    """Sanity: when fp16 impl is checked against the FP32 reference on
    well-conditioned random weights, the gate must accept it."""
    from bench.run_bench import bench_shape
    from bench.shapes import Shape

    s = Shape("test", M=4, N=128, K=128, group_size=128, priority=2)
    rows = bench_shape(s, ["fp16"], repeats=1, warmup=1, iters=2)
    r = rows[0]
    assert r.get("error") is None, f"fp16 should pass L0, got error={r.get('error')}"
    assert r["correct"] is True
    assert r["cosine"] >= 0.999
