from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submission import custom_kernel  # noqa: E402


def _generate_input(
    *,
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
    device: torch.device,
):
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)
    if distribution == "cauchy":
        x = (
            torch.distributions.Cauchy(0, 2)
            .sample((bs, seqlen, seqlen, dim))
            .to(
                device=device,
                dtype=torch.float32,
            )
        )
    else:
        x = torch.randn(
            (bs, seqlen, seqlen, dim), device=device, dtype=torch.float32, generator=gen
        )

    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device=device)
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device=device, generator=gen)

    weights = {
        "norm.weight": torch.randn(dim, device=device),
        "norm.bias": torch.randn(dim, device=device),
        "left_proj.weight": torch.randn(hiddendim, dim, device=device) / math.sqrt(hiddendim),
        "right_proj.weight": torch.randn(hiddendim, dim, device=device) / math.sqrt(hiddendim),
        "left_gate.weight": torch.randn(hiddendim, dim, device=device) / math.sqrt(hiddendim),
        "right_gate.weight": torch.randn(hiddendim, dim, device=device) / math.sqrt(hiddendim),
        "out_gate.weight": torch.randn(hiddendim, dim, device=device) / math.sqrt(hiddendim),
        "to_out_norm.weight": torch.randn(hiddendim, device=device),
        "to_out_norm.bias": torch.randn(hiddendim, device=device),
        "to_out.weight": torch.randn(dim, hiddendim, device=device) / math.sqrt(dim),
    }
    return x, mask, weights, {"dim": dim, "hidden_dim": hiddendim}


def _reference(data):
    x, mask, weights, config = data
    dim = config["dim"]
    hidden = config["hidden_dim"]

    x = F.layer_norm(x, (dim,), weights["norm.weight"], weights["norm.bias"])
    left = F.linear(x, weights["left_proj.weight"])
    right = F.linear(x, weights["right_proj.weight"])
    mask_view = mask.unsqueeze(-1)
    left = left * mask_view
    right = right * mask_view
    left = left * torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right = right * torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))
    out = torch.einsum("bikd,bjkd->bijd", left, right)
    out = F.layer_norm(out, (hidden,), weights["to_out_norm.weight"], weights["to_out_norm.bias"])
    return F.linear(out * out_gate, weights["to_out.weight"])


@pytest.mark.parametrize("nomask", [True, False])
@pytest.mark.parametrize("distribution", ["normal", "cauchy"])
def test_trimul_custom_kernel_matches_reference_cpu(nomask: bool, distribution: str):
    data = _generate_input(
        seqlen=8,
        bs=1,
        dim=16,
        hiddendim=8,
        seed=123,
        nomask=nomask,
        distribution=distribution,
        device=torch.device("cpu"),
    )

    actual = custom_kernel(data)
    expected = _reference(data)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_trimul_custom_kernel_matches_reference_cuda_smoke():
    data = _generate_input(
        seqlen=32,
        bs=1,
        dim=128,
        hiddendim=128,
        seed=9371,
        nomask=True,
        distribution="normal",
        device=torch.device("cuda"),
    )

    actual = custom_kernel(data)
    expected = _reference(data)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
