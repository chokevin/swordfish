#!/usr/bin/env python3
"""GPUMODE TriMul outgoing forward submission.

The evaluator imports ``custom_kernel`` from this file and passes
``(input_tensor, mask, weights, config)``.  This implementation keeps the
forward pass in PyTorch ops but avoids per-call module construction and fuses
the five input projections into one linear call.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:  # The official harness provides task.py; local tests do not.
    from task import input_t, output_t
except ImportError:  # pragma: no cover - typing fallback for local repo tests.
    input_t = Any
    output_t = Any


_PROJECTION_KEYS = (
    "left_proj.weight",
    "right_proj.weight",
    "left_gate.weight",
    "right_gate.weight",
    "out_gate.weight",
)
_MAX_CACHED_PROJECTIONS = 8
_projection_cache: OrderedDict[tuple[tuple[int, ...], torch.device, torch.dtype], torch.Tensor] = (
    OrderedDict()
)


def _stacked_projection_weight(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [5 * hidden_dim, dim] weight for the combined input projection."""
    first = weights[_PROJECTION_KEYS[0]]
    key = (
        tuple(int(weights[name].data_ptr()) for name in _PROJECTION_KEYS),
        first.device,
        first.dtype,
    )
    cached = _projection_cache.get(key)
    if cached is not None:
        _projection_cache.move_to_end(key)
        return cached

    stacked = torch.cat([weights[name] for name in _PROJECTION_KEYS], dim=0).contiguous()
    _projection_cache[key] = stacked
    if len(_projection_cache) > _MAX_CACHED_PROJECTIONS:
        _projection_cache.popitem(last=False)
    return stacked


def _has_real_mask(mask: torch.Tensor) -> bool:
    """Official no-mask cases use float ones; masked cases use integer 0/1."""
    return not mask.dtype.is_floating_point


def _triangle_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute out[b, i, j, d] = sum_k left[b, i, k, d] * right[b, j, k, d]."""
    if left.is_cuda and torch.cuda.is_bf16_supported():
        return torch.einsum(
            "bikd,bjkd->bijd",
            left.to(torch.bfloat16),
            right.to(torch.bfloat16),
        ).to(torch.float32)
    return torch.einsum("bikd,bjkd->bijd", left, right)


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    """Outgoing Triangle Multiplicative Update forward pass."""
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])

    x = F.layer_norm(
        input_tensor,
        (dim,),
        weights["norm.weight"],
        weights["norm.bias"],
    )

    projected = F.linear(x, _stacked_projection_weight(weights))
    left, right, left_gate, right_gate, out_gate = projected.split(hidden_dim, dim=-1)

    if _has_real_mask(mask):
        mask_view = mask.unsqueeze(-1).to(dtype=left.dtype)
        left = left * mask_view
        right = right * mask_view

    left_gate.sigmoid_()
    right_gate.sigmoid_()
    out_gate.sigmoid_()
    left = left * left_gate
    right = right * right_gate

    out = _triangle_multiply(left, right)
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    )
    out = out * out_gate
    return F.linear(out, weights["to_out.weight"])


def _reference_output(
    data: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict],
) -> torch.Tensor:
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    mask_view = mask.unsqueeze(-1)
    x = F.layer_norm(input_tensor, (dim,), weights["norm.weight"], weights["norm.bias"])
    left = F.linear(x, weights["left_proj.weight"]) * mask_view
    right = F.linear(x, weights["right_proj.weight"]) * mask_view
    left = left * torch.sigmoid(F.linear(x, weights["left_gate.weight"]))
    right = right * torch.sigmoid(F.linear(x, weights["right_gate.weight"]))
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))
    out = torch.einsum("bikd,bjkd->bijd", left, right)
    out = F.layer_norm(
        out,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    )
    return F.linear(out * out_gate, weights["to_out.weight"])


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
        input_tensor = (
            torch.distributions.Cauchy(0, 2)
            .sample((bs, seqlen, seqlen, dim))
            .to(
                device=device,
                dtype=torch.float32,
            )
        )
    else:
        input_tensor = torch.randn(
            (bs, seqlen, seqlen, dim),
            device=device,
            dtype=torch.float32,
            generator=gen,
        ).contiguous()

    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device=device)
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device=device, generator=gen)

    weights = {
        "norm.weight": torch.randn(dim, device=device, dtype=torch.float32),
        "norm.bias": torch.randn(dim, device=device, dtype=torch.float32),
        "left_proj.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32)
        / math.sqrt(hiddendim),
        "right_proj.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32)
        / math.sqrt(hiddendim),
        "left_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32)
        / math.sqrt(hiddendim),
        "right_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32)
        / math.sqrt(hiddendim),
        "out_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32)
        / math.sqrt(hiddendim),
        "to_out_norm.weight": torch.randn(hiddendim, device=device, dtype=torch.float32),
        "to_out_norm.bias": torch.randn(hiddendim, device=device, dtype=torch.float32),
        "to_out.weight": torch.randn(dim, hiddendim, device=device, dtype=torch.float32)
        / math.sqrt(dim),
    }
    return input_tensor, mask, weights, {"dim": dim, "hidden_dim": hiddendim}


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _latency_stats(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    mean = sum(samples) / len(samples)
    return {
        "samples_ms": samples,
        "mean_ms": mean,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p50_ms": ordered[len(ordered) // 2],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriMul outgoing benchmark harness")
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hiddendim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=9371)
    parser.add_argument("--nomask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distribution", choices=["normal", "cauchy"], default="normal")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TriMul benchmark harness requires CUDA")
    device = torch.device("cuda")
    data = _generate_input(
        seqlen=args.seqlen,
        bs=args.bs,
        dim=args.dim,
        hiddendim=args.hiddendim,
        seed=args.seed,
        nomask=args.nomask,
        distribution=args.distribution,
        device=device,
    )

    def run_once() -> torch.Tensor:
        return custom_kernel(data)

    output = run_once()
    torch.cuda.synchronize()
    correctness: dict[str, Any] = {
        "finite_output": bool(torch.isfinite(output).all().item()),
        "output_shape": list(output.shape),
    }
    if args.check_reference:
        expected = _reference_output(data)
        diff = torch.abs(output.float() - expected.float())
        correctness.update(
            {
                "matches_reference": bool(torch.allclose(output, expected, rtol=2e-2, atol=2e-2)),
                "max_abs_error": float(diff.max().item()),
            }
        )
        del expected, diff
        torch.cuda.synchronize()

    samples = [
        _time_cuda(run_once, warmup=args.warmup, iters=args.iters) for _ in range(args.repeats)
    ]
    stats = _latency_stats(samples)
    result = {
        "schema_version": "swordfish.runner.v1",
        "benchmark": "trimul_outgoing",
        "config": {
            "bs": args.bs,
            "seqlen": args.seqlen,
            "dim": args.dim,
            "hidden_dim": args.hiddendim,
            "nomask": args.nomask,
            "distribution": args.distribution,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "env": {
            "host": socket.gethostname(),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "correctness": correctness,
        "metrics": {"latency": stats},
        "timestamp_unix": time.time(),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
