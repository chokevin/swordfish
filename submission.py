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
import os
import socket
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional outside CUDA images.
    triton = None
    tl = None

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
_DEFAULT_TRIANGLE_BACKEND = "auto"
_DEFAULT_TRITON_BLOCK_M = 32
_DEFAULT_TRITON_BLOCK_N = 32
_DEFAULT_TRITON_BLOCK_K = 64
_DEFAULT_TRITON_NUM_WARPS = 4
_DEFAULT_TRITON_NUM_STAGES = 3


if triton is not None and tl is not None:

    @triton.jit
    def _triangle_multiply_kernel(
        left,
        right,
        out,
        n: tl.constexpr,
        hidden_dim: tl.constexpr,
        left_stride_b: tl.constexpr,
        left_stride_i: tl.constexpr,
        left_stride_k: tl.constexpr,
        left_stride_d: tl.constexpr,
        right_stride_b: tl.constexpr,
        right_stride_j: tl.constexpr,
        right_stride_k: tl.constexpr,
        right_stride_d: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_i: tl.constexpr,
        out_stride_j: tl.constexpr,
        out_stride_d: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_ij = tl.program_id(0)
        pid_bd = tl.program_id(1)
        blocks_j = tl.cdiv(n, block_n)
        block_i = pid_ij // blocks_j
        block_j = pid_ij - block_i * blocks_j
        batch = pid_bd // hidden_dim
        hidden = pid_bd - batch * hidden_dim

        offs_i = block_i * block_m + tl.arange(0, block_m)
        offs_j = block_j * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k_start in range(0, n, block_k):
            k = k_start + offs_k
            left_offsets = (
                batch * left_stride_b
                + offs_i[:, None] * left_stride_i
                + k[None, :] * left_stride_k
                + hidden * left_stride_d
            )
            right_offsets = (
                batch * right_stride_b
                + k[:, None] * right_stride_k
                + offs_j[None, :] * right_stride_j
                + hidden * right_stride_d
            )
            left_tile = tl.load(
                left + left_offsets,
                mask=(offs_i[:, None] < n) & (k[None, :] < n),
                other=0.0,
            )
            right_tile = tl.load(
                right + right_offsets,
                mask=(k[:, None] < n) & (offs_j[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(
                left_tile.to(tl.bfloat16),
                right_tile.to(tl.bfloat16),
                out_dtype=tl.float32,
            )

        out_offsets = (
            batch * out_stride_b
            + offs_i[:, None] * out_stride_i
            + offs_j[None, :] * out_stride_j
            + hidden * out_stride_d
        )
        tl.store(
            out + out_offsets,
            acc,
            mask=(offs_i[:, None] < n) & (offs_j[None, :] < n),
        )

    @triton.jit
    def _triangle_multiply_packed_kernel(
        left,
        right,
        out,
        n: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid_ij = tl.program_id(0)
        pid_bh = tl.program_id(1)
        blocks_j = tl.cdiv(n, block_n)
        block_i = pid_ij // blocks_j
        block_j = pid_ij - block_i * blocks_j

        offs_i = block_i * block_m + tl.arange(0, block_m)
        offs_j = block_j * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        base = pid_bh * n * n
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)

        for k_start in range(0, n, block_k):
            k = k_start + offs_k
            left_offsets = base + offs_i[:, None] * n + k[None, :]
            right_offsets = base + k[:, None] * n + offs_j[None, :]
            left_tile = tl.load(
                left + left_offsets,
                mask=(offs_i[:, None] < n) & (k[None, :] < n),
                other=0.0,
            )
            right_tile = tl.load(
                right + right_offsets,
                mask=(k[:, None] < n) & (offs_j[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(left_tile, right_tile, out_dtype=tl.float32)

        out_offsets = base + offs_i[:, None] * n + offs_j[None, :]
        tl.store(
            out + out_offsets,
            acc,
            mask=(offs_i[:, None] < n) & (offs_j[None, :] < n),
        )

else:
    _triangle_multiply_kernel = None
    _triangle_multiply_packed_kernel = None


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


def _triangle_multiply_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute out[b, i, j, d] = sum_k left[b, i, k, d] * right[b, j, k, d]."""
    if left.is_cuda and torch.cuda.is_bf16_supported():
        return torch.einsum(
            "bikd,bjkd->bijd",
            left.to(torch.bfloat16),
            right.to(torch.bfloat16),
        ).to(torch.float32)
    return torch.einsum("bikd,bjkd->bijd", left, right)


def _triangle_multiply_triton(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Triangle contraction in Triton, one tiled matmul per batch/hidden channel."""
    if triton is None or _triangle_multiply_kernel is None:
        raise RuntimeError("Triton triangle backend requested, but Triton is not available")
    if not left.is_cuda:
        raise RuntimeError("Triton triangle backend requires CUDA tensors")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Triton triangle backend requires BF16-capable CUDA hardware")

    batch, n, right_n, hidden_dim = left.shape
    if right.shape != (batch, n, right_n, hidden_dim) or right_n != n:
        raise ValueError(
            f"expected matching [B, N, N, H] tensors, got {left.shape} and {right.shape}"
        )

    out = torch.empty((batch, n, n, hidden_dim), device=left.device, dtype=torch.float32)
    grid = (
        triton.cdiv(n, block_m) * triton.cdiv(n, block_n),
        batch * hidden_dim,
    )
    _triangle_multiply_kernel[grid](
        left,
        right,
        out,
        n,
        hidden_dim,
        left.stride(0),
        left.stride(1),
        left.stride(2),
        left.stride(3),
        right.stride(0),
        right.stride(1),
        right.stride(2),
        right.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _triangle_multiply_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Triangle contraction as explicit batched GEMM over [B * H, N, N]."""
    batch, n, right_n, hidden_dim = left.shape
    if right.shape != (batch, n, right_n, hidden_dim) or right_n != n:
        raise ValueError(
            f"expected matching [B, N, N, H] tensors, got {left.shape} and {right.shape}"
        )

    matmul_dtype = torch.bfloat16 if left.is_cuda and torch.cuda.is_bf16_supported() else left.dtype
    left_bh = (
        left.permute(0, 3, 1, 2).contiguous().to(matmul_dtype).reshape(batch * hidden_dim, n, n)
    )
    right_bh = (
        right.permute(0, 3, 2, 1).contiguous().to(matmul_dtype).reshape(batch * hidden_dim, n, n)
    )
    out = torch.bmm(left_bh, right_bh).to(torch.float32)
    return out.reshape(batch, hidden_dim, n, n).permute(0, 2, 3, 1).contiguous()


def _triangle_multiply_triton_packed(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Triton contraction over packed [B * H, N, N] BF16 matrices."""
    if triton is None or _triangle_multiply_packed_kernel is None:
        raise RuntimeError("Triton packed backend requested, but Triton is not available")
    if not left.is_cuda:
        raise RuntimeError("Triton packed backend requires CUDA tensors")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Triton packed backend requires BF16-capable CUDA hardware")

    batch, n, right_n, hidden_dim = left.shape
    if right.shape != (batch, n, right_n, hidden_dim) or right_n != n:
        raise ValueError(
            f"expected matching [B, N, N, H] tensors, got {left.shape} and {right.shape}"
        )

    left_packed = (
        left.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16).reshape(batch * hidden_dim, n, n)
    )
    right_packed = (
        right.permute(0, 3, 2, 1).contiguous().to(torch.bfloat16).reshape(batch * hidden_dim, n, n)
    )
    out_packed = torch.empty_like(left_packed, dtype=torch.float32)
    grid = (
        triton.cdiv(n, block_m) * triton.cdiv(n, block_n),
        batch * hidden_dim,
    )
    _triangle_multiply_packed_kernel[grid](
        left_packed,
        right_packed,
        out_packed,
        n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_packed.reshape(batch, hidden_dim, n, n).permute(0, 2, 3, 1).contiguous()


def _triangle_multiply(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    backend: str,
    triton_block_m: int,
    triton_block_n: int,
    triton_block_k: int,
    triton_num_warps: int,
    triton_num_stages: int,
) -> torch.Tensor:
    if backend == "auto":
        backend = (
            "triton_packed"
            if (
                left.is_cuda
                and triton is not None
                and _triangle_multiply_packed_kernel is not None
                and torch.cuda.is_bf16_supported()
                and left.shape[1] <= 256
            )
            else "torch"
        )
    if backend == "triton":
        return _triangle_multiply_triton(
            left,
            right,
            block_m=triton_block_m,
            block_n=triton_block_n,
            block_k=triton_block_k,
            num_warps=triton_num_warps,
            num_stages=triton_num_stages,
        )
    if backend == "bmm":
        return _triangle_multiply_bmm(left, right)
    if backend == "triton_packed":
        return _triangle_multiply_triton_packed(
            left,
            right,
            block_m=triton_block_m,
            block_n=triton_block_n,
            block_k=triton_block_k,
            num_warps=triton_num_warps,
            num_stages=triton_num_stages,
        )
    if backend == "torch":
        return _triangle_multiply_torch(left, right)
    raise ValueError(f"unknown triangle backend: {backend}")


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    """Outgoing Triangle Multiplicative Update forward pass."""
    input_tensor, mask, weights, config = data
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    triangle_backend = str(
        config.get(
            "triangle_backend", os.environ.get("TRIMUL_TRIANGLE_BACKEND", _DEFAULT_TRIANGLE_BACKEND)
        )
    ).lower()
    triton_block_m = int(
        config.get(
            "triton_block_m", os.environ.get("TRIMUL_TRITON_BLOCK_M", _DEFAULT_TRITON_BLOCK_M)
        )
    )
    triton_block_n = int(
        config.get(
            "triton_block_n", os.environ.get("TRIMUL_TRITON_BLOCK_N", _DEFAULT_TRITON_BLOCK_N)
        )
    )
    triton_block_k = int(
        config.get(
            "triton_block_k", os.environ.get("TRIMUL_TRITON_BLOCK_K", _DEFAULT_TRITON_BLOCK_K)
        )
    )
    triton_num_warps = int(
        config.get(
            "triton_num_warps",
            os.environ.get("TRIMUL_TRITON_NUM_WARPS", _DEFAULT_TRITON_NUM_WARPS),
        )
    )
    triton_num_stages = int(
        config.get(
            "triton_num_stages",
            os.environ.get("TRIMUL_TRITON_NUM_STAGES", _DEFAULT_TRITON_NUM_STAGES),
        )
    )

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

    out = _triangle_multiply(
        left,
        right,
        backend=triangle_backend,
        triton_block_m=triton_block_m,
        triton_block_n=triton_block_n,
        triton_block_k=triton_block_k,
        triton_num_warps=triton_num_warps,
        triton_num_stages=triton_num_stages,
    )
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
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "bmm", "triton", "triton_packed"],
        default=os.environ.get("TRIMUL_TRIANGLE_BACKEND", _DEFAULT_TRIANGLE_BACKEND),
        help="triangle contraction backend",
    )
    parser.add_argument("--triton-block-m", type=int, default=_DEFAULT_TRITON_BLOCK_M)
    parser.add_argument("--triton-block-n", type=int, default=_DEFAULT_TRITON_BLOCK_N)
    parser.add_argument("--triton-block-k", type=int, default=_DEFAULT_TRITON_BLOCK_K)
    parser.add_argument("--triton-num-warps", type=int, default=_DEFAULT_TRITON_NUM_WARPS)
    parser.add_argument("--triton-num-stages", type=int, default=_DEFAULT_TRITON_NUM_STAGES)
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
    data[3]["triangle_backend"] = args.backend
    data[3]["triton_block_m"] = args.triton_block_m
    data[3]["triton_block_n"] = args.triton_block_n
    data[3]["triton_block_k"] = args.triton_block_k
    data[3]["triton_num_warps"] = args.triton_num_warps
    data[3]["triton_num_stages"] = args.triton_num_stages

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
            "triangle_backend": args.backend,
            "triton_block_m": args.triton_block_m,
            "triton_block_n": args.triton_block_n,
            "triton_block_k": args.triton_block_k,
            "triton_num_warps": args.triton_num_warps,
            "triton_num_stages": args.triton_num_stages,
        },
        "env": {
            "host": socket.gethostname(),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": getattr(triton, "__version__", None),
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
