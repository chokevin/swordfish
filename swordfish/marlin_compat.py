"""Marlin layout adapter.

Converts swordfish's ``(packed, scales)`` pair into Marlin's expected weight
layout, and exposes a thin ``marlin_matmul`` wrapper around ``marlin.mul``.

We *delegate* the permutation to Marlin's own packing utility rather than
reverse-engineering the permutation table — Marlin's ``Layer.pack`` is
authoritative, and re-quantizing the dequantized fp16 yields bit-identical
INT4 values for symmetric quant (it's deterministic absmax → divide → round).

Marlin is not importable on macOS; this module is therefore split into:
  - ``to_marlin_layout`` / ``marlin_matmul``: GPU-only, raise on missing dep.
  - ``dequantized_for_marlin``: CPU-runnable; what the GPU path feeds Marlin.
The CPU test in ``tests/test_marlin_compat.py`` exercises the dequant path,
which is what would fail silently if ``swordfish.pack`` ever changed format.

Workspace cache (rvLLM lesson §3.1, our W1 finding):
  ``marlin.mul`` requires a small int32 scratch buffer sized by N. Allocating
  it on every call adds ~5–10 µs of host-side overhead — material when the
  kernel itself is ~17 µs. We cache one workspace per (N, device) tuple at
  module level. This is the single-arena pattern from rvLLM, scoped down.
"""

from __future__ import annotations

import torch

from swordfish.reference import dequantize_int4

# Workspace cache: (N, device_str) -> persistent int32 tensor.
# Replaces the per-call torch.zeros allocation that dominated the wrapper
# overhead identified in W1 (marlin-bottlenecks.md). The tensor is filled
# with zeros once on creation; marlin.mul does not require it to be re-zeroed
# between calls (it's used as scratch, not accumulator). We DO zero it on
# eviction-style reuse just to be safe — see _get_workspace.
_WORKSPACE_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _get_workspace(N: int, device: torch.device) -> torch.Tensor:
    """Return a zeroed int32 scratch buffer for marlin.mul.

    Cached per (N, device) — the cost we eliminate is the per-call
    `torch.zeros(...)` ALLOCATION (~10 µs of host-side overhead identified
    in the W1 bottleneck analysis), not the zeroing itself. Zeroing a
    16-int tensor is ~1 µs and is REQUIRED: marlin's upstream uses this
    buffer as an inter-CTA tile counter / sync barrier; stale values from
    a previous call have caused silent reduction corruption in some
    versions. We zero on every access — cheap, safe.

    Capture caveat (rvLLM §5.2): the FIRST call for a new (N, device)
    allocates inside the captured region, which binds to a stale device
    offset on replay. Callers that capture into a CUDA graph MUST warm
    this cache (call marlin_matmul once outside capture) before the
    capture region. ``cuda_graph_time_ms`` in the bench harness does this
    correctly via its warmup loop. New captured callers must do the same.
    """
    key = (N, str(device))
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        # 16 ints per 128-wide N-tile is the upstream marlin convention.
        ws = torch.zeros(N // 128 * 16, device=device, dtype=torch.int32)
        _WORKSPACE_CACHE[key] = ws
    else:
        ws.zero_()
    return ws


def clear_workspace_cache() -> None:
    """For tests that want to verify allocation behavior."""
    _WORKSPACE_CACHE.clear()


def dequantized_for_marlin(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Returns the fp16 [N, K] tensor in nn.Linear-weight layout that Marlin's
    packer expects. CPU-runnable. This is what we hand to Marlin on the A100."""
    w_fp = dequantize_int4(packed, scales, group_size=group_size)  # [K, N]
    return w_fp.t().contiguous()  # [N, K]


def to_marlin_layout(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (swordfish-packed, scales) -> (marlin_B, marlin_s).

    Strategy: dequantize → build an nn.Linear → call ``marlin.Layer.pack``,
    which performs Marlin's own quantize + permute. This is the only way to
    guarantee layout compatibility without re-implementing Marlin's
    permutation tables.

    Requires Marlin and a CUDA device. Raises ImportError on Mac / non-CUDA.
    """
    if packed.device.type != "cuda":
        raise RuntimeError("to_marlin_layout requires CUDA tensors")

    try:
        import marlin  # type: ignore
    except ImportError as e:
        raise ImportError("marlin not installed. See docs/profiling/RUN_ME_ON_A100.md.") from e

    K, N_half = packed.shape
    N = N_half * 2

    w_fp_NK = dequantized_for_marlin(packed, scales, group_size=group_size)  # [N, K]

    linear = torch.nn.Linear(K, N, bias=False).to(device=packed.device, dtype=torch.float16)
    linear.weight.data.copy_(w_fp_NK)

    layer = marlin.Layer(infeatures=K, outfeatures=N, groupsize=group_size)
    layer = layer.to(packed.device)
    # Marlin's Layer.pack(linear, scales) — scales shape is [K//group, N]
    layer.pack(linear, scales)
    return layer.B, layer.s


def marlin_matmul(
    a: torch.Tensor,  # [M, K] fp16
    B: torch.Tensor,  # marlin-packed weight from to_marlin_layout
    s: torch.Tensor,  # marlin scales from to_marlin_layout
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Thin wrapper around ``marlin.mul``. Returns [M, N] fp16.

    `out` may be passed to skip the per-call output allocation — important
    inside a CUDA-graph capture region (rvLLM §5.2 lesson: any allocation
    inside capture binds to a stale device offset on replay).
    """
    try:
        import marlin  # type: ignore
    except ImportError as e:
        raise ImportError("marlin not installed.") from e

    M, _K = a.shape
    N = s.shape[1]
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.float16)
    workspace = _get_workspace(N, a.device)
    marlin.mul(a, B, out, s, workspace)
    return out
