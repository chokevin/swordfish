"""Target-shape catalog for benchmarks and tests.

Add a new shape here and it's automatically picked up by the harness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    name: str
    M: int  # batch
    N: int  # output hidden (columns)
    K: int  # input hidden (rows, reduction)
    group_size: int = 128
    priority: int = 1  # 0 = P0 (headline), 1 = P1, 2 = P2
    tag: str = ""  # optional tag like "llama-3-8b", "qwen3"


# ---------- Llama-3-8B (hidden=4096) ----------
LLAMA_8B = [
    Shape("8b-b1", M=1, N=4096, K=4096, group_size=128, priority=0, tag="llama-3-8b"),
    Shape("8b-b4", M=4, N=4096, K=4096, group_size=128, priority=0, tag="llama-3-8b"),
    Shape("8b-b8", M=8, N=4096, K=4096, group_size=128, priority=0, tag="llama-3-8b"),
    Shape("8b-b16", M=16, N=4096, K=4096, group_size=128, priority=1, tag="llama-3-8b"),
    Shape("8b-b1-g64", M=1, N=4096, K=4096, group_size=64, priority=1, tag="llama-3-8b"),
    Shape("8b-b4-g64", M=4, N=4096, K=4096, group_size=64, priority=1, tag="llama-3-8b"),
]

# ---------- Llama-3-70B with TP=2 (per-GPU hidden=4096, N=8192 on QKV/out) ----------
LLAMA_70B_TP2 = [
    Shape("70b-tp2-b1", M=1, N=8192, K=4096, group_size=128, priority=0, tag="llama-3-70b"),
    Shape("70b-tp2-b4", M=4, N=8192, K=4096, group_size=128, priority=0, tag="llama-3-70b"),
    Shape("70b-tp2-b8", M=8, N=8192, K=4096, group_size=128, priority=0, tag="llama-3-70b"),
    Shape("70b-tp2-mlp", M=8, N=14336, K=4096, group_size=128, priority=1, tag="llama-3-70b"),
]

# ---------- Hybrid / Qwen-ish ----------
QWEN_HYBRID = [
    Shape("qwen-3584-b4", M=4, N=3584, K=3584, group_size=128, priority=1, tag="qwen3"),
    Shape("qwen-5120-b4", M=4, N=5120, K=5120, group_size=128, priority=1, tag="qwen3"),
]

ALL_SHAPES: list[Shape] = LLAMA_8B + LLAMA_70B_TP2 + QWEN_HYBRID

SHAPE_SETS: dict[str, list[Shape]] = {
    "voice": [s for s in ALL_SHAPES if s.priority == 0],
    "full": ALL_SHAPES,
    "llama8b": LLAMA_8B,
    "llama70b": LLAMA_70B_TP2,
    "qwen": QWEN_HYBRID,
}


def resolve(name: str) -> list[Shape]:
    if name not in SHAPE_SETS:
        raise KeyError(f"unknown shape set {name!r}; known: {list(SHAPE_SETS)}")
    return SHAPE_SETS[name]
