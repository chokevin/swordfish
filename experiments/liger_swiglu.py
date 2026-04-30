#!/usr/bin/env python3
"""Liger SwiGLU vs HF reference, bf16. Sibling of liger_rmsnorm.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from swordfish.runner.liger_perkernel import run_liger_perkernel
from swordfish.runner.schema import gpu_class_from_name


def main() -> int:
    arch = gpu_class_from_name(
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        fallback=os.environ.get("SWORDFISH_ARCH_LABEL"),
    )
    data_dir = Path(os.environ.get("RUNE_DATA_DIR", "/data"))
    out_path = data_dir / "swordfish" / "week1" / "liger-perkernel" / f"swiglu-{arch}.json"

    print(f"[experiment] arch={arch} out={out_path}", file=sys.stderr)

    result = run_liger_perkernel(
        kernel="swiglu",
        batch=4,
        seq=2048,
        hidden=4096,
        intermediate=14336,
        eps=1e-6,
        dtype="bf16",
        repeats=5,
        warmup=10,
        iters=50,
        device_name="auto",
        arch_label=arch if arch != "unknown" else None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[experiment] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
