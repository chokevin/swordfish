"""End-to-end perplexity validation for W4A16 kernels.

Loads a HF causal LM, swaps every nn.Linear in the transformer blocks with
a QuantLinear that routes the matmul through one of {fp16, reference,
marlin, swordfish}, and computes WikiText-2 perplexity.

This is the L2 layer of our correctness pyramid:
- L0: per-kernel allclose vs reference (in `bench/run_bench.py`)
- L1: per-layer correctness on real GPTQ weights (TODO)
- L2: end-to-end perplexity on a real model (this script)

Acceptance bands (Llama-3-8B-Instruct on WikiText-2, seq_len 2048):
- fp16 baseline:        ~6.1   (the floor; what we'd ship if size weren't a concern)
- marlin/GPTQ-int4:     ~6.2   (≤ 0.15 over fp16 = the quant tax)
- swordfish/GPTQ-int4:  marlin ± 0.001  (same math, must match within accumulation noise)
- reference/GPTQ-int4:  marlin ± 0.005  (same math, fp32 accumulation)

If swordfish drifts > 0.001 from marlin on the same packed weights, the
kernel has a numerical bug — fail the run.

Why WikiText-2 and not C4? Standard published-numbers anchor — Marlin paper,
GPTQ paper, AWQ paper, every quant paper uses WikiText-2 raw. We can compare
to their tables.

Usage:
    uv run python -m bench.eval_ppl \\
        --model TheBloke/Llama-3-8B-Instruct-GPTQ \\
        --impls fp16,marlin,swordfish \\
        --seq-len 2048 \\
        --out runs/ppl-$(date -u +%Y%m%dT%H%M%SZ)

Outputs:
    <out>/ppl.csv         one row per (impl, dataset)
    <out>/ppl.json        full manifest with env + per-chunk losses
    <out>/ppl_summary.md  human-readable table for PR description

Reference for the math:
    https://huggingface.co/docs/transformers/perplexity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# QuantLinear: nn.Linear shape, kernel-swappable matmul
# --------------------------------------------------------------------------


@dataclass
class PplResult:
    impl: str
    model: str
    dataset: str
    seq_len: int
    n_chunks: int
    nll_sum: float
    n_tokens: int
    ppl: float
    elapsed_s: float
    error: str | None = None


class QuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear over packed W4A16 weights.

    Holds a (packed, scales, group_size) tuple, dispatches to one of our
    impls. The same packed tensor is shared across impls so any PPL delta
    is purely a kernel-arithmetic delta, not a quantization-scheme delta.
    """

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        bias: torch.Tensor | None,
        impl: str,
    ):
        super().__init__()
        # Use buffers so .to(device) works and they don't get caught by
        # .parameters() / optimizer.
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        if bias is not None:
            self.register_buffer("bias_t", bias, persistent=False)
        else:
            self.bias_t = None
        self.group_size = group_size
        self.impl = impl
        # Cache the marlin-layout weights once per layer to avoid repacking
        # on every forward (W2 lesson: per-call alloc is the long pole).
        self._marlin_w = None
        self._marlin_s = None

    def _matmul(self, a: torch.Tensor) -> torch.Tensor:
        if self.impl == "fp16":
            from swordfish.reference import dequantize_int4

            w = dequantize_int4(self.packed, self.scales, group_size=self.group_size)
            return torch.matmul(a, w)
        if self.impl == "reference":
            from swordfish.reference import reference_w4a16_matmul

            return reference_w4a16_matmul(
                a, self.packed, self.scales, group_size=self.group_size
            )
        if self.impl == "marlin":
            from swordfish.marlin_compat import marlin_matmul, to_marlin_layout

            if self._marlin_w is None:
                self._marlin_w, self._marlin_s = to_marlin_layout(
                    self.packed, self.scales, group_size=self.group_size
                )
            return marlin_matmul(
                a, self._marlin_w, self._marlin_s, group_size=self.group_size
            )
        if self.impl == "swordfish":
            from swordfish.kernels.triton_w4a16 import triton_w4a16_matmul

            return triton_w4a16_matmul(
                a, self.packed, self.scales, group_size=self.group_size
            )
        raise ValueError(f"unknown impl: {self.impl}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., K] -> flatten to [B, K], matmul, restore shape.
        in_shape = x.shape
        x2 = x.reshape(-1, in_shape[-1]).to(torch.float16)
        out = self._matmul(x2)
        out = out.reshape(*in_shape[:-1], -1)
        if self.bias_t is not None:
            out = out + self.bias_t
        return out


# --------------------------------------------------------------------------
# Linear-swap loop
# --------------------------------------------------------------------------


def _iter_linears(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, child, module, name
        else:
            yield from _iter_linears(child, full)


def swap_linears_with_quantlinear(model: nn.Module, impl: str, *, skip_lm_head: bool = True) -> int:
    """Walks the model, replaces every nn.Linear with QuantLinear[impl].

    Quantizes weights on the fly (group-quant, group_size=128) using the
    SAME pack/scale machinery as our synthetic bench — so the QUANTIZATION
    is identical across impls, isolating kernel arithmetic as the only
    variable.

    Returns the number of layers swapped.
    """
    from swordfish.pack import quantize_symmetric_int4

    swapped = 0
    for full, lin, parent, attr in list(_iter_linears(model)):
        if skip_lm_head and full.endswith("lm_head"):
            # Output head usually kept in fp16 — quantizing it shifts PPL
            # in a way that's not about the kernel under test.
            continue
        w = lin.weight.data.to(torch.float16)  # [out, in]
        # Our pack convention: weight as [K, N] = [in, out]. nn.Linear
        # stores W as [out, in]; transpose so K is the contracting dim.
        w_kn = w.t().contiguous()
        packed, scales = quantize_symmetric_int4(w_kn, group_size=128)
        bias = lin.bias.data.to(torch.float16) if lin.bias is not None else None
        ql = QuantLinear(packed, scales, group_size=128, bias=bias, impl=impl)
        ql = ql.to(w.device)
        setattr(parent, attr, ql)
        swapped += 1
    return swapped


# --------------------------------------------------------------------------
# Perplexity loop (HF-canonical: sliding window over concatenated text)
# --------------------------------------------------------------------------


def load_wikitext2(tokenizer, seq_len: int):
    """Returns one big tensor of token ids — the standard PPL anchor.

    Following GPTQ/AWQ/Marlin papers exactly: concat the test split with
    \\n\\n separators, then chunk into non-overlapping seq_len windows.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]  # [T]
    n_chunks = input_ids.numel() // seq_len
    return input_ids[: n_chunks * seq_len].view(n_chunks, seq_len)


@torch.no_grad()
def compute_perplexity(model, chunks: torch.Tensor, device: str) -> tuple[float, float, int]:
    """Per HF reference: sum of token NLL over all chunks, divided by total
    tokens, then exp().

    Returns (ppl, nll_sum, n_tokens).
    """
    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    for i in range(chunks.shape[0]):
        ids = chunks[i : i + 1].to(device)
        out = model(ids, labels=ids)
        # HF returns mean NLL over (seq_len - 1) tokens. Multiply back out.
        per_chunk_tokens = ids.numel() - 1
        nll_sum += float(out.loss.item()) * per_chunk_tokens
        n_tokens += per_chunk_tokens
    ppl = math.exp(nll_sum / n_tokens)
    return ppl, nll_sum, n_tokens


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def env_capture() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model id. Defaults to Llama-3-8B-Instruct (gated, needs HF auth).",
    )
    p.add_argument("--impls", default="fp16,marlin", help="comma list")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-chunks", type=int, default=0, help="0 = full test set; small N for smoke")
    p.add_argument("--out", type=Path, default=Path("runs/ppl-latest"))
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer + model ({args.model})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    chunks = load_wikitext2(tok, args.seq_len)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    print(f"  wikitext-2 test: {chunks.shape[0]} chunks of {args.seq_len} tokens", flush=True)

    results: list[PplResult] = []
    for impl in impls:
        print(f"\n=== impl={impl} ===", flush=True)
        t0 = time.time()
        # Reload model fresh per impl. fp16 weights -> on-the-fly quant in
        # swap_linears_with_quantlinear. Cheap correctness over speed.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=device
        )
        try:
            if impl == "fp16-baseline":
                # Skip quantization entirely; pure fp16 reference.
                pass
            else:
                n = swap_linears_with_quantlinear(model, impl)
                print(f"  swapped {n} linears -> QuantLinear[{impl}]", flush=True)
            ppl, nll_sum, n_tok = compute_perplexity(model, chunks, device)
            elapsed = time.time() - t0
            print(f"  ppl={ppl:.4f}  nll_sum={nll_sum:.2f}  n_tokens={n_tok}  elapsed={elapsed:.1f}s", flush=True)
            results.append(
                PplResult(
                    impl=impl,
                    model=args.model,
                    dataset="wikitext-2-raw-v1/test",
                    seq_len=args.seq_len,
                    n_chunks=int(chunks.shape[0]),
                    nll_sum=nll_sum,
                    n_tokens=n_tok,
                    ppl=ppl,
                    elapsed_s=elapsed,
                )
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            err = f"{type(e).__name__}:{str(e)[:200]}"
            print(f"  ERROR: {err}", flush=True)
            results.append(
                PplResult(
                    impl=impl,
                    model=args.model,
                    dataset="wikitext-2-raw-v1/test",
                    seq_len=args.seq_len,
                    n_chunks=int(chunks.shape[0]),
                    nll_sum=0.0,
                    n_tokens=0,
                    ppl=float("nan"),
                    elapsed_s=elapsed,
                    error=err,
                )
            )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- write CSV + JSON + markdown summary ---
    fields = [f.name for f in PplResult.__dataclass_fields__.values()]  # type: ignore
    with (args.out / "ppl.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    with (args.out / "ppl.json").open("w") as f:
        json.dump({"env": env_capture(), "args": vars(args), "results": [asdict(r) for r in results]}, f, indent=2, default=str)

    # markdown summary, with delta-vs-marlin if marlin present (the band
    # check the user actually wants to see)
    marlin_ppl = next((r.ppl for r in results if r.impl == "marlin" and r.error is None), None)
    lines = [
        f"# perplexity — {args.model}",
        "",
        f"dataset: wikitext-2-raw-v1/test, seq_len={args.seq_len}, chunks={int(chunks.shape[0])}",
        "",
        "| impl | ppl | Δ vs marlin | elapsed | error |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        d = ""
        if marlin_ppl is not None and r.error is None and r.impl != "marlin":
            d = f"{r.ppl - marlin_ppl:+.4f}"
        elif r.impl == "marlin":
            d = "—"
        lines.append(
            f"| {r.impl} | {r.ppl:.4f} | {d} | {r.elapsed_s:.1f}s | {r.error or ''} |"
        )
    lines += [
        "",
        "## acceptance bands",
        "- swordfish vs marlin: |Δ| ≤ 0.001 (kernel arithmetic must match)",
        "- reference vs marlin: |Δ| ≤ 0.005 (fp32 accumulation, same algo)",
        "- marlin vs fp16-baseline: |Δ| ≤ 0.15 (the quant tax)",
        "",
    ]
    (args.out / "ppl_summary.md").write_text("\n".join(lines))
    print(f"\nwrote {args.out / 'ppl.csv'}, {args.out / 'ppl.json'}, {args.out / 'ppl_summary.md'}")

    # Hard fail if swordfish drifts from marlin beyond band.
    if marlin_ppl is not None:
        sf = next((r for r in results if r.impl == "swordfish" and r.error is None), None)
        if sf and abs(sf.ppl - marlin_ppl) > 0.001:
            print(f"FAIL: swordfish drift {sf.ppl - marlin_ppl:+.4f} > 0.001 PPL", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
