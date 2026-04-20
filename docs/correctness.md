# Correctness validation pyramid

Speed numbers from a wrong kernel are worse than no number at all. This
doc is the source-of-truth for how we keep swordfish numerically honest
as we trade FP16 accumulation for INT4 weights, fuse epilogues, and swap
out tile shapes.

We layer correctness checks from cheapest to most expensive. Every layer
must pass before we trust a speed number; we never report a TFLOPS for a
kernel that flunked a lower layer.

## L0 — element-wise allclose vs reference (every bench run, ~1 ms/shape)

**What:** `bench/run_bench.py:bench_shape` runs each impl's matmul on the
same `(packed, scales, group_size)` triple and compares its output to
`reference_w4a16_matmul` (FP32 accumulation, dequant-then-matmul). Tolerances
`atol=5e-3 rtol=5e-2` are picked from the bounded error of group-quant W4A16
under FP16 accumulation over a 4096-K reduction.

**Catches:** scale-format mismatches, packing-layout off-by-ones, group-size
boundary bugs, sign/zero-point handling errors, transposed weights.

**Output:** every CSV row gets a `max_relerr` and `correct` field. A failing
impl skips timing and emits `error="correctness_failed:max_relerr=…"` so
the operator sees HOW wrong before fixing.

**Limit:** uses synthetic random weights with a known clean distribution.
Real GPTQ/AWQ weights have outliers, mixed-precision groups, and asymmetric
zero-points that synthetic tensors don't stress.

## L1 — layer correctness on real GPTQ weights (TODO, week 2 deliverable)

**What:** load one real `Llama-3-8B-Instruct-GPTQ` linear (e.g.
`model.layers.0.mlp.down_proj`), feed it a real activation captured from
the model running on a WikiText prompt, and compare each impl's output to
the reference path.

**Catches:** what L0 misses — outlier handling, real activation distribution,
the actual quantization scheme used by the world's GPTQ models (asymmetric,
act-order permutations, varying group_size at boundaries).

**Status:** not yet implemented. Tracked as a W2 todo.

## L2 — end-to-end perplexity on WikiText-2 (`bench/eval_ppl.py`)

**What:** load Llama-3-8B-Instruct in FP16, walk the module tree, replace
every `nn.Linear` (except `lm_head`) with a `QuantLinear` whose `forward`
routes through one of `{fp16, reference, marlin, swordfish}`. Quantize
weights on the fly with the SAME `quantize_symmetric_int4` used by the
synthetic bench, so quantization is held constant across impls. Run the
WikiText-2 raw test split through the model in non-overlapping `seq_len`
chunks (HF-canonical PPL recipe), compute `exp(sum(NLL) / n_tokens)`.

**Acceptance bands** (Llama-3-8B-Instruct, seq_len=2048, full test set):

| compare | band | meaning |
|---|---|---|
| `marlin` vs `fp16-baseline` | ≤ 0.15 PPL | the quant tax we accept |
| `reference` vs `marlin` | ≤ 0.005 PPL | same algo, FP32 vs FP16 accum drift |
| `swordfish` vs `marlin` | ≤ 0.001 PPL | **kernel arithmetic must match** |

The third band is the one we care about. Marlin and swordfish run on the
SAME packed tensor; any PPL delta is purely a kernel-arithmetic delta
(accumulation order, intermediate dtype, fused-epilogue rounding). A
delta above 0.001 means our kernel is doing different math, not just
faster math. The script exits non-zero in that case so it can gate CI.

**Catches:** averaging effects that L0/L1 miss. A kernel can be `allclose`
on a single shape and still drift cumulatively across 32 layers × 2048
tokens. PPL surfaces the cumulative drift as one scalar.

**Anchor:** WikiText-2 raw is the universal quant-paper benchmark (GPTQ,
AWQ, Marlin, SmoothQuant all publish on it), so we can sanity-check our
absolute number against the published table.

**Cost:** ~10 min per impl on one A100 for full Llama-3-8B-Instruct over
the WikiText-2 test (~280 chunks of 2048 tokens). Acceptable for the
autoresearch loop. Use `--max-chunks 16` for smoke (~30 s).

## L3 — generation-quality benchmarks (deferred, post-W8)

PPL is a single scalar over a fixed corpus. It misses mode collapse, RLHF
breakage, repetition loops. If we ever ship a kernel that changes math
non-trivially (e.g. mixed-FP8 accumulation), we'd add HumanEval / MMLU /
MT-Bench. Not in scope for the 8-week window.

## How this maps to the autoresearch pod

- **Every cluster run** runs L0 (it's part of `run_bench.py`). Result lands
  in `results.csv` columns `correct` and `max_relerr`.
- **Opt-in** L2 via `RUN_PPL=1` env on the pod, gated by HF auth secret
  for the gated Llama-3 model. Adds ~30 min to the run; not on the default
  iteration loop.
- L0 failure short-circuits the rest of the run — we don't open a draft
  PR with a known-broken kernel.

## Useful invocations

```bash
# L0 only (default; runs on every bench)
uv run python -m bench.run_bench --shapes voice --impls fp16,marlin,swordfish --out runs/foo

# L2 smoke (16 chunks, ~30s on A100)
uv run python -m bench.eval_ppl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --impls fp16-baseline,marlin,swordfish \
  --max-chunks 16 \
  --out runs/ppl-smoke

# L2 full (~10 min/impl on A100)
uv run python -m bench.eval_ppl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --impls fp16-baseline,marlin,swordfish \
  --out runs/ppl-full
```
