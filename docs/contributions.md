# Public contribution ledger

Track every public issue, PR, review comment, benchmark gist, dashboard update,
and writeup that comes out of `swordfish`. The goal is not volume by itself; the
goal is a durable trail of maintainer-useful artifacts with exact repro context.

## Fields

- **Date:** when the artifact went public.
- **Upstream:** project, forum, or publication venue.
- **Type:** issue, PR, review, gist, writeup, dashboard update, or discussion.
- **Artifact:** link once public; keep draft names before publication.
- **Status:** planned, drafted, open, merged, closed, published, or superseded.
- **Outcome:** what changed because of it.
- **Lesson:** one sentence to carry into the next contribution.

## Ledger

| Date | Upstream | Type | Artifact | Status | Outcome | Lesson |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | Liger Kernel (LinkedIn / Microsoft) | Discussion + writeup + result JSONs | Cross-arch reproduction of Liger per-kernel and Llama-3-8B numbers on A100 / H100 NVL / H200 | planned (Week 1 first touchpoint) | Not started — Wed: per-kernel sweep; Thu: 8×A100 e2e stretch; Fri: publish | Cross-arch reproduction with NCU SOL fields and reproducible JSON is the gap. Discussion not Issue: share data, do not imply a bug. |
| TBD | Triton | Issue or docs PR | Minimal GEMM benchmark/correctness repro from the `swordfish` Triton backend | planned | Not started | Attach exact GPU, CUDA, Triton version, shape, expected behavior, actual behavior, and correctness diff. |
| TBD | PyTorch/Inductor | Issue | `torch.compile` repro from the GPT-style block or GEMM harness | planned | Not started | A useful compiler issue needs one file, one command, stable env metadata, and eager-vs-compiled comparison. |
| TBD | CUTLASS/CuTe | Docs/example PR or issue | Hopper FP8/FP4 GEMM example reproduction note | planned | Not started | Prefer tested build commands and small documentation fixes before proposing template changes. |
| TBD | vLLM | Issue, PR, or discussion | Quantized GEMM benchmark/correctness evidence tied to Machete/Marlin paths | planned | Not started | Serving maintainers need end-to-end relevance, not just microbenchmark speedups. |
| TBD | ONNX Runtime / ORT GenAI | Issue or PR | CUDA EP quant kernel map plus one targeted correctness/benchmark improvement | planned | Not started | ORT contributions should connect kernel details to provider behavior and tests. |
| TBD | pyptx | Docs/test PR or benchmark gist | H100/H200 reproduction data for an existing pyptx kernel | planned | Not started | This remains a sidecar: instruction-level learning plus benchmark provenance, not the main quant lane. |

## Rules

1. Do not publish a performance claim without the result JSON, source commit,
   shape, dtype, GPU, driver/CUDA, baseline, and correctness status.
2. Keep PRs narrow enough that a maintainer can review them without learning the
   whole `swordfish` project.
3. Record closed or rejected artifacts too; a rejected issue with a clear lesson
   is still useful signal.

## Packet generator

Before opening an issue, PR, gist, or discussion, generate a maintainer-ready
packet from the benchmark JSON:

```bash
uv run python -m swordfish.runner render-upstream-packet \
  --result /path/to/result.json \
  --target triton \
  --out /tmp/swordfish-upstream-packet.md
```

The packet includes the target, exact command, GPU/toolchain provenance,
correctness summary, latency summary, NCU completeness when attached, and common
result-protocol validation status. It is a draft body, not publication by
itself; paste/edit it into the upstream venue only after checking that the ask is
specific and maintainer-useful.
