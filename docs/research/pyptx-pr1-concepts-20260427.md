# Research: Should swordfish adapt concepts from pyptx PR #1?

**Date:** 2026-04-27
**Asker:** user / swordfish kernel roadmap
**Decision:** ADAPT

## Question

Should `swordfish` adapt concepts from `patrick-toulme/pyptx` PR #1 into the kernel-learning roadmap now, and if so where without derailing Week 1?

## TL;DR

ADAPT by using pyptx as a learning-in-public upstream contribution lane, not as a `swordfish` runtime dependency yet. The PR is a useful model for instruction-level education, benchmark documentation, and memory-bound kernel structure, and the repo is young enough that careful benchmark/test/docs contributions can matter.

## What I read

| Source | Type | Date | What it said |
|---|---|---:|---|
| [pyptx PR #1](https://github.com/patrick-toulme/pyptx/pull/1) | code-repo / PR | 2026-04-27 | Adds a fused row-wise f32 softmax and reports 2.80 TB/s on H100, 1.16x faster than `torch.softmax` at B=2048 N=8192. |
| [PR diff: `examples/hopper/softmax.py`](https://github.com/patrick-toulme/pyptx/pull/1/files) | code-repo | 2026-04-27 | Implements one CTA per row, v4 global load/store, max reduction, `ex2(fma(...))`, sum reduction, and `rcp.approx.f32`. |
| [pyptx README](https://github.com/patrick-toulme/pyptx) | code-repo | 2026 | Describes pyptx as "one call = one instruction", with JAX/PyTorch launch paths and a PTX-to-Python transpiler. |
| [pyptx performance docs](https://pyptx.dev/performance/) | project docs | 2026 | Reports H100/B200 GEMM, grouped GEMM, norm, SwiGLU, softmax, launch-overhead numbers, and reproduction commands. |
| [pyptx comparison docs](https://pyptx.dev/comparison/) | project docs | 2026 | Says pyptx is the right answer for explicit WGMMA/TMA/tcgen05 control, but wrong when you want portability or compiler scheduling. |
| [NVIDIA PTX ISA 9.2 docs](https://docs.nvidia.com/cuda/parallel-thread-execution/) | official docs | 2026 | PTX is a stable virtual ISA; NVIDIA explicitly lists "hand-coding of libraries, performance kernels, and architecture tests" as a PTX goal. |
| [CUDA Driver API module docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) | official docs | 2026 | The driver supports loading PTX/cubin modules through APIs like `cuModuleLoadData` and JIT linking via `cuLinkAddData`. |
| [Triton `inline_asm_elementwise`](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html) | official docs | 2026 | Triton supports inline assembly over tensor elements, but the API is elementwise and uses LLVM-style constraints. |
| [Numba CUDA PTX compilation docs](https://numba.readthedocs.io/en/stable/cuda/cuda_compilation.html) | official docs | 2026 | Numba can compile Python functions to PTX/LTO-IR, but it is not a raw instruction-by-instruction PTX DSL. |

(Read budget: 9 sources across code-repo, project-docs, and official-docs. Stopped because the adoption answer was clear.)

## Findings

1. **The interesting thing is not the softmax alone; it is the pyptx programming model.** The README says "one call = one instruction" and the softmax PR makes that concrete with explicit `ld.global.v4.f32`, warp reductions, `ex2.approx.f32`, and `rcp.approx.f32`. This is exactly the kind of notation that helps learn PTX/SASS.

2. **PR #1 is a good educational memory-bound kernel pattern.** It uses one CTA per row, keeps f32 row values in registers for rows up to 8192, does a max pass and sum pass, and avoids online softmax because re-reading HBM is unnecessary for that shape. That is a useful lesson for the H200 bandwidth angle.

3. **pyptx fits a sidecar contribution lane better than the main quant-GEMM lane.** The comparison docs explicitly say pyptx is wrong when you want compiler scheduling or one kernel across GPU generations. `swordfish` still needs Triton/CuTe/CUTLASS for production-shaped quant GEMM work, but pyptx is a strong place to contribute benchmark hygiene, H100/H200 reproduction data, tests, and narrowly scoped examples while learning raw PTX.

4. **The launch-overhead numbers are directly relevant to swordfish.** pyptx reports ~14 us PyTorch eager via C++ extension, ~34 us ctypes, ~4 us CUDA graph replay. This reinforces the existing swordfish lesson that wrapper/launch overhead can dominate small kernels.

5. **The transpiler concept is worth validating upstream.** pyptx's claim that it can transpile PTX from nvcc/Triton/Pallas/DeepGEMM into editable Python suggests a powerful contribution loop: generate PTX from a Triton/CUDA kernel, round-trip it into a readable form, then contribute either corpus coverage, docs, or bug reports upstream.

## Counter-evidence

The strongest case against adding pyptx now is focus. The package is marked alpha (`Development Status :: 3 - Alpha` in `pyproject.toml`), targets Hopper/Blackwell specifically, and deliberately avoids optimizers/autotuners. PR #1's softmax also loses badly at small shapes because dispatch overhead dominates, and the win at the largest shape is 1.16x over eager torch, not a production replacement for fused attention/softmax stacks. Most importantly, Week 1's active objective is the A100/H100/H200 runner; adding a new PTX DSL dependency now would create toolchain risk before the measurement loop exists.

## Decision: ADAPT

Do not add `pyptx` as a `swordfish` dependency or rewrite any `swordfish` kernel around it now. After the runner is working, use pyptx as a small upstream contribution track: reproduce a benchmark on H100/H200, contribute clean benchmark artifacts or docs, and only then consider a narrowly scoped kernel/test PR.

## What this means in practice

- **First concrete move:** add a deferred pyptx upstream-contribution track depending on `airun-runner`; start by reproducing one memory-bound benchmark on H100/H200 and opening either a benchmark-data/docs PR or a well-scoped issue with numbers.
- **Watch-fors:** if pyptx gains stable releases, active maintainer response, and a clean H100/H200 install path in our runner image, reconsider optional dependency status.
- **Out of scope for this research:**
  - Porting pyptx softmax into `swordfish` immediately.
  - Benchmarking pyptx on our cluster.
  - Replacing Triton/CuTe/CUTLASS in the quant-GEMM roadmap.

## Open questions / what I'd read next

1. Can pyptx install cleanly inside the airun CUDA image without fighting JAX/PyTorch versions?
2. Does the PTX transpiler round-trip a Triton kernel from our own environment into useful readable Python?
3. Do pyptx kernels expose enough metadata to fit our benchmark manifest and dashboard schema cleanly?
