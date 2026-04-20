# Lessons from rvLLM (Norris, 2026) applied to swordfish

> rvLLM: single-GPU FP8 LLM inference engine in Rust, 14–23% faster than
> vLLM at every batch size. Paper:
> https://docs.solidsf.com/docs/paper/rvllm.pdf

rvLLM is an inference engine; swordfish is a kernel. But the paper's
architectural lessons land directly on the W1 finding we just confirmed
(`docs/profiling/marlin-bottlenecks.md`): **our wall-clock loss vs cuBLAS
is in the wrapper, not the kernel**. rvLLM is independent confirmation
that the wrapper IS the optimization surface, plus a recipe.

## What we adopted

### 1. CUDA-graph capture as a measurement primitive (rvLLM §7.3)

rvLLM went 551 → 14,745 tok/s (27×) from CUDA-graph capture alone, before
any kernel work. That is the same lever we identified as W2 priority #3,
but the paper's number makes it priority #0.

**Implemented:** `bench/run_bench.py:cuda_graph_time_ms` and a `--capture`
flag. Every bench run now reports BOTH eager and graph-captured timings
per impl, with `wrapper_overhead_ms = ms_mean - ms_captured` as a
derived column. This is a measurement, not a fix — it tells us how big
the wrapper share of wall-clock is BEFORE we attempt to remove it. The
fix lives in W2.

**Methodology trap (rvLLM §5.2):** anything that allocates inside the
captured region binds to a stale device offset on replay and produces
silent wrong-data. Our `marlin_compat.marlin_matmul` now accepts an
optional `out=` kwarg specifically so a captured caller can pre-allocate
output once and rebind. The internal int32 workspace is module-cached
(see #2), so the only remaining per-call alloc is the output tensor.

### 2. Pre-allocated workspace cache (rvLLM §3.1)

rvLLM does ONE `cuMemAlloc` at startup sized for the worst case across all
bucket × variant × shape combinations. We don't own enough state to need
a full arena, but the principle is the same.

**Implemented:** `swordfish/marlin_compat._WORKSPACE_CACHE` keyed by
`(N, device)`. Replaces the per-call `torch.zeros(N // 128 * 16, …)` that
was directly identified in the W1 bottleneck doc as a wrapper-overhead
contributor. Module-level dict is the cheapest possible "arena."

### 3. Cosine similarity in the L0 correctness gate (rvLLM §3.3)

rvLLM holds every fused kernel to "cosine 0.999 for FP8 outputs" against
a pure-Rust f32 reference. Cosine is the right shape for "do these
compute the same math up to accumulation order" because it's invariant
to per-row scale — `allclose` with a generous rtol can pass a kernel
whose every output is biased by a constant factor.

**Implemented:** `bench/run_bench.py:_cosine_sim` plus a `cosine` column
in the CSV. The L0 gate now requires `allclose AND cosine ≥ 0.999`.

### 4. No silent fallbacks (rvLLM §5.1)

> Missing autotune-policy entry for a shape panics with the shape.
> Missing libfa3_kernels.so refuses startup.

Our `bench/run_bench.py:impl_marlin` returns `error="not_installed"` when
Marlin is missing — appropriate for Mac dev (the user can ignore the
column). But on the autoresearch pod, missing Marlin means the run is
useless and someone has to notice the missing row in the PR.

**Implemented:** `deploy/image/run.sh` has a post-bench gate that exits
non-zero if any IMPL listed in the request didn't produce a successful
row. The autoresearch pod will fail loudly instead of opening a
half-empty PR.

## What we documented but deferred

### Bucketed graph dispatch (rvLLM §2.2)

rvLLM pre-captures one CUDA graph per bucket size `{1,2,4,8,16,24,32,…,256}`
at engine init and dispatches to the nearest bucket-up at call time. Voice
agent's batch 1–16 maps cleanly to `{1,2,4,8,16}` — we'd need at most 5
captured graphs.

**Why deferred:** swordfish is a kernel library, not a serving engine. The
right place for bucketed dispatch is in the *caller* (vLLM, our own future
runtime), not in `marlin_compat.py`. We make the kernel cheap to capture
(see #1, #2); the caller decides how many buckets to capture.

### Autotune as a build artifact (rvLLM §3.1)

rvLLM ships `policy.json` SHA-pinned alongside the binary; runtime does
NOT consult `~/.cache/rvllm/cutlass_autotune.json`. Stale caches from
prior deploys cannot influence dispatch.

**Why deferred:** we don't have an autotune surface yet (Marlin uses one
hand-pinned variant per shape; our future Triton kernels will introduce
block-size choices in W3). When that lands, the policy lives in the repo
under `swordfish/autotune/policy.json`, not in `~/.cache`.

### `GraphSafe` trait — type-system enforcement that no-alloc invariants
hold inside capture region (rvLLM §2.2)

In Rust, `&mut HbmArena` is not in scope inside a `CaptureScope` closure;
runtime realloc inside capture is unrepresentable. Python can't get this
guarantee from the type system, but we can approximate with a runtime
assertion: `torch.cuda.memory_allocated()` should not change between the
start and end of a captured region.

**Why deferred:** we don't yet have multi-kernel composites under capture
in this repo. When the W2 deliverable lands a "captured marlin call" path,
add a `with capture_scope():` context manager that asserts the no-alloc
invariant. Until then, eager testing catches the same bugs.

### Per-step launch count as a reported metric (rvLLM §4)

rvLLM reports "311 kernel launches per decode step" as a first-class
number. Our analog: "kernel launches per matmul call". Marlin = 1 (just
`marlin.mul`). Triton W4A16 in our W2 plan = 1 (a single Triton kernel).
Worth having in the bench output once we add a second kernel that's a
multi-launch composite (e.g., dequant + GEMM split).

**Why deferred:** trivially 1 for both impls today; no information.

## What we explicitly did NOT take

- **Rust runtime / no-Python-in-hot-path.** rvLLM's headline. Out of
  scope: we're a kernel library, our caller is whoever (vLLM, custom
  runtime, etc.). The wrapper-overhead lesson is what matters; the
  language is not.
- **CUTLASS schedule-pairing static_assert (rvLLM §5.4).** SM90-specific
  (Hopper). We're A100 (SM80); CUTLASS templates we'd touch don't have
  this pairing constraint.
- **FP8 KV cache.** We don't touch attention or KV.
- **31 crates, DAG-enforced dependency graph.** Premature for a
  ~2K-line kernel project.

## Takeaways for the W2 plan

The rvLLM paper changes the order of W2 work but not the content:

1. **(was W2 #3, now #0) Capture marlin.mul into a CUDA graph.** Use the
   `--capture` flag we just added to MEASURE the win; then add a
   `marlin_matmul_captured()` API that takes a pre-allocated output and
   uses graph replay internally.
2. **(was W2 #1, now done) Workspace cache.** Shipped this commit.
3. **(was W2 #2, still on the list) Eliminate per-call dtype/contiguity
   asserts** in `marlin_matmul` — small win but free.
4. Then attack the kernel itself.

The biggest single change in priority: rvLLM's 27× from graph capture
alone tells us we may not need to write a faster kernel at all to beat
Marlin at voice batch sizes. The wrapper, captured, is enough.
