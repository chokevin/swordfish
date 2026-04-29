# vLLM architecture map for quant-kernel work

This note is a kernel-contributor map, not a full vLLM guide. The question is:
where does a quantized GEMM change enter vLLM, and what serving constraints does
it have to survive?

## End-to-end path

### Engine and scheduler

vLLM v1 scheduling is token-budget driven rather than a clean split between
"prefill" and "decode" phases. The scheduler has to mix continuously changing
requests, chunked prefill, cached prefixes, and speculative decoding. For kernel
work, this means a fast GEMM in isolation is not enough: it must behave well
under heterogeneous token counts and changing batch composition.

### GPU model runner

The GPU model runner turns scheduled work into a GPU forward pass. It prepares
slot mappings, attention metadata, CUDA graph decisions, and model inputs. This
is the bridge between scheduling policy and the actual model/kernels.

For quantized GEMM contributions, the model runner matters because it determines
whether a shape is stable enough for CUDA graph capture and whether a custom
kernel sees the layout/metadata it expects.

### Forward context and CUDA graphs

`vllm/forward_context.py` defines `ForwardContext`, carrying attention metadata,
slot mappings, CUDA graph runtime mode, batch descriptors, and compile bypass
state through a forward pass. A quant kernel that allocates unexpectedly, depends
on unstable shapes, or requires Python-side dynamic behavior may break graph
capture or compiled execution.

Spot-check: upstream `ForwardContext` includes `attn_metadata`, `slot_mapping`,
`cudagraph_runtime_mode`, `batch_descriptor`, and `skip_compiled`.

### Attention backend abstraction

`vllm/v1/attention/backends/registry.py` maps backend names to importable
backend classes. The registry includes backends such as FlashAttention, Triton
attention, FlashInfer, CUTLASS MLA, TurboQuant, and custom overrides.

Spot-check: the registry exposes an `AttentionBackendEnum` whose values are
qualified class paths and whose `get_class()` resolves the backend dynamically.

### PagedAttention and KV-cache paths

PagedAttention is split between Python wrappers and lower-level ops/kernels.
The Python side prepares layout and dispatch, while low-level code owns the
cache write/read mechanics. For the quant-GEMM lane, attention is secondary
unless a GEMM change touches KV-cache format, decode batching, or quantized
attention integration.

## Concepts to answer quickly

| Concept | Short answer |
| --- | --- |
| PagedAttention | vLLM's KV-cache organization that lets requests share and reuse cache blocks while batching dynamically. |
| Continuous batching | Scheduler policy that keeps active requests moving together instead of running one static batch at a time. |
| Chunked prefill | Large prefills can be split so decode traffic is not blocked behind one long prompt. |
| Prefix caching | Previously computed prefix KV cache can be reused instead of recomputing the same tokens. |
| Attention backend abstraction | Attention implementations are selected through a backend registry rather than hard-coded at every callsite. |
| ForwardContext | Per-forward container for attention metadata, slot mapping, CUDA graph mode, batch descriptor, and compile bypass state. |
| CUDA graph capture | vLLM can replay stable GPU work through captured graphs; custom kernels must avoid dynamic behavior that breaks capture. |
| Quant stack | Python quantization layers choose methods and packing, while C++/CUDA code implements Marlin, Machete, and related kernels. |

## Quant-kernel contribution surfaces

### 1. Python quantization method selection

Surface:

```text
vllm/model_executor/layers/quantization/
```

Use this when the contribution is about method selection, scale/layout
validation, model-loading behavior, or routing to a backend. This is often the
reviewable place for correctness guards before touching CUDA.

### 2. Marlin and Machete CUDA kernels

Surfaces:

```text
csrc/quantization/marlin/
csrc/quantization/machete/
```

Use this when the contribution is truly about quantized GEMM implementation,
prepack/repack behavior, tile choice, or performance. This is the home-lane
surface for INT4 weight-only and related serving GEMM work.

### 3. Attention and KV-cache integration

Surfaces:

```text
vllm/v1/attention/ops/
csrc/attention/
csrc/nvfp4_kv_cache_kernels.cu
```

Use this only when the quant change touches cache format or attention-path
integration. Do not let this displace the GEMM lane unless the serving path
requires it.

## First vLLM contribution candidate

Start with a maintainer-useful issue or docs/benchmark PR around Machete/Marlin
quantized GEMM evidence:

1. Pick one decode-like GEMM shape.
2. Run the `swordfish` torch/Triton baseline with protocol JSON.
3. Reproduce the closest vLLM Marlin/Machete path if possible.
4. Attach shape, dtype, GPU, driver/CUDA, source commit, correctness error, and
   latency samples.
5. Frame the ask narrowly: correctness guard, benchmark harness clarity, or one
   layout/prepack question.

The goal is not to arrive with a hero kernel first. The goal is to produce data
that tells vLLM maintainers exactly which quantized GEMM path was tested and why
the result matters for serving.
