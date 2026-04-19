# swordfish design notes

## The problem in one sentence

On A100 at batch 4–16, Marlin under-utilizes tensor cores because its tile structure was optimized for batch=1 speculative decode; we want a kernel that keeps tensor cores fed across the voice-agent batch range.

## Anatomy of INT4×FP16 decode

Every kernel in this space does some variant of:

```
for each output tile (M_tile × N_tile):
    acc = 0
    for each K_tile:
        # load INT4 weights (packed)
        w_packed = cp.async from global memory into SMEM
        # load scales (and zeros, if asymmetric)
        scales = load from global
        # load activations (FP16)
        a = cp.async from global memory into SMEM
        # dequant
        w_fp16 = (w_packed.unpack() - zero) * scales
        # tensor-core matmul
        acc += mma(a, w_fp16)
    # epilogue: bias, activation, output cast
    write acc to global as FP16
```

## Where Marlin wins and loses on A100

**Wins:**
- At batch=1, the kernel is memory-bandwidth bound on weight loads. Marlin's cp.async pipeline and permuted weight layout keep HBM saturated.
- group_size=128 matches Marlin's tile width exactly → one scale load per K_tile.
- Tight code, low instruction count, good ICache behavior.

**Loses (or leaves on the table):**
- At batch 4–16, the bottleneck shifts from weight bandwidth to tensor-core feed rate. Marlin's tile sizing is still batch=1 shaped.
- group_size != 128 requires awkward partial loads → extra indexing overhead.
- Zero-point handling adds 2 ops per dequant per element, measurable at batch > 4.
- No fused epilogue; bias and activation are separate kernels.
- Pre-CUDA-graph design; re-launching per step at batch=1 decode has ~8µs overhead.

## swordfish design principles

1. **Two kernels, one dispatch.** Small-batch (M ≤ 4) kernel looks like Marlin. Large-batch (M ≥ 8) kernel uses different tile dimensions tuned for tensor-core feed rate.
2. **CUDA-graph-first.** Fixed kernel shape per (M, N, K, group_size) tuple. No dynamic dispatch inside the kernel.
3. **Arbitrary group_size.** Parameterize scale loads properly; accept the overhead at non-128 group sizes as a cost of flexibility.
4. **Fused epilogues.** Bias, SiLU, GELU, output cast all in one kernel. Decode sees this across every MLP and output projection.
5. **Numerical parity with FP16 reference.** No "close enough" tricks; the output of swordfish@group128 must match Marlin@group128 bit-exactly (or near-exactly).

## SM80 (A100) hardware constraints that shape the design

- 108 SMs, 64 FP32 / 64 INT32 / 4 tensor cores per SM
- 192 KB configurable L1/SMEM per SM (we'll use ~164 KB for SMEM)
- 40 MB L2 (shared across all SMs)
- 2 TB/s HBM2e (80 GB variant)
- `mma.sync.m16n8k16` is the tensor-core instruction for FP16
- `cp.async.cg.shared.global` for 16-byte async loads into SMEM
- Async copy has a pipeline depth of up to 8 outstanding requests

## Tile sizing first guesses (to be tuned)

**Small batch (M ≤ 4):**
- `BLOCK_M = 8`, `BLOCK_N = 128`, `BLOCK_K = 64`
- 2 warps per block
- 4-stage pipeline

**Large batch (M ≥ 8):**
- `BLOCK_M = 16`, `BLOCK_N = 128`, `BLOCK_K = 64`
- 4 warps per block
- 3-stage pipeline (less stages, larger tiles)

Will sweep these in week 3/4.

## Numerical strategy

- Accumulate in FP32 (tensor-core output is FP32)
- Apply scales after accumulation? Or fold into the weights as we dequant? Depends on group_size vs K_tile alignment — decide in week 2
- Output cast to FP16 at epilogue

## Out of scope

- No sparsity (neither 2:4 nor unstructured)
- No mixed-precision within a matmul (always FP16 activations)
- No quantization-aware *training* support — that's separate
