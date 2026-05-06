# Research: What can beat Liger for our FSDP A100 test?

**Date:** 2026-05-03  
**Asker:** Kevin / Swordfish kernel lab  
**Decision:** ADAPT

## Question

For our Llama-3-8B-like bf16 FSDP1 train-step test on 8xA100, is there an existing kernel or training stack that is likely to beat Liger Kernel?

## TL;DR

No public source shows a clean drop-in replacement that beats Liger on the exact same Hugging Face + PyTorch FSDP1 + bf16 + 8xA100 setup. The credible "beat Liger" paths are stack changes: PyTorch FSDP2/`torch.compile`/TorchTitan for a close PyTorch-native variant, or Megatron-Core/Transformer Engine for a higher-rewrite NVIDIA stack. For Swordfish, keep Liger as the current same-test baseline and run a targeted ADAPT experiment against FSDP2/compile before considering a Megatron rewrite.

## What I read

| Source | Type | Date | What it said |
|---|---|---:|---|
| [Liger-Kernel paper](https://arxiv.org/html/2410.10989v3) | paper | 2024 | Liger reports average 20% training-throughput gain and 60% GPU-memory reduction via Triton fusion/chunking, and explicitly supports FSDP/DeepSpeed/DDP. |
| [Liger-Kernel README](https://github.com/linkedin/Liger-Kernel) | code-repo | 2026 | Its headline benchmark is exactly close to ours: Llama-3-8B, batch 8, bf16, AdamW, gradient checkpointing, FSDP1 on 8 A100s. |
| [PyTorch: Maximizing Training Throughput](https://pytorch.org/blog/maximizing-training-throughput/) | vendor-blog | 2024 | `torch.compile` + selective activation checkpointing raised 7B A100 MFU from 57% to 68%, with 10-23% MFU gains across model sizes. |
| [SimpleFSDP paper](https://arxiv.org/abs/2411.00284) | paper | 2024 | Compiler-friendly FSDP can trace communication and reorder/bucket IR nodes for overlap, reporting up to 68.67% throughput improvement vs eager FSDP2 when composed with other techniques. |
| [AWS/Meta TorchTitan Llama 3 blog](https://aws.amazon.com/blogs/machine-learning/efficient-pre-training-of-llama-3-like-model-architectures-using-torchtitan-on-amazon-sagemaker/) | vendor-blog | 2024 | TorchTitan pretrains Llama-3-8B-like models with FSDP2, `torch.compile`, and FP8, showing 38.23% throughput speedup on H100. |
| [NVIDIA Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) | official-docs | 2026 | TE provides optimized Transformer blocks and FP8 support on Hopper/Ada/Blackwell, plus BF16/FP16 optimizations on Ampere and later. |
| [Megatron-LM README](https://github.com/NVIDIA/Megatron-LM) | code-repo | 2026 | Megatron-Core is a GPU-optimized training library with TP/PP/DP/CP/EP, BF16/FP8/FP4, and explicit communication-overlap optimizations. |
| [DeepSpeed training docs](https://www.deepspeed.ai/training/) | official-docs | 2026 | DeepSpeed/ZeRO focuses on memory, communication, and scale; it can combine ZeRO data parallelism with model parallelism for speed and scale. |
| [Unsloth multi-GPU docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) | official-docs | 2026 | Unsloth supports multi-GPU through Accelerate/DeepSpeed, but says the process is complex/manual and official multi-GPU support is still coming. |
| [FlashAttention paper](https://arxiv.org/abs/2205.14135) | paper | 2022 | FlashAttention trains Transformers faster by IO-aware exact attention, but it is an attention kernel, not a full Llama/FSDP replacement. |

(Read budget: 10 sources across paper, code-repo, official-docs, and vendor/practitioner blogs. Stopped because the answer was clear enough for a next experiment.)

## Findings

1. **Liger is the strongest same-shape baseline, not just a random kernel pack.** Its README names the same benchmark family as ours: Llama-3-8B, bf16, AdamW, gradient checkpointing, FSDP1, 8 A100s. The paper says the gain comes from fusion/chunking, matching our trace where memory/elementwise time fell sharply.

2. **The closest credible challenger is PyTorch-native FSDP2 + `torch.compile`/TorchTitan, not another one-line kernel.** PyTorch reports 10-23% MFU gains from compile on A100 7B/13B/34B/70B runs; SimpleFSDP attacks exactly our open bottleneck by tracing collectives for compute/communication overlap. This maps directly to our exposed-NCCL problem.

3. **Megatron-Core + Transformer Engine can probably beat Liger as an end-to-end training stack, but it changes the experiment.** Megatron brings tensor/pipeline/context parallelism and communication overlap; TE brings optimized transformer blocks and FP8 on newer GPUs. On A100 bf16, TE may still help, but the strongest TE story is Hopper+FP8, not our exact A100 bf16 FSDP1 row.

4. **FlashAttention/xFormers are unlikely to beat Liger alone in this trace.** Our trace is already dominated by GEMM, attention, and NCCL after Liger, and attention is only one slice. FlashAttention is essential tech, but swapping attention alone is not a full replacement for Liger's RMSNorm/SwiGLU/CE fusion and will not solve exposed FSDP collectives.

5. **Unsloth is not the next benchmark for this exact test.** Its public claims are strong for fine-tuning and memory efficiency, but its own docs say multi-GPU is still manual/complex. That makes it a poor immediate contender for 8xA100 full bf16 FSDP pretraining-step reproduction.

## Counter-evidence

The strongest counter-case is that PyTorch `torch.compile` and TorchTitan may already beat our Liger row if we port the test: PyTorch reports 7B A100 MFU rising from 57% to 68%, and SimpleFSDP claims compiler-visible collective overlap can reduce communication exposure. That is directly relevant because our Liger root trace still had fully exposed NCCL.

The weakness is apples-to-apples. Those sources are not the exact Llama-3-8B/HF/FSDP1/8xA100/Liger comparison. TorchTitan's Llama-3-8B blog result is H100 with FP8 features, and SimpleFSDP compares against eager FSDP2, not Liger+FSDP1. Also, the PyTorch torchtune+Liger blog says Liger composes with `torch.compile`; if compile helps, the best result may be **Liger plus compile**, not compile instead of Liger.

## Decision: ADAPT

Do not replace Liger yet. Adapt the benchmark matrix to test the closest credible challengers: Liger+`torch.compile` if feasible, FSDP2/TorchTitan-style compile, and only then Megatron-Core/Transformer Engine if we are willing to change model/runtime structure.

## What this means in practice

- **First concrete move:** Add a Swordfish row for PyTorch-native compile/FSDP2 or TorchTitan-style Llama-3-8B 8xA100, with the same steady-state NSYS overlap analysis and tokens/sec schema.
- **Watch-fors:** If compile/FSDP2 reduces exposed NCCL without regressing step time, it is a real challenger. If it only improves eager elementwise work, combine it with Liger instead of replacing Liger.
- **Out of scope for this research:** H100 FP8-only wins; inference-only kernels; LoRA-only or QLoRA-only fine-tuning; convergence/quality beyond parity checks.

## Open questions / what I'd read next

1. Does TorchTitan/FSDP2 currently support a close Llama-3-8B bf16 8xA100 config without H100-only FP8 assumptions?
2. Can Liger's monkey patch coexist with `torch.compile` for our exact FSDP runner without graph breaks?
3. Does Megatron-Core have a minimal Llama-3-8B BF16 A100 recipe whose checkpoint/model semantics are close enough to compare fairly?
