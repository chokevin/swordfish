# Target role criteria (external JD)

This file captures the external job description that the swordfish roadmap
is being optimized to satisfy. It is the **scoring rubric** for whether
the roadmap is on track. When a week's plan is being written, check it
against this file: if a week does not advance any of the bullets below,
ask whether it should be tilted.

The tilt analysis comparing the current 35-week roadmap to this rubric is
in [`docs/notes/jd-coverage-tilt-2026-w1.md`](jd-coverage-tilt-2026-w1.md).

## Key responsibilities

1. **High-Performance Kernel Development** — Design, implement, and
   optimize high-performance GPU kernels for AI/ML workloads to maximize
   hardware utilization.
2. **Performance Optimization** — Analyze and optimize kernel execution
   for latency and throughput, addressing bottlenecks in memory bandwidth,
   instruction latency, and thread divergence.
3. **Workload Analysis** — Evaluate the end-to-end performance impact of
   individual kernels on full-stack AI models, ensuring that
   micro-optimizations translate to application-level speedups.
4. **Profiling & Tuning** — Utilize advanced GPU profiling tools (e.g.,
   ROCm Profiler, PyTorch Profiler) to identify performance cliffs, stall
   pipelines, and memory hierarchy inefficiencies.
5. **Architecture Adaptation** — Tailor implementation strategies to
   leverage specific features of modern GPU architectures (e.g., Matrix
   Cores, HBM characteristics).
6. **Framework Integration** — Collaborate with software stack teams to
   expose optimized kernels within high-level frameworks and inference
   engines.

## Preferred experience

- **GPU Architecture Mastery** — Modern GPU underlying architectures
  including streaming multiprocessors (SMs/CUs), memory hierarchy
  (registers, shared memory, L1/L2 cache, HBM), and warp/wavefront
  execution models.
- **Kernel Programming Expertise** — Strong proficiency in C++ and
  parallel computing, with extensive hands-on experience in **NVIDIA
  CUDA or AMD HIP** kernel programming.
- **Performance Engineering** — Demonstrated ability to debug and
  profile complex GPU workloads, interpreting low-level metrics to drive
  architectural-aware optimizations.
- **Systems Knowledge** — Familiarity with asynchronous execution,
  stream management, and host-device memory transfers.
- **Python DSLs & Triton** — Experience implementing kernels using
  OpenAI Triton or other Python-based DSLs for agile kernel development
  and auto-tuning.
- **Inference Engine Experience** — Hands-on experience integrating
  custom kernels into large-scale inference frameworks such as **vLLM,
  SGLang, or TensorRT-LLM**.
- **Deep Learning Frameworks** — Familiarity with writing custom
  extensions or operators for **PyTorch (C++/CUDA extensions)**.
- **Hardware Agnosticism** — Experience porting kernels between NVIDIA
  and AMD architectures or working with cross-platform HPC libraries.

## What this means for swordfish positioning

- The role is materially **AMD-leaning** (HIP listed alongside CUDA,
  Matrix Cores terminology, ROCm Profiler listed first, hardware
  agnosticism preferred). ROCm/HIP must move from "nice-to-have
  sidecar" to a **primary track**.
- The inference-engine target list converges on **vLLM, SGLang,
  TensorRT-LLM**. TGI is out. TVM is out. The current contributions
  ledger needs SGLang and TensorRT-LLM rows added.
- **PyTorch C++/CUDA extensions** is a concrete, named preferred skill
  with a small artifact size. Worth shipping at least one such
  extension in swordfish as a learning artifact (a Triton kernel
  re-implemented as a `torch.utils.cpp_extension` op with the same
  result protocol).
- The Workload Analysis bullet — "ensuring that micro-optimizations
  translate to application-level speedups" — is exactly the W1
  reanalysis insight (kernel-time vs wall-clock). This should be
  documented as an explicit swordfish methodology, not left implicit
  in a single commit message.

## How this guides the weekly cadence

Each Friday writeup should be able to point at one or more of these
bullets and say "this artifact advances the bullet because X." If a
week ships nothing that maps to a Key Responsibility bullet, that's a
signal the roadmap is drifting and needs a tilt.
