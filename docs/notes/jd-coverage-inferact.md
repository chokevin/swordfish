# Inferact JD coverage analysis (Tuesday PM, Apr 29 2026)

> Analyzes the four currently-open Inferact roles
> (`https://jobs.ashbyhq.com/Inferact`) against swordfish's current shape
> and against the prior AMD-leaning JD captured in
> `jd-target-criteria.md`. Produces a delta on the W1 tilt analysis.

## The four open roles (Jan 2026, all SF, $200-400k + equity)

1. **MTS, Inference** — vLLM runtime engineer. KV-cache, prefix caching,
   hybrid serving, multimodal.
2. **MTS, Performance and Scale** — distributed systems for inference
   at scale. Rust/Go/C++. **Disaggregated inference architecture.**
   GPU interconnects (NVLink, InfiniBand, RoCE).
3. **MTS, Kernel Engineering** — CUDA / CuTeDSL / Triton / TileLang /
   **Pallas** kernels. Nsight + rocprof. Quantization (INT8, FP8,
   mixed-precision). Multi-vendor (NVIDIA, AMD, TPU, Intel). Compiler
   tech (LLVM, MLIR, XLA).
4. **MTS, Cloud Orchestration** — K8s + custom operators, multi-cloud
   GPU clusters, **Ray, Slurm**, vLLM deployment patterns.

Inferact is the company founded by the vLLM core (Simon Mo, Woosuk Kwon,
Kaichao You, Roger Wang, Joey Gonzalez, Ion Stoica). Mission: grow vLLM
as the world's AI inference engine.

## What Inferact indexes on (cross-cutting themes)

1. **vLLM-contributor lineage above everything else.** Three of four
   roles name vLLM in min-quals or as the central work. "Implemented
   core features in vLLM" and "kernel-related contributions to vLLM"
   show up as bonus points. Founders maintain the contributor graph;
   they will read GitHub before the résumé.
2. **Public technical writing is a measurable selection criterion.**
   "Written widely-shared technical blogs or side projects" appears as
   bonus on two of four roles (Inference, Kernel). Not decoration.
3. **Multi-vendor breadth, not depth.** Kernel role lists "NVIDIA,
   AMD, TPU, Intel" — all four. Prior JD wanted AMD depth; Inferact
   wants breadth across the accelerator zoo.
4. **vLLM-adjacent OSS ecosystem.** Bonus row literally names verl,
   OpenRLHF, Unsloth, LlamaFactory. These are the post-training /
   fine-tuning libraries that integrate with vLLM. **Liger Kernel** is
   adjacent (Liger integrates with Axolotl, LLaMA-Factory, TRL,
   torchtune); the W1 first touchpoint is already moving toward this
   neighborhood.
5. **Disaggregated inference + GPU interconnects** is an explicit
   Performance-role specialization (DistServe, Splitwise, MoonCake
   territory).
6. **Compiler tech (LLVM, MLIR, XLA)** for Kernel; **Pallas** appears
   alongside CUDA/Triton/TileLang/CuTeDSL. Kernel engineers are
   expected to be comfortable across multiple DSLs and the compilers
   under them.
7. **K8s operators + Ray + Slurm** for Orchestration — exactly the
   airun/Kueue stack swordfish already exercises.
8. **KV-cache, prefix caching, hybrid model serving** for the
   Inference role — the load-bearing engineering inside vLLM, not just
   kernels.

## Per-role swordfish fit

### Kernel Engineering — strongest fit

| JD bullet | Swordfish state | Gap |
| --- | --- | --- |
| CUDA / CuTeDSL / Triton / TileLang / Pallas | Triton current; CUTLASS first-touch; pyptx sidecar; CuTeDSL/TileLang/Pallas in target list | TileLang and Pallas are notes-only; no CuTeDSL; Tilt 4 covers C++/CUDA earlier |
| GPU arch (mem hierarchy, warp scheduling, tiling, tensor cores) | Implicit in W7-10 sprints | Make explicit per Tilt 6 |
| C++ + Python | C++ planned at W9; tilt to W3-W5 | Tilt 4 |
| Nsight + rocprof | NCU/Nsys yes; rocprof needs Tilt 7 + Tilt 1 | Tilt 7 |
| ML-specific kernel optimization (FlashAttention, fused) | Not in roadmap; Liger touch is adjacent | New |
| Quantization (INT8, FP8, mixed-precision) | Lane decision (FP8/FP4/INT4) — well aligned | None |
| Multi-platform (NVIDIA, AMD, TPU, Intel) | NVIDIA-only; Tilt 1 adds AMD; TPU/Intel out | Add Pallas-on-TPU sidecar (cheap), skip Intel |
| Compiler tech (LLVM, MLIR, XLA) | Not in roadmap | New gap |
| Bonus: kernel contributions to vLLM | W11/12/26/27 — too late for Inferact-shaped target | Pull earlier (see below) |
| Bonus: deep technical blogs | W20, W28 only | Cadence too low (see below) |

### Inference (runtime) — partial fit

| JD bullet | Swordfish state | Gap |
| --- | --- | --- |
| Transformer architectures + variants | W7-10 implicit | None |
| Python + PyTorch internals | Implicit | OK |
| vLLM / TRT-LLM / SGLang / TGI | vLLM yes; others gap (Tilt 2 adds SGLang + TRT-LLM) | TGI not worth chasing |
| Read+implement from papers | Implicit | OK |
| KV-cache mgmt, prefix caching, hybrid model serving | Absent | **Major gap** — no current artifact |
| RL frameworks for LLMs | Absent; Liger has post-training (DPO/ORPO/etc.), adjacent | Could leverage Liger sweep to demonstrate adjacency |
| Multimodal inference | Absent | Major gap |
| Bonus: vLLM-integration projects (verl, OpenRLHF, Unsloth, LlamaFactory) | Not in roadmap; Liger is adjacent | New ledger rows |

### Cloud Orchestration — surprisingly close

| JD bullet | Swordfish state | Gap |
| --- | --- | --- |
| K8s + container orchestration at scale | airun = Kueue + DRA + GPU debug already lived | Strong |
| Custom K8s operators | airun has no CRD; **`kaito-project/airunway` does** | Possible wedge — contribute to airunway |
| Python/Rust/Go + Terraform/Helm | Helm yes; Terraform gap; no Rust/Go yet | Add Terraform sidecar |
| GPU cluster mgmt + hardware debug | DCGM/NCU permission, H200 capacity blockers — exactly this | Strong |
| Multi-cloud (AWS/GCP/Azure) + on-prem | Azure voice-agent-flex only | Multi-cloud sidecar realistic only at week ≥20 |
| Ray, Slurm | Kueue covered; Ray partial via airun infra; Slurm absent | Add Ray serving demo |
| vLLM deployment patterns | Not yet | Becomes natural after vLLM contributions land |
| 1000+ GPU clusters | Out of scope | Skip |

### Performance and Scale — weakest fit

Almost everything missing: no Rust, no Go, no distributed systems, no
interconnect work, no disaggregated inference. **If this role were the
target**, the swordfish lane would need to materially redirect — not
just tilt. Recommend treating this role as not-the-target for now.

## Delta vs the prior AMD-leaning JD

| Dimension | Prior JD | Inferact | Delta for swordfish |
| --- | --- | --- | --- |
| Vendor depth vs breadth | AMD depth (HIP, Matrix Cores, ROCm Profiler first) | Multi-vendor breadth (NVIDIA, AMD, TPU, Intel) | **AMD becomes one vendor in a multi-vendor story, not the headline.** Tilt 1 still useful, less dominant. |
| vLLM positioning | vLLM = one of several inference engines | vLLM = the company | **vLLM contribution cadence becomes the dominant tilt.** |
| OSS ecosystem | Generic "open-source contributions" | Specific list (verl, OpenRLHF, Unsloth, LlamaFactory) | **Track these as named ecosystem targets.** Liger touchpoint is adjacent. |
| Public writing | Mentioned implicitly | Bonus on 2 of 4 roles | **Increase writing cadence; make Friday writeups public-facing.** |
| Disaggregated inference / interconnects | Absent | Performance role specialization | **Optional sidecar; not a primary track.** |
| Compiler tech (LLVM/MLIR/XLA) | Absent | Kernel role preferred | **Sidecar week reading MLIR / Triton compilation pipeline.** |
| Pallas | Marked "skip" in `jax-pallas-first-touch.md` | Listed alongside CUDA/Triton | **Reopen Pallas decision; cheap small artifact unlocks TPU breadth.** |

## New / revised tilts (delta on the W1 tilt analysis)

### Inferact-tilt A: pull vLLM contribution cadence forward

**Current:** first vLLM PR at W11-W12, major at W26-W27.

**Revised:** small vLLM PR by W4-W5 (one well-scoped issue or docs PR
landed); second by W8; first kernel-related vLLM PR by W12; major by
W26. Track each PR in `docs/contributions.md` and the dashboard.

**Why:** Inferact founders maintain the vLLM contributor graph. Three
PRs by W12 puts the user on their radar 6+ months earlier than the
current plan.

### Inferact-tilt B: increase public writing cadence

**Current:** 2 blog posts in 35 weeks (W20, W28).

**Revised:** at least one *publicly-published* writeup per month
(blog, gist, GitHub Discussion, or Twitter/X thread with diagrams).
The Friday writeups already exist as internal docs; the tilt is to
publish 4-6 of them with additional context-setting and edits.

**Why:** Inferact bonus-points line on 2 of 4 roles. Public writing
volume is itself a hiring signal.

### Inferact-tilt C: track vLLM-adjacent OSS ecosystem as named targets

**Current:** ledger has Triton/PyTorch-Inductor/CUTLASS/vLLM/ORT/pyptx
+ Liger.

**Revised:** add ledger rows (planned, no scheduled artifact yet) for
**verl, OpenRLHF, Unsloth, LlamaFactory**. These are the post-training
/ fine-tuning ecosystems Inferact explicitly names. Liger touchpoint
already lives in this neighborhood (Liger integrates with Axolotl,
LLaMA-Factory, TRL, torchtune); the W1 artifact may naturally surface
maintainer overlap with these projects.

### Inferact-tilt D: reopen the Pallas decision

**Current:** `jax-pallas-first-touch.md` says skip Pallas unless
quantized GEMM repro available.

**Revised:** Inferact lists Pallas alongside CUDA/Triton/TileLang/CuTeDSL
as kernel-DSL options. A small Pallas first-touch artifact (single
GEMM, JAX reference, swordfish result protocol) is now JD-justified
even at fp16. Schedule as a low-cost W6-W8 sidecar.

### Inferact-tilt E: contribute to airunway as Cloud-Orchestration wedge

**Current:** airun is local infra; airunway not in scope.

**Revised:** airunway (kaito-project) is a custom K8s operator wrapping
KAITO/KubeRay/Dynamo/LLM-D for vLLM-class deployments. A docs PR or
small operator-side contribution reproducing a vLLM ModelDeployment on
the AKS fleet directly maps to **K8s operators + GPU clusters + vLLM
deployment patterns** — three Cloud-Orchestration JD bullets in one
artifact. Schedule as a W9-W12 sidecar.

### Inferact-tilt F: compiler tech sidecar (MLIR / Triton compilation pipeline)

**Current:** Not in roadmap.

**Revised:** one sidecar week reading Triton's frontend → MLIR →
LLVM → PTX pipeline, with one writeup that traces a single Triton
kernel through each lowering stage. Optional follow-up: read XLA's
Pallas lowering for the TPU side.

**Why:** Kernel role lists LLVM/MLIR/XLA as preferred. A "I traced a
kernel through the Triton compiler" writeup is itself a public-writing
artifact (Inferact-tilt B) and a JD-bullet check.

### Inferact-tilt G: optional disaggregated-inference experiment

**Current:** Not in roadmap.

**Revised:** *if* the user is targeting the Performance-and-Scale role
specifically, schedule a W14-W16 experiment reproducing a
prefill/decode split (DistServe-style) on the AKS fleet using vLLM's
existing disaggregated serving primitives. *If not*, treat as
out-of-scope; the Performance role is the weakest swordfish fit and
chasing it requires a real lane redirect.

## Recommendation

If Inferact is the dominant target, swordfish should:

1. **Adopt Inferact-tilts A (vLLM cadence) and B (writing cadence) immediately** —
   these are the two highest-leverage signals for any Inferact role.
2. **Adopt Inferact-tilts C (OSS ecosystem targets) and E (airunway) by end of W4** —
   small ledger / contribution edits that light up multiple JD bullets.
3. **Adopt Inferact-tilt D (Pallas) and F (compiler tech) as W6-W12 sidecars.**
4. **Demote (not cancel) prior Tilt 1 (AMD primary track)** — AMD remains
   one vendor in the multi-vendor breadth story, but is no longer the
   headline tilt.
5. **Skip Performance-and-Scale role unless a real lane redirect is on
   the table.** The fit is too thin.

If both the prior AMD-leaning JD *and* Inferact are real targets, the
overlap is large (vLLM kernel work, multi-vendor breadth, profiling,
quantization). The two JDs share enough core ground that a single
swordfish lane satisfies most of both. The main divergence is **AMD
depth (prior JD)** vs **vLLM lineage + writing cadence (Inferact)** —
both can be served, but the weekly emphasis differs.
