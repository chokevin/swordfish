# JD coverage tilt analysis (Week 1, Tuesday PM, Apr 29 2026)

> Compares the current 35-week roadmap (`docs/dashboard/app.js`) and the
> contributions ledger (`docs/contributions.md`) against the target role
> criteria in `jd-target-criteria.md`. Produces a ranked list of
> roadmap tilts.
>
> **Headline finding:** the role is **AMD-leaning** (HIP, Matrix Cores,
> ROCm Profiler, NVIDIA↔AMD porting). The current roadmap is
> NVIDIA-only. The single highest-leverage tilt is to escalate
> AMD/HIP/ROCm from "absent" to a primary track in the next 4 weeks.

## Coverage matrix

| JD bullet (R = required, P = preferred) | Current roadmap coverage | Confidence | Gap severity |
| --- | --- | --- | --- |
| **R1** High-perf kernel dev | W7-10 (matmul/DeepGEMM), W18-19 (FP4), W26-27 (vLLM PR) — mostly Triton/CUTLASS | Mid | **Medium** — heavy on Triton, light on hand-written CUDA C++. No HIP. |
| **R2** Perf optimization (BW / latency / divergence) | W1, W7-10, W17, W21 | High | Low — strong axis. |
| **R3** Workload analysis (kernel → app speedups) | Implicit in Liger first-touch (cross-arch e2e); W1 reanalysis insight not formalized | Mid | **Medium** — needs explicit methodology doc; the "wall-clock vs kernel-time" lesson is exactly this bullet. |
| **R4** Profiling tools (ROCm Profiler, PyTorch Profiler) | NCU + Nsys covered (W1, W17). PyTorch Profiler **absent**. ROCm Profiler **absent**. | Low | **High** — JD lists ROCm Profiler **first**. |
| **R5** Architecture adaptation (Matrix Cores, HBM) | W7-10 (Tensor Cores implicit), W14 (H200/HBM3e). Matrix Cores (AMD term) **absent**. | Mid | **High** — terminology and AMD-side architecture missing. |
| **R6** Framework integration | W11-12, W21, W26-27 (vLLM); Liger touchpoint advances this | High | Low — already present. |
| **P1** GPU arch mastery (SMs/CUs, mem hierarchy, warp/wavefront) | W7-10 implicit; CUs/wavefront (AMD) absent | Mid | **High** — explicit deep dives missing for both vendors. |
| **P2** C++ + CUDA/HIP kernel programming | W9 (DeepGEMM = CUTLASS = C++) is the first explicit C++ ship — **week 9.** | Low | **High** — too late. Need C++ kernel earlier. HIP entirely absent. |
| **P3** Performance engineering / low-level metrics | W1 NCU SOL, W7-10 perf | High | Low. |
| **P4** Async execution, stream mgmt, host-device transfers | W1 reanalysis touched it (workspace alloc, dispatch chain); not a tracked theme | Mid | **Medium** — worth a dedicated systems-sidecar week. |
| **P5** Triton + Python DSLs | Triton in target list; current backend; Liger touchpoint | High | Low. |
| **P6** Inference engines (vLLM / SGLang / TensorRT-LLM) | vLLM heavy. **SGLang absent. TensorRT-LLM absent.** | Mid | **High** — both missing-targets are JD-named. |
| **P7** PyTorch C++/CUDA extensions | Not in roadmap. | None | **Medium** — small, shippable artifact. |
| **P8** NVIDIA↔AMD porting | Not in roadmap. | None | **High** — JD-named preferred. |

## Tilts ranked by leverage / cost ratio

Each tilt is a *roadmap edit*, not a today-action. The recommendation is
to fold these into the appropriate Wn before they're scheduled, not to
re-plan every week now.

### Tilt 1 (highest leverage): escalate AMD/HIP/ROCm to a primary track

**Why now:** JD is AMD-leaning. The current 35-week plan does not
contain a single AMD week. The Liger first touchpoint already gives a
natural wedge — Liger has AMD CI; cross-vendor reproduction is the
JD's "Hardware Agnosticism" bullet.

**Concrete edits:**

- **W2-W3 (now-soon):** Acquire MI300X access. Realistic options
  ranked by cost: (a) Microsoft internal AMD testbed if accessible;
  (b) Hot Aisle / TensorWave bare-metal MI300X (~$2-3/hour);
  (c) ND-MI300X-v5 on Azure when GA; (d) AMD's developer cloud.
  Document what was tried and what worked in
  `docs/airun/amd-access-2026.md`.
- **W4-W5:** Liger cross-arch reproduction adds an MI300X column
  (single-GPU per-kernel sweep). Lands as a follow-up artifact to the
  W1 NVIDIA-only Liger Discussion.
- **W7-W8 (matmul learning sprints):** instead of "A100 matmul
  sprint" + "Hopper matmul sprint", split as "Hopper matmul sprint"
  + "**CDNA3 matmul sprint** (MI300X Matrix Cores, MFMA instructions,
  LDS/HBM hierarchy)". A100 already covered by W1 baseline.
- **W9-W10 (DeepGEMM reproduction):** add a parallel HIP track or a
  hipify pass over the CUTLASS reproduction so the artifact is
  cross-vendor.
- **Tools:** add ROCm Profiler (rocprof v3) wrapping to
  `swordfish.runner` alongside NCU. Result schema's `ncu` block becomes
  a generic `profiler` block.

**Cost:** non-trivial (compute access, learning curve on HIP/CDNA3),
but it converts a *missing requirement* into an *evidenced strength*.

**Risk:** If AMD compute access is unobtainable, this tilt cannot
ship. Falling back to "hipify-only / no AMD measurement" is theatre and
should not be done.

### Tilt 2: add SGLang and TensorRT-LLM to the contributions ledger

**Why:** JD lists vLLM, SGLang, TensorRT-LLM as the inference-engine
trio. Current ledger has only vLLM.

**Concrete edits:**

- Add two ledger rows:
  - **SGLang** — first touch is a Discussion or issue with cross-arch
    profiling data of SGLang's RadixAttention / structured-output paths
    on A100/H100 NVL/H200. Maintainers actively want cross-runtime
    perf data.
  - **TensorRT-LLM** — first touch is a documentation/repro PR or a
    benchmark gist comparing a single quantized GEMM through TRT-LLM's
    plugin path against the swordfish reference. Microsoft uses
    TRT-LLM heavily.
- Add `docs/notes/sglang-first-touch.md` and
  `docs/notes/tensorrt-llm-first-touch.md` modeled on existing
  first-touch notes.
- Reorder the existing weekly mentions: where current weeks say
  "vLLM," consider rotating to one of the three so the pattern is
  "vLLM-then-SGLang-then-TRT-LLM" across W11/W21/W26 etc.

**Cost:** low — two notes, two ledger rows, gentle weekly rotation.

### Tilt 3: ship one PyTorch C++/CUDA extension as a learning artifact

**Why:** JD's "Familiarity with writing custom extensions or operators
for PyTorch (C++/CUDA extensions)" is a concrete preferred skill. Not
demonstrating this is a clear interview-loop weakness.

**Concrete edits:**

- New W6-W7 sub-track: re-implement the existing Triton GEMM kernel
  as a `torch.utils.cpp_extension`-based C++/CUDA extension. Same
  shape, same result protocol JSON, registered as a custom torch op.
- Writeup at `docs/notes/torch-cpp-extension-gemm.md` documenting
  build system, autograd registration, dispatcher key, and any
  surprises.
- Adds a `swordfish/kernels/cuda/` directory mirroring the existing
  `swordfish/kernels/ptx/` and `swordfish/kernels/cute/` layout.

**Cost:** low-medium. Self-contained, no compute coordination required
beyond a CUDA host.

### Tilt 4: introduce CUDA C++ earlier

**Why:** Current roadmap's first explicit C++ ship is W9 (DeepGEMM /
CUTLASS reproduction). That is 8 weeks of mostly Python/Triton before
any C++ artifact. JD lists "Strong proficiency in C++" as preferred.
Move a C++ touchpoint into W3-W5.

**Concrete edits:**

- W3 sidecar: a tiny C++/CUDA vector-add or tiled GEMM v0 shipped via
  the same backend interface as Triton. Single file, single command,
  result JSON.
- The PyTorch C++/CUDA extension from Tilt 3 also satisfies this.

**Cost:** low. Mostly editing the W3-W5 weekly cells in `app.js` and
the SQL workweek table.

### Tilt 5: add a Workload-Analysis methodology doc

**Why:** JD bullet "ensuring that micro-optimizations translate to
application-level speedups" is exactly the W1 reanalysis insight
(GPU-kernel time vs wall-clock; the wrapper can dominate). This is a
swordfish-original methodology that should be a citable doc, not left
in a commit message.

**Concrete edits:**

- New `docs/methodology/workload-vs-kernel.md` documenting:
  - The two-clock taxonomy (wall-clock vs per-kernel via Perfetto/NCU).
  - The wrapper-overhead failure mode and how to detect it.
  - The bottleneck taxonomy with the "Wrapper / Python dispatch
    bound" category as a swordfish addition.
  - Required evidence for any swordfish performance claim (both clocks,
    explicit attribution).
- Cite this doc from every Friday writeup that makes a perf claim.

**Cost:** low. Mostly a writeup pass.

### Tilt 6: add explicit GPU architecture deep-dive weeks

**Why:** JD "GPU Architecture Mastery" is a preferred bullet. Current
roadmap covers it implicitly inside matmul sprints. Make it explicit.

**Concrete edits:**

- W8 (Hopper matmul sprint) → add explicit deliverable:
  `docs/research/hopper-deep-dive.md` covering TMA, async barriers,
  distributed shared memory, wgmma, and the L2 partition.
- W14 (H200 angle v1) → add `docs/research/h200-hbm3e.md` covering
  HBM3e bandwidth/capacity vs H100 and what it changes for kernel
  scheduling.
- New W14.5 sidecar (or fold into Tilt 1's CDNA3 sprint):
  `docs/research/cdna3-mi300x-deep-dive.md` covering CUs, wavefronts,
  LDS, MFMA instructions, HBM3, and Infinity Fabric.

**Cost:** low (mostly a writeup pass alongside existing weeks).

### Tilt 7: add PyTorch Profiler + ROCm Profiler tooling support

**Why:** JD names both. Current swordfish only wraps NCU and Nsys.

**Concrete edits:**

- Wrap `torch.profiler` (with on-device CUDA + CPU activity) in
  `swordfish.runner` for any benchmark that runs an actual model
  forward (Liger per-kernel sweep is the natural first user).
- Add `rocprof` (or rocprofv3) wrapping behind a backend flag, gated
  on AMD compute being available (Tilt 1 dependency).
- Generalize the result schema's `ncu` block to a `profiler` block
  with `tool` ∈ {ncu, nsys, torch_profiler, rocprof}.

**Cost:** low-medium. PyTorch Profiler is plug-and-play; rocprof gated
on Tilt 1.

## Tilts NOT recommended

- **VTune / perf coverage** — these are CPU profilers. Without a
  host-side bottleneck to chase, adding them is theatre. Revisit if
  the wrapper-overhead lane (W1 insight) ever needs deep CPU profiling.
- **TVM** — JD does not name it; mostly Apache/community; declining
  momentum vs vLLM/SGLang/TRT-LLM. Skip.
- **TGI** — JD removed it. Skip.

## Where the Liger first touchpoint sits in this analysis

The Liger Kernel cross-fleet profile chosen as the W1 first upstream
touchpoint **already advances** several JD bullets:

- **R3 Workload Analysis** — cross-arch e2e measurement that ties
  per-kernel to model-step performance.
- **R4 Profiling & Tuning** — NCU SOL fields per kernel.
- **R5 Architecture Adaptation** — A100 SXM4 vs H100 NVL vs H200,
  with HBM/compute/clock differences attributed in the writeup.
- **R6 Framework Integration** — Liger is itself a framework-integrated
  kernel library.
- **P5 Triton** — Liger kernels are Triton.
- **P6 Inference engines** — adjacency to vLLM/SGLang serving stacks
  via the trained model.
- **Tilt 1 wedge** — adding an AMD MI300X column to the Liger
  reproduction is the cleanest cross-vendor first artifact.

In other words, the Liger choice was even better-aligned than the
Tuesday-handoff justification claimed. This analysis does not change
the W1 touchpoint pick; it tilts W2-W10 to compound on it.
